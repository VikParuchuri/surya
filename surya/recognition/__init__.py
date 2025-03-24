from copy import deepcopy
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import DynamicCache

from surya.common.surya import SuryaModelConfig
from surya.common.util import mark_step
from surya.common.predictor import BasePredictor
from surya.detection import DetectionPredictor
from surya.input.processing import convert_if_not_rgb, slice_polys_from_image, slice_bboxes_from_image
from surya.layout import prediction_to_polygon
from surya.recognition.loader import RecognitionModelLoader
from surya.recognition.postprocessing import truncate_repetitions, replace_invalid_tags
from surya.recognition.util import sort_text_lines, clean_close_polygons
from surya.recognition.schema import TextLine, OCRResult, TextChar, TaskNames
from surya.settings import settings


class RecognitionPredictor(BasePredictor):
    model_loader_cls = RecognitionModelLoader
    batch_size = settings.RECOGNITION_BATCH_SIZE
    torch_dtype = settings.MODEL_DTYPE_BFLOAT
    default_batch_sizes = {
        "cpu": 32,
        "mps": 64,
        "cuda": 256,
        "xla": 128
    }
    tasks = {
        TaskNames.ocr_with_boxes: {
            "needs_bboxes": True,
            "img_size": (1024, 256)
        },
        TaskNames.ocr_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 256)
        },
        TaskNames.block_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 768)
        }
    }

    def __call__(
            self,
            images: List[Image.Image],
            task_names: List[str] | None = None,
            det_predictor: DetectionPredictor | None = None,
            detection_batch_size: int | None = None,
            recognition_batch_size: int | None = None,
            highres_images: List[Image.Image] | None = None,
            bboxes: List[List[List[int]]] | None = None,
            polygons: List[List[List[List[int]]]] | None = None,
            input_text: List[str | None] | None = None,
            sort_lines: bool = True
    ) -> List[OCRResult]:
        allowed_tasks = self.tasks.keys()
        if task_names is None:
            task_names = [TaskNames.ocr_with_boxes] * len(images)

        assert all([task_name in allowed_tasks for task_name in task_names]), f"One or more tasks in {task_names} is not supported. Supported tasks are {allowed_tasks}"
        assert len(images) == len(task_names), "You need to pass in one task name for each image"

        images = convert_if_not_rgb(images)
        if highres_images is not None:
            assert len(images) == len(highres_images), "You need to pass in one highres image for each image"

        highres_images = convert_if_not_rgb(highres_images) if highres_images is not None else [None] * len(images)

        if bboxes is None and polygons is None:
            assert det_predictor is not None, "You need to pass in a detection predictor if you don't provide bboxes or polygons"

            # Detect then slice
            flat = self.detect_and_slice_bboxes(
                images,
                task_names,
                det_predictor,
                detection_batch_size=detection_batch_size,
                highres_images=highres_images,
            )
        else:
            if bboxes is not None:
                assert len(images) == len(bboxes), "You need to pass in one list of bboxes for each image"
            if polygons is not None:
                assert len(images) == len(polygons), "You need to pass in one list of polygons for each image"

            flat = self.slice_bboxes(
                images,
                bboxes=bboxes,
                polygons=polygons,
                input_text=input_text,
                task_names=task_names
            )

        char_predictions = self.batch_recognition(
            flat["slices"],
            flat["task_names"],
            input_text=flat["input_text"],
            batch_size=recognition_batch_size
        )

        predictions_by_image = []
        slice_start = 0
        for idx, image in enumerate(images):
            slice_end = slice_start + flat["slice_map"][idx]
            image_lines = char_predictions[slice_start:slice_end]
            polygons = flat["polygons"][slice_start:slice_end]
            slice_start = slice_end

            lines = []
            for text_line, polygon in zip(image_lines, polygons):
                # Special case when input text is good
                if not text_line:
                    lines.append(TextLine(
                        text="",
                        polygon=polygon,
                        chars=[],
                        confidence=1,
                        original_text_good=True
                    ))
                else:
                    text = "".join([char.text for char in text_line])
                    confidence = float(np.mean([char.confidence for char in text_line]))
                    lines.append(TextLine(
                        text=text,
                        polygon=polygon,
                        chars=text_line,
                        confidence=confidence,
                    ))

            if sort_lines:
                lines = sort_text_lines(lines)
            predictions_by_image.append(OCRResult(
                text_lines=lines,
                image_bbox=[0, 0, image.size[0], image.size[1]]
            ))

        return predictions_by_image

    def detect_and_slice_bboxes(
        self,
        images: List[Image.Image],
        task_names: List[str],
        det_predictor: DetectionPredictor,
        detection_batch_size: int | None = None,
        highres_images: List[Image.Image] | None = None,
    ):
        det_predictions = det_predictor(images, batch_size=detection_batch_size)

        all_slices = []
        slice_map = []
        all_polygons = []
        all_task_names = []

        for idx, (det_pred, image, highres_image, task_name) in enumerate(zip(det_predictions, images, highres_images, task_names)):
            polygons = [p.polygon for p in det_pred.bboxes]
            if highres_image:
                width_scaler = highres_image.size[0] / image.size[0]
                height_scaler = highres_image.size[1] / image.size[1]
                scaled_polygons = [[[int(p[0] * width_scaler), int(p[1] * height_scaler)] for p in polygon] for
                                   polygon in polygons]
                slices = slice_polys_from_image(highres_image, scaled_polygons)
            else:
                slices = slice_polys_from_image(image, polygons)
            slice_map.append(len(slices))
            all_slices.extend(slices)
            all_polygons.extend(polygons)
            all_task_names.extend([task_name] * len(slices))


        assert len(all_slices) == sum(slice_map) == len(all_polygons) == len(all_task_names)

        return {
            "slices": all_slices,
            "slice_map": slice_map,
            "polygons": all_polygons,
            "task_names": all_task_names,
            "input_text": [None] * len(all_slices)
        }

    def slice_bboxes(
        self,
        images: List[Image.Image],
        task_names: List[str],
        bboxes: List[List[List[int]]] | None = None,
        polygons: List[List[List[List[int]]]] | None = None,
        input_text: List[List[str | None]] | None = None
    ):
        assert bboxes is not None or polygons is not None
        slice_map = []
        all_slices = []
        all_polygons = []
        all_text = []
        all_task_names = []

        for idx, image in enumerate(images):
            if polygons is not None:
                polys = polygons[idx]
                slices = slice_polys_from_image(image, polys)
            else:
                slices = slice_bboxes_from_image(image, bboxes[idx])
                polys = [
                    [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                    for bbox in bboxes[idx]
                ]
            slice_map.append(len(slices))
            all_slices.extend(slices)
            all_polygons.extend(polys)
            all_task_names.extend([task_names[idx]] * len(slices))

            if input_text is None:
                all_text.extend([None] * len(slices))
            else:
                all_text.extend(input_text[idx])


        assert len(all_slices) == sum(slice_map)  == len(all_polygons) == len(all_text) == len(all_task_names)

        return {
            "slices": all_slices,
            "slice_map": slice_map,
            "polygons": all_polygons,
            "input_text": all_text,
            "task_names": all_task_names
        }

    def prepare_input(self, task_names: List[str], images: List[Image.Image], input_text: List[str | None]):
        batch = []
        for (image, text, task_name) in zip(images, input_text, task_names):
            image_size = self.tasks[task_name]["img_size"]
            image = image.resize(image_size)

            # Task input is the same for all tasks for now
            text = text or ""
            inputs = [{"type": "image", "image": image}, {"type": "text", "text": text}]
            batch.append({"task": task_name, "inputs": inputs})

        return batch

    def batch_recognition(
        self,
        images: List[Image.Image],
        task_names: List[str],
        input_text: List[str | None],
        batch_size: int | None = None
    ) -> List[List[TextChar]]:
        assert all(isinstance(image, Image.Image) for image in images)

        if len(images) == 0:
            return []

        assert len(images) == len(task_names) == len(input_text), "You need to pass in one task name and text line for each image"

        if batch_size is None:
            batch_size = self.get_batch_size()

        # Sort images by width, so similar length ones go together, sort input text to match
        sorted_data = sorted(enumerate(zip(images, input_text)), key=lambda x: x[1][0].width)
        indices, pairs = zip(*sorted_data)
        indices = list(indices)
        images, input_text = map(list, zip(*pairs))

        config: SuryaModelConfig = self.model.config

        # Setup various tokens on-device
        device_bbox_ignore = torch.from_numpy(
            np.array(self.processor.ignore_bbox_token_ids, dtype=np.int64)
        ).to(self.model.device)
        device_blank_bbox = torch.from_numpy(
            np.asarray([config.blank_bbox_token_id] * 6)
        ).to(self.model.device).to(torch.long)
        device_pad_token = torch.tensor(self.processor.pad_token_id, device=self.model.device, dtype=torch.long)

        output_text = []
        for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text", disable=self.disable_tqdm):
            batch_images = images[i:i + batch_size]
            batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images
            batch_text = input_text[i:i + batch_size]
            batch_task_names = task_names[i:i + batch_size]
            current_batch_size = len(batch_images)

            batch_input = self.prepare_input(
                batch_task_names,
                batch_images,
                batch_text
            )
            processed_inputs = self.processor(batch_input, padding_side="left").to(
                device=self.model.device
            )
            input_ids = processed_inputs["input_ids"].to(torch.long)
            input_boxes = processed_inputs["input_boxes"].to(torch.long)
            image_tiles = processed_inputs["image_tiles"].to(self.model.dtype)
            attention_mask = processed_inputs["attention_mask"].to(torch.long)
            position_ids = processed_inputs["position_ids"].to(torch.long)

            needs_boxes = [self.tasks[task_name]["needs_bboxes"] for task_name in batch_task_names]
            skip_box_idxs = ~torch.from_numpy(np.array(needs_boxes)).to(self.model.device)

            token_count = 0
            inference_token_count = 1

            # Batch pixel values is the real current batch size
            sequence_scores = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device).unsqueeze(1)
            all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device)

            # Setup tensors to hold predictions
            batch_predictions = torch.zeros((current_batch_size, 1), dtype=torch.long, device=self.model.device)
            batch_box_predictions = torch.zeros((current_batch_size, 1, 6), dtype=torch.long, device=self.model.device)

            past_key_values = DynamicCache()
            attention_ones = torch.ones(current_batch_size, 1, device=self.model.device, dtype=torch.long)

            with settings.INFERENCE_MODE():
                while token_count < settings.RECOGNITION_MAX_TOKENS - 1:
                    outputs = self.model(
                        input_ids=input_ids,
                        input_boxes=input_boxes,
                        image_tiles=image_tiles,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        inputs_embeds=None,
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                    # Update cache
                    past_key_values = outputs["past_key_values"]

                    # Get logits and initial preds
                    next_token_logits = outputs["lm_logits"][:, -1:, :].clone().float()
                    next_bbox_logits = outputs["bbox_logits"][:, -1:, :].clone().float()
                    preds = torch.argmax(next_token_logits, dim=-1)

                    # Handle math sections
                    is_special_token = (preds < self.processor.ocr_tokenizer.qwen_offset)  | (torch.isin(preds, device_bbox_ignore))

                    # Handle inference completion
                    done = (preds == self.processor.eos_token_id) | (preds == self.processor.pad_token_id)
                    all_done = all_done | done.squeeze(-1)
                    all_done_cpu = all_done.cpu()

                    # Account for possible padding of batch
                    if all_done_cpu[:current_batch_size].all():
                        break

                    # Update input ids
                    input_ids = preds

                    # Confidence score for the current token
                    scores = torch.max(F.softmax(next_token_logits[:, -1], dim=-1), dim=-1).values
                    scores = scores.masked_fill(all_done, 0).unsqueeze(1)
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                    # Update input boxes
                    box_preds = next_bbox_logits * self.model.config.bbox_size
                    expanded_blank_bbox = device_blank_bbox.expand(box_preds.shape)
                    box_preds = torch.where(
                        torch.isin(preds, device_bbox_ignore).unsqueeze(-1),
                        expanded_blank_bbox,
                        box_preds
                    )

                    # Set bbox to blank if we're in a match section
                    box_preds = torch.where(
                        is_special_token.unsqueeze(-1),
                        expanded_blank_bbox,
                        box_preds
                    )
                    input_boxes = box_preds.to(torch.long)
                    input_boxes[skip_box_idxs, -1] = device_blank_bbox # Set blank for tasks that don't need boxes

                    # Update output sequences
                    batch_predictions = torch.cat([batch_predictions, input_ids], dim=1)
                    batch_box_predictions = torch.cat([batch_box_predictions, input_boxes], dim=1)

                    # Updates for next iteration
                    # If this batch item is done, input a pad token
                    input_ids = torch.where(all_done.unsqueeze(1), device_pad_token, input_ids).to(torch.long)
                    position_ids = position_ids[:, -1:] + 1
                    attention_mask = torch.cat(
                        [attention_mask, attention_ones], dim=1
                    )
                    image_tiles = None # Remove after prefill

                    # Update token counts and mark XLA step
                    token_count += inference_token_count
                    inference_token_count = input_ids.shape[-1]
                    mark_step()

            sequence_scores = sequence_scores.cpu()[:current_batch_size, 1:].tolist()
            batch_predictions = batch_predictions.cpu()[:current_batch_size, 1:].tolist() # Remove the start token
            batch_bboxes = batch_box_predictions.cpu()[:current_batch_size, 1:]

            assert len(batch_predictions) == len(batch_images) == len(batch_bboxes) == len(sequence_scores) == len(batch_task_names), f"Batch size mismatch, {len(batch_predictions)} != {len(batch_images)} != {len(batch_bboxes)} != {len(sequence_scores)} != {len(batch_task_names)}"
            detected_polygons = []
            for img, bboxes in zip(batch_images, batch_bboxes):
                detected_polygons.append([
                    prediction_to_polygon(bbox, img.size, config.bbox_size, config.bbox_size // 2)
                    for bbox in bboxes
                ])

            detected_chars = []
            for (poly, pred, seq_score, needs_box, task_name) in zip(detected_polygons, batch_predictions, sequence_scores, needs_boxes, batch_task_names):

                # Special case when no output
                if self.processor.no_output_token in pred:
                    detected_chars.append(None)
                    continue

                detokenize_sequences = []
                detokenize_sequence = []
                past_char_qwen_token = False

                def _add_detokenize_sequence(qwen_token: bool, past_char_qwen_token: bool, special_token: bool, past_special_token: bool, force: bool = False):
                    nonlocal detokenize_sequence, detokenize_sequences

                    if (qwen_token != past_char_qwen_token or force or special_token or past_special_token) and detokenize_sequence:
                        chars = [dt[0] for dt in detokenize_sequence]
                        scores = mean([dt[1] for dt in detokenize_sequence]) if detokenize_sequence else 0
                        bboxes = [dt[2] for dt in detokenize_sequence]

                        if past_char_qwen_token:
                            detokenize_sequences.append((chars, scores, None, "qwen"))
                        elif past_special_token:
                            detokenize_sequences.append((chars, scores, None, "special"))
                        else:
                            detokenize_sequences.append((chars, scores, bboxes, "ocr"))

                        detokenize_sequence = []

                # Split up into sequences to detokenize separately
                past_special_token = False
                for bbox, char_id, score in zip(poly, pred, seq_score):
                    if char_id in [self.processor.eos_token_id, self.processor.pad_token_id]:
                        break

                    qwen_token = char_id < self.processor.ocr_tokenizer.qwen_offset
                    special_token = self.processor.ocr_tokenizer.qwen_offset <= char_id < self.processor.ocr_tokenizer.special_token_offset
                    _add_detokenize_sequence(qwen_token, past_char_qwen_token, special_token, past_special_token)
                    detokenize_sequence.append((char_id, score, bbox))
                    past_char_qwen_token = qwen_token
                    past_special_token = special_token

                _add_detokenize_sequence(False, past_char_qwen_token, False, past_special_token, force=True)

                img_chars = []
                for sequence in detokenize_sequences:
                    token_ids, seq_score, bboxes, token_type = sequence
                    blank_bbox = [[0, 0], [0, 1], [1, 1], [1, 0]]
                    if token_type == "ocr":
                        text = self.processor.ocr_tokenizer.decode(token_ids, task="ocr_with_boxes")
                        bboxes = clean_close_polygons(bboxes) # clean out bboxes that are close, like what happens with multiple utf-16 tokens per char
                        bbox_idx = 0
                        for text_idx, text_line in enumerate(text):
                            img_chars.append(TextChar(
                                text=text_line,
                                polygon=bboxes[bbox_idx],
                                confidence=seq_score,
                                bbox_valid=True
                            ))

                            # Ensure we don't exceed the bbox count
                            if bbox_idx < len(bboxes) - 1:
                                bbox_idx += 1
                    elif token_type == "special":
                        text = self.processor.ocr_tokenizer.decode(token_ids, task="ocr_without_boxes")
                        img_chars.append(TextChar(
                            text=text,
                            polygon=blank_bbox,
                            confidence=seq_score,
                            bbox_valid=False
                        ))
                    else:
                        text = self.processor.ocr_tokenizer.decode(token_ids, task="block_without_boxes")
                        img_chars.append(TextChar(
                            text=text,
                            polygon=blank_bbox,
                            confidence=seq_score,
                            bbox_valid=False
                        ))

                detected_chars.append(img_chars)

            # Convert sequence_scores to list for the current batch
            output_text.extend(detected_chars)

        output_text = sorted(zip(indices, output_text), key=lambda x: x[0])
        output_text = [text for _, text in output_text]
        return output_text