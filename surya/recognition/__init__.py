from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import DynamicCache

from surya.common.surya import SuryaModelConfig, SuryaModelOutput
from surya.common.util import mark_step
from surya.common.predictor import BasePredictor
from surya.detection import DetectionPredictor
from surya.input.processing import convert_if_not_rgb, slice_polys_from_image, slice_bboxes_from_image
from surya.layout import prediction_to_polygon
from surya.recognition.loader import RecognitionModelLoader
from surya.recognition.postprocessing import truncate_repetitions, replace_invalid_tags
from surya.recognition.util import sort_text_lines
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
        device_math_start = torch.from_numpy(np.array(self.processor.math_start_token_ids, dtype=np.int64)).to(self.model.device)
        device_math_end = torch.from_numpy(np.array(self.processor.math_end_token_ids, dtype=np.int64)).to(self.model.device)

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
                device=self.model.device,
                dtype=self.model.dtype
            )
            input_ids = processed_inputs["input_ids"]
            input_boxes = processed_inputs["input_boxes"]
            image_tiles = processed_inputs["image_tiles"]
            attention_mask = processed_inputs["attention_mask"]
            position_ids = processed_inputs["position_ids"]

            needs_boxes = [self.tasks[task_name]["needs_bboxes"] for task_name in batch_task_names]
            skip_box_idxs = ~torch.from_numpy(np.array(needs_boxes)).to(self.model.device)

            token_count = 0
            inference_token_count = 1

            # Batch pixel values is the real current batch size
            sequence_scores = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device).unsqueeze(1)
            all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device)
            in_math = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device)

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
                        use_cache=True,
                        logits_to_keep=1
                    )

                    # Get logits and initial preds
                    next_token_logits = outputs["lm_logits"][:, -1:, :].clone().float()
                    next_bbox_logits = outputs["bbox_logits"][:, -1:, :].clone().float()
                    preds = torch.argmax(next_token_logits, dim=-1)

                    # Handle math sections
                    math_start = torch.isin(preds, device_math_start).squeeze(-1)
                    in_math = in_math | math_start
                    math_end = torch.isin(preds, device_math_end).squeeze(-1)
                    in_math = in_math & ~math_end

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
                        in_math.unsqueeze(-1).unsqueeze(-1),
                        expanded_blank_bbox,
                        box_preds
                    )
                    input_boxes = box_preds.to(torch.long)
                    input_boxes[skip_box_idxs, -1] = device_blank_bbox

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

            assert len(batch_predictions) == len(batch_images) == len(batch_bboxes) == len(sequence_scores), f"Batch size mismatch, {len(batch_predictions)} != {len(batch_images)} != {len(batch_bboxes)}"
            detected_polygons = []
            for img, bboxes in zip(batch_images, batch_bboxes):
                detected_polygons.append([
                    prediction_to_polygon(bbox, img.size, config.bbox_size, config.bbox_size // 2)
                    for bbox in bboxes
                ])

            detected_chars = []
            for (poly, pred, seq_score, needs_box) in zip(detected_polygons, batch_predictions, sequence_scores, needs_boxes):
                img_chars = []
                assert len(poly) == len(pred) == len(seq_score), f"Prediction mismatch found, {len(poly)} != {len(pred)} != {len(seq_score)}"
                for bbox, char_id, score in zip(poly, pred, seq_score):

                    # Special case when input text is good, don't overwrite
                    if char_id == self.processor.no_output_token:
                        img_chars = None
                        break

                    if char_id == self.processor.eos_token_id:
                        break

                    if not needs_box:
                        bbox = [[0, 0], [0, 1], [1, 1], [1, 0]]

                    img_chars.append(TextChar(
                        text=self.processor.decode([char_id]),
                        polygon=bbox,
                        confidence=score,
                        bbox_valid=needs_box
                    ))

                # Cleanup tags that aren't properly balanced
                #img_chars = replace_invalid_tags(img_chars, self.model.config.special_ocr_tokens)
                detected_chars.append(img_chars)

            # Convert sequence_scores to list for the current batch
            output_text.extend(detected_chars)

        output_text = sorted(zip(indices, output_text), key=lambda x: x[0])
        output_text = [text for _, text in output_text]
        return output_text

class ContinuousBatchingCache(DynamicCache):
    def pad_left(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        padding_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Size is assumed to be (batch_size, num_kv_heads, seq_length, head_dim) - To match huggingface
        key_padding = torch.zeros((key_states.shape[0], key_states.shape[1], padding_size, key_states.shape[3]), device=key_states.device, dtype=key_states.dtype)
        key_states_padded = torch.cat([key_padding, key_states], dim=-2)  # Pad along the sequence length dimension (dim=-2)

        # Pad value_states to the left by `padding_size`
        value_padding = torch.zeros((value_states.shape[0], value_states.shape[1], padding_size, value_states.shape[3]), device=value_states.device, dtype=value_states.dtype)
        value_states_padded = torch.cat([value_padding, value_states], dim=-2)  # Pad along the sequence length dimension (dim=-2)

        return key_states_padded, value_states_padded

    def merge(
        self, 
        new_cache: DynamicCache,
        merge_idxs: List[int]
    ):
        assert len(new_cache) == len(self), "The two caches should have the same number of layers"
        
        # We should TECHNICALLY be able to pad these values to 0s now, since they will be attention masked
        current_seq_length = self.get_seq_length()
        new_cache_seq_length = new_cache.get_seq_length()
        offset = current_seq_length - new_cache_seq_length      # Generally positive, but negative case is handled too
        with torch.inference_mode():
            # As long as we set the attention mask and position ids correctly, padding value can be anything
            for layer_idx in range(len(self)):
                new_k, new_v = new_cache[layer_idx]
                if offset > 0:
                    new_k, new_v = self.pad_left(new_k, new_v, offset)

                if offset < 0:
                    adjusted_key_cache, adjusted_value_cache = self.pad_left(self.key_cache[layer_idx], self.value_cache[layer_idx], abs(offset))
                else:
                    adjusted_key_cache, adjusted_value_cache = self.key_cache[layer_idx], self.value_cache[layer_idx]

                for i, merge_idx in enumerate(merge_idxs):
                    adjusted_key_cache[merge_idx] = new_k[i]
                    adjusted_value_cache[merge_idx] = new_v[i]

                    self.key_cache[layer_idx] = adjusted_key_cache
                    self.value_cache[layer_idx] = adjusted_value_cache

        return offset
        
@dataclass
class ContinuousBatchInput:
    input_ids: torch.Tensor
    input_boxes: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    in_math: torch.Tensor
    skip_box_idxs: torch.Tensor

@dataclass
class ContinuousBatchOutput:
    input_ids: torch.Tensor
    input_boxes: torch.Tensor
    preds: torch.Tensor
    bbox_preds: torch.Tensor
    math_start: torch.Tensor
    math_end: torch.Tensor
    done: torch.Tensor
    scores: torch.Tensor

@dataclass
class RecognitionPrompt:
    id: int
    task_name: TaskNames
    image: Image
    text: str

# TODO When we evict a sample (WHICH IS STILL NOT IMPLEMENTED YOU DUMBFUCK) then also trim the cache as necessary from the left
# This will also allow require modification of attention mask (position_ids should remain unchanged, since they are only for the last element)
class ContinuousBatchingRecognitionPredictor(RecognitionPredictor):
    
    min_prefill_ratio: int = 0.2

    def __init__(self, checkpoint = None, device = settings.TORCH_DEVICE_MODEL, dtype = None):
        super().__init__(checkpoint, device, dtype)
        self.kv_cache = None
        self.prompt_queue = deque()
        self.batch_prompt_mapping = None

        config: SuryaModelConfig = self.model.config
        # Setup various tokens on-device
        self.device_bbox_ignore = torch.from_numpy(
            np.array(self.processor.ignore_bbox_token_ids, dtype=np.int64)
        ).to(self.model.device)
        self.device_blank_bbox = torch.from_numpy(
            np.asarray([config.blank_bbox_token_id] * 6)
        ).to(self.model.device).to(torch.long)
        self.device_pad_token = torch.tensor(self.processor.pad_token_id, device=self.model.device, dtype=torch.long)
        self.device_math_start = torch.from_numpy(np.array(self.processor.math_start_token_ids, dtype=np.int64)).to(self.model.device)
        self.device_math_end = torch.from_numpy(np.array(self.processor.math_end_token_ids, dtype=np.int64)).to(self.model.device)

    def setup_cache(self, batch_size: int):
        self.kv_cache = None
        self.prompt_queue.clear()
        self.batch_prompt_mapping = {i: None for i in range(batch_size)}

    @property
    def num_empty_slots(self):
        return sum(v is None for v in self.batch_prompt_mapping.values())

    @property
    def num_active_slots(self):
        return len(self.batch_prompt_mapping) - self.num_empty_slots

    def process_outputs(self, outputs: SuryaModelOutput, skip_box_idxs: torch.Tensor, in_math: torch.BoolTensor) -> ContinuousBatchOutput:
        # Get logits and initial preds
        next_token_logits = outputs["lm_logits"][:, -1:, :].clone().float()
        next_bbox_logits = outputs["bbox_logits"][:, -1:, :].clone().float()
        preds = torch.argmax(next_token_logits, dim=-1)

        # Handle math sections
        math_start = torch.isin(preds, self.device_math_start).squeeze(-1)
        math_end = torch.isin(preds, self.device_math_end).squeeze(-1)

        # Handle inference completion
        done = (preds == self.processor.eos_token_id) | (preds == self.processor.pad_token_id)
        done = done.squeeze(-1)
        # If this batch item is done, input a pad token
        input_ids = torch.where(done.unsqueeze(1), self.device_pad_token, preds).to(torch.long)

        # Confidence score for the current token
        scores = torch.max(F.softmax(next_token_logits[:, -1], dim=-1), dim=-1).values
        scores = scores.masked_fill(done, 0).unsqueeze(1)

        # Update input boxes
        box_preds = next_bbox_logits * self.model.config.bbox_size
        expanded_blank_bbox = self.device_blank_bbox.expand(box_preds.shape)
        box_preds = torch.where(
            torch.isin(preds, self.device_bbox_ignore).unsqueeze(-1),
            expanded_blank_bbox,
            box_preds
        )
        # Set bbox to blank if we're in a match section
        box_preds = torch.where(
            in_math.unsqueeze(-1).unsqueeze(-1),
            expanded_blank_bbox,
            box_preds
        )
        input_boxes = box_preds.to(torch.long)
        input_boxes[skip_box_idxs, -1] = self.device_blank_bbox

        return ContinuousBatchOutput(
            input_ids=input_ids,
            input_boxes=input_boxes,
            preds=preds,
            bbox_preds=input_boxes,
            math_start=math_start,
            math_end=math_end,
            done=done,
            scores=scores
        )

    def decode(
        self,
        current_inputs: Optional[ContinuousBatchInput] = None
    ):
        input_ids = current_inputs.input_ids
        input_boxes = current_inputs.input_boxes
        attention_mask = current_inputs.attention_mask
        position_ids = current_inputs.position_ids
        skip_box_idxs = current_inputs.skip_box_idxs
        in_math = current_inputs.in_math

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                input_boxes=input_boxes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=self.kv_cache,
                num_logits_to_keep=1
            )

        processed_output: ContinuousBatchOutput = self.process_outputs(outputs, skip_box_idxs=skip_box_idxs, in_math=in_math)
        in_math = in_math | processed_output.math_start
        in_math = in_math & ~processed_output.math_end

        attention_mask = torch.cat(
            [attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=torch.long, device=attention_mask.device)], dim=1
        )
        position_ids = position_ids[:, -1:] + 1
        new_input = ContinuousBatchInput(
            input_ids=processed_output.input_ids,
            input_boxes=processed_output.input_boxes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            in_math=in_math,
            skip_box_idxs=skip_box_idxs
        )

        return new_input, processed_output

    def prefill(
            self,
            current_inputs: Optional[ContinuousBatchInput] = None
        ):

        prompts: List[RecognitionPrompt] = [self.prompt_queue.popleft() for _ in range(min(self.num_empty_slots, len(self.prompt_queue)))]
        
        batch_input = self.prepare_input(
            task_names=[p.task_name for p in prompts],
            images=[p.image for p in prompts],
            input_text=[p.text for p in prompts]
        )
        processed_inputs = self.processor(batch_input, padding_side="left").to(
            device=self.model.device,
            dtype=self.model.dtype
        )
        input_ids = processed_inputs["input_ids"]
        input_boxes = processed_inputs["input_boxes"]
        image_tiles = processed_inputs["image_tiles"]
        attention_mask = processed_inputs["attention_mask"]
        position_ids = processed_inputs["position_ids"]
        needs_boxes = [self.tasks[p.task_name]["needs_bboxes"] for p in prompts]
        skip_box_idxs = ~torch.from_numpy(np.array(needs_boxes)).to(self.model.device)
        in_math = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.model.device)

        prefill_cache = ContinuousBatchingCache()

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                input_boxes=input_boxes,
                image_tiles=image_tiles,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                past_key_values=prefill_cache,
                use_cache=True,
                num_logits_to_keep=1
            )

        # Process outputs
        processed_outputs = self.process_outputs(outputs, skip_box_idxs=skip_box_idxs, in_math=in_math)
        in_math = in_math | processed_outputs.math_start
        in_math = in_math & ~processed_outputs.math_end

        # Merge new kv cache with existing, update batch mapping
        non_active_idxs = [k for k,v in self.batch_prompt_mapping.items() if v is None]
        idxs_to_merge = non_active_idxs[:len(prompts)]
        for i, prompt in zip(idxs_to_merge, prompts):
            self.batch_prompt_mapping[i] = prompt.id

        if self.kv_cache:
            offset = self.kv_cache.merge(prefill_cache, idxs_to_merge)
        else:
            self.kv_cache = prefill_cache
            offset = 0


        # Adjust attention mask and position ids to account for the newly generated tokens
        attention_mask = torch.cat(
            [attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=torch.long, device=attention_mask.device)], dim=1
        )
        position_ids = position_ids[:, -1:] + 1

        if current_inputs is None:
            new_input = ContinuousBatchInput(
                input_ids=processed_outputs.input_ids,
                input_boxes=processed_outputs.input_boxes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                skip_box_idxs=skip_box_idxs,
                in_math=in_math
            )

            return new_input, processed_outputs, range(processed_outputs.input_ids.shape[0])

        # Merging input_ids, input_boxes, attention masks and position ids
        current_input_ids = current_inputs.input_ids
        current_input_ids[idxs_to_merge] = processed_outputs.input_ids
        
        current_input_boxes = current_inputs.input_boxes
        current_input_boxes[idxs_to_merge] = processed_outputs.input_boxes

        current_attention_mask = current_inputs.attention_mask
        if offset > 0:
            attention_mask = F.pad(attention_mask, (offset, 0), value=0)
        elif offset < 0:
            current_attention_mask = F.pad(current_attention_mask, (abs(offset), 0), value=0)
        current_attention_mask[idxs_to_merge] = attention_mask

        current_position_ids = current_inputs.position_ids
        current_position_ids[idxs_to_merge] = position_ids

        current_skip_box_idxs = current_inputs.skip_box_idxs
        current_skip_box_idxs[idxs_to_merge] = skip_box_idxs

        current_in_math = current_inputs.in_math
        current_in_math[idxs_to_merge] = in_math

        new_input = ContinuousBatchInput(
            input_ids=current_input_ids,
            input_boxes=current_input_boxes,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids,
            skip_box_idxs=current_skip_box_idxs,
            in_math=current_in_math
        )

        return new_input, processed_outputs, idxs_to_merge

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

        predicted_tokens = [[] for _ in range(len(flat['slices']))]
        predicted_boxes = [[] for _ in range(len(flat['slices']))]
        scores = [[] for _ in range(len(flat['slices']))]

        if recognition_batch_size is None:
            recognition_batch_size = self.get_batch_size()
        current_inputs = None
        self.setup_cache(recognition_batch_size)
        for idx, (img, txt, task) in enumerate(zip(flat['slices'], flat['input_text'], flat['task_names'])):
            self.prompt_queue.append(RecognitionPrompt(
                id=idx,
                task_name=task,
                text=txt,
                image=img
            ))

        pbar = tqdm(total=len(self.prompt_queue), desc='Recognizing Text')
        while self.prompt_queue or self.num_active_slots > 0:
            if (self.num_empty_slots / recognition_batch_size) > self.min_prefill_ratio and self.prompt_queue:
                updated_inputs, outputs, merge_idxs = self.prefill(current_inputs)

                for temp_idx, b_idx in enumerate(merge_idxs):
                    if self.batch_prompt_mapping[b_idx]:
                        p_idx = self.batch_prompt_mapping[b_idx]
                        predicted_tokens[p_idx].append(outputs.preds[temp_idx].cpu().item())
                        predicted_boxes[p_idx].append(outputs.bbox_preds[temp_idx].cpu()[0])
                        scores[p_idx].append(outputs.scores[temp_idx].cpu().item())

                        if predicted_tokens[p_idx][-1] in [self.processor.eos_token_id, self.processor.pad_token_id]:
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)
            else:
                updated_inputs, outputs = self.decode(current_inputs)

                # TODO Find a cleaner way of popping from the dict
                for b_idx, p_idx in self.batch_prompt_mapping.items():
                    if p_idx is not None:
                        predicted_tokens[p_idx].append(outputs.preds[b_idx].cpu().item())
                        predicted_boxes[p_idx].append(outputs.bbox_preds[b_idx].cpu()[0])
                        scores[p_idx].append(outputs.scores[b_idx].cpu().item())

                        if predicted_tokens[p_idx][-1] in [self.processor.eos_token_id, self.processor.pad_token_id] or len(predicted_tokens[p_idx]) == settings.RECOGNITION_MAX_TOKENS:
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)

            current_inputs = updated_inputs
        pbar.close()

        char_predictions = []
        needs_boxes = [self.tasks[task_name]["needs_bboxes"] for task_name in flat['task_names']]
        bbox_size = self.model.config.bbox_size
        for slice_idx, (slice_image, image_tokens, image_boxes, image_scores, needs_box) in enumerate(zip(flat['slices'], predicted_tokens, predicted_boxes, scores, needs_boxes)):
            image_polygons = [prediction_to_polygon(bbox, slice_image.size, bbox_size, bbox_size // 2) for bbox in image_boxes]
            img_chars = []
            for bbox, char_id, score in zip(image_polygons, image_tokens, image_scores):
                # Special case when input text is good, don't overwrite
                if char_id == self.processor.no_output_token:
                    img_chars = None
                    break

                if char_id == self.processor.eos_token_id:
                    break

                if not needs_box:
                    bbox = [[0, 0], [0, 1], [1, 1], [1, 0]]

                img_chars.append(TextChar(
                    text=self.processor.decode([char_id]),
                    polygon=bbox,
                    confidence=score,
                    bbox_valid=needs_box
                ))

            char_predictions.append(img_chars)

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