from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from collections import deque

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from surya.common.polygon import PolygonBox
from surya.common.surya import SuryaModelConfig, SuryaModelOutput
from surya.common.surya.processor import NOMATH_TOKEN
from surya.common.util import mark_step
from surya.common.predictor import BasePredictor
from surya.detection import DetectionPredictor
from surya.input.processing import (
    convert_if_not_rgb,
    slice_polys_from_image,
    slice_bboxes_from_image,
)
from surya.layout import prediction_to_polygon
from surya.recognition.loader import RecognitionModelLoader
from surya.recognition.postprocessing import fix_unbalanced_tags
from surya.recognition.util import (
    sort_text_lines,
    clean_close_polygons,
    words_from_chars,
)
from surya.recognition.schema import TextLine, OCRResult, TextChar
from surya.common.surya.schema import TaskNames
from surya.recognition.cache import ContinuousBatchingCache
from surya.settings import settings


@dataclass
class ContinuousBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    skip_box_idxs: torch.Tensor


@dataclass
class ContinuousBatchOutput:
    input_ids: torch.Tensor
    preds: torch.Tensor
    bbox_preds: torch.Tensor
    done: torch.Tensor
    scores: torch.Tensor


@dataclass
class RecognitionPrompt:
    id: int
    task_name: TaskNames
    image: Image
    text: str
    math_mode: bool


# TODO When we evict a sample then also trim the cache as necessary from the left
# This will also allow require modification of attention mask (position_ids should remain unchanged, since they are only for the last element)
class RecognitionPredictor(BasePredictor):
    model_loader_cls = RecognitionModelLoader
    batch_size = settings.RECOGNITION_BATCH_SIZE
    torch_dtype = settings.MODEL_DTYPE_BFLOAT
    default_batch_sizes = {"cpu": 32, "mps": 64, "cuda": 256, "xla": 128}
    min_prefill_ratio: int = 0.2
    tasks = {
        TaskNames.ocr_with_boxes: {
            "needs_bboxes": True,
            "img_size": (1024, 256),
            "max_tokens": 400,
        },
        TaskNames.ocr_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 256),
            "max_tokens": 256,
        },
        TaskNames.block_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 768),
            "max_tokens": 1024,
        },
    }

    def __init__(self, checkpoint=None, device=settings.TORCH_DEVICE_MODEL, dtype=None):
        super().__init__(checkpoint, device, dtype)
        self.kv_cache = None
        self.prompt_queue = deque()
        self.batch_prompt_mapping = None

        config: SuryaModelConfig = self.model.config
        # Setup various tokens on-device
        self.device_bbox_ignore = torch.from_numpy(
            np.array(self.processor.ignore_bbox_token_ids, dtype=np.int64)
        ).to(self.model.device)
        self.device_blank_bbox = (
            torch.from_numpy(np.asarray([config.blank_bbox_token_id] * 6))
            .to(self.model.device)
            .to(torch.long)
        )
        self.device_pad_token = torch.tensor(
            self.processor.pad_token_id, device=self.model.device, dtype=torch.long
        )
        self.device_math_start = torch.from_numpy(
            np.array(self.processor.math_start_token_ids, dtype=np.int64)
        ).to(self.model.device)
        self.device_math_end = torch.from_numpy(
            np.array(self.processor.math_end_token_ids, dtype=np.int64)
        ).to(self.model.device)

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
        all_res_scales = []

        for idx, (det_pred, image, highres_image, task_name) in enumerate(
            zip(det_predictions, images, highres_images, task_names)
        ):
            polygons = [p.polygon for p in det_pred.bboxes]
            if highres_image:
                width_scaler = highres_image.size[0] / image.size[0]
                height_scaler = highres_image.size[1] / image.size[1]
                scaled_polygons = [
                    [
                        [int(p[0] * width_scaler), int(p[1] * height_scaler)]
                        for p in polygon
                    ]
                    for polygon in polygons
                ]
                slices = slice_polys_from_image(highres_image, scaled_polygons)
                res_scales = [(width_scaler, height_scaler) for _ in range(len(slices))]
            else:
                slices = slice_polys_from_image(image, polygons)
                res_scales = [(1, 1) for _ in range(len(slices))]

            slice_map.append(len(slices))
            all_slices.extend(slices)
            all_polygons.extend(polygons)
            all_task_names.extend([task_name] * len(slices))
            all_res_scales.extend(res_scales)

        assert (
            len(all_slices)
            == sum(slice_map)
            == len(all_polygons)
            == len(all_task_names)
            == len(all_res_scales)
        )

        return {
            "slices": all_slices,
            "slice_map": slice_map,
            "polygons": all_polygons,
            "task_names": all_task_names,
            "input_text": [None] * len(all_slices),
            "res_scales": all_res_scales,
        }

    def slice_bboxes(
        self,
        images: List[Image.Image],
        task_names: List[str],
        bboxes: List[List[List[int]]] | None = None,
        polygons: List[List[List[List[int]]]] | None = None,
        input_text: List[List[str | None]] | None = None,
    ) -> dict:
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
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                    ]
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

        assert (
            len(all_slices)
            == sum(slice_map)
            == len(all_polygons)
            == len(all_text)
            == len(all_task_names)
        )

        return {
            "slices": all_slices,
            "slice_map": slice_map,
            "polygons": all_polygons,
            "input_text": all_text,
            "task_names": all_task_names,
            "res_scales": [(1, 1) for _ in range(len(all_slices))],
        }

    def prepare_input(
        self,
        task_names: List[str],
        images: List[Image.Image],
        input_text: List[str | None],
        math_modes: List[bool],
    ):
        batch = []
        for image, text, task_name, math_mode in zip(
            images, input_text, task_names, math_modes
        ):
            image_size = self.tasks[task_name]["img_size"]
            image, rotated = self.processor.align_long_axis(image)
            image = image.resize(image_size)

            # Task input is the same for all tasks for now
            text = text or ""
            inputs = [
                {"type": "image", "image": image, "rotated": rotated},
                {"type": "text", "text": text, "math": math_mode},
            ]
            batch.append({"task": task_name, "inputs": inputs})

        return batch

    def process_outputs(
        self, outputs: SuryaModelOutput, skip_box_idxs: torch.Tensor
    ) -> ContinuousBatchOutput:
        # Get logits and initial preds
        next_token_logits = outputs["lm_logits"][:, -1:, :].clone().float()
        next_bbox_logits = outputs["bbox_logits"][:, -1:, :].clone().float()
        preds = torch.argmax(next_token_logits, dim=-1)

        is_special_token = (preds < self.processor.ocr_tokenizer.qwen_offset) | (
            torch.isin(preds, self.device_bbox_ignore)
        )

        # Handle inference completion
        done = (preds == self.processor.eos_token_id) | (
            preds == self.processor.pad_token_id
        )
        done = done.squeeze(-1)
        # If this batch item is done, input a pad token
        input_ids = torch.where(done.unsqueeze(1), self.device_pad_token, preds).to(
            torch.long
        )

        # Confidence score for the current token
        scores = torch.max(F.softmax(next_token_logits[:, -1], dim=-1), dim=-1).values
        scores = scores.masked_fill(done, 0).unsqueeze(1)

        # Update input boxes
        box_preds = next_bbox_logits * self.model.config.bbox_size
        expanded_blank_bbox = self.device_blank_bbox.expand(box_preds.shape)
        box_preds = torch.where(
            torch.isin(preds, self.device_bbox_ignore).unsqueeze(-1),
            expanded_blank_bbox,
            box_preds,
        )
        # Set bbox to blank if we're in a math section
        box_preds = torch.where(
            is_special_token.unsqueeze(-1), expanded_blank_bbox, box_preds
        )
        input_boxes = box_preds.to(torch.long)

        # Set blank for tasks that don't need boxes
        input_boxes[skip_box_idxs, -1] = self.device_blank_bbox

        return ContinuousBatchOutput(
            input_ids=input_ids,
            preds=preds,
            bbox_preds=input_boxes,
            done=done,
            scores=scores,
        )

    def decode(self, current_inputs: Optional[ContinuousBatchInput] = None):
        input_ids = current_inputs.input_ids
        attention_mask = current_inputs.attention_mask
        position_ids = current_inputs.position_ids
        skip_box_idxs = current_inputs.skip_box_idxs

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=self.kv_cache,
                logits_to_keep=1,
            )

        processed_output: ContinuousBatchOutput = self.process_outputs(
            outputs, skip_box_idxs=skip_box_idxs
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=torch.long,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        position_ids = position_ids[:, -1:] + 1
        new_input = ContinuousBatchInput(
            input_ids=processed_output.input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            skip_box_idxs=skip_box_idxs,
        )

        return new_input, processed_output

    def prefill(self, current_inputs: Optional[ContinuousBatchInput] = None):
        prompts: List[RecognitionPrompt] = [
            self.prompt_queue.popleft()
            for _ in range(min(self.num_empty_slots, len(self.prompt_queue)))
        ]

        batch_input = self.prepare_input(
            task_names=[p.task_name for p in prompts],
            images=[p.image for p in prompts],
            input_text=[p.text for p in prompts],
            math_modes=[
                p.math_mode for p in prompts
            ],  # Pass math mode to the processor
        )
        processed_inputs = self.processor(batch_input, padding_side="left").to(
            device=self.model.device, dtype=self.model.dtype
        )
        input_ids = processed_inputs["input_ids"]
        image_tiles = processed_inputs["image_tiles"]
        attention_mask = processed_inputs["attention_mask"]
        position_ids = processed_inputs["position_ids"]
        needs_boxes = [self.tasks[p.task_name]["needs_bboxes"] for p in prompts]
        skip_box_idxs = ~torch.from_numpy(np.array(needs_boxes)).to(self.model.device)

        prefill_cache = ContinuousBatchingCache()

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                image_tiles=image_tiles,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                past_key_values=prefill_cache,
                use_cache=True,
                logits_to_keep=1,
            )

        # Process outputs
        processed_outputs = self.process_outputs(outputs, skip_box_idxs=skip_box_idxs)

        # Merge new kv cache with existing, update batch mapping
        non_active_idxs = [k for k, v in self.batch_prompt_mapping.items() if v is None]
        idxs_to_merge = non_active_idxs[: len(prompts)]

        assert len(idxs_to_merge) == len(prompts), (
            "Number of prompts should match number of empty slots"
        )
        for i, prompt in zip(idxs_to_merge, prompts):
            self.batch_prompt_mapping[i] = prompt.id

        if self.kv_cache:
            offset = self.kv_cache.merge(prefill_cache, idxs_to_merge)
        else:
            self.kv_cache = prefill_cache
            offset = 0

        # Adjust attention mask and position ids to account for the newly generated tokens
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=torch.long,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        position_ids = position_ids[:, -1:] + 1

        if current_inputs is None:
            new_input = ContinuousBatchInput(
                input_ids=processed_outputs.input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                skip_box_idxs=skip_box_idxs,
            )

            return (
                new_input,
                processed_outputs,
                range(processed_outputs.input_ids.shape[0]),
            )

        # Merging input_ids, attention masks and position ids
        current_input_ids = current_inputs.input_ids
        current_input_ids[idxs_to_merge] = processed_outputs.input_ids

        current_attention_mask = current_inputs.attention_mask
        if offset > 0:
            attention_mask = F.pad(attention_mask, (offset, 0), value=0)
        elif offset < 0:
            current_attention_mask = F.pad(
                current_attention_mask, (abs(offset), 0), value=0
            )
        current_attention_mask[idxs_to_merge] = attention_mask

        current_position_ids = current_inputs.position_ids
        current_position_ids[idxs_to_merge] = position_ids

        current_skip_box_idxs = current_inputs.skip_box_idxs
        current_skip_box_idxs[idxs_to_merge] = skip_box_idxs

        new_input = ContinuousBatchInput(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids,
            skip_box_idxs=current_skip_box_idxs,
        )

        return new_input, processed_outputs, idxs_to_merge

    # Due to continuous batching, we left pad the attention mask and cache to match new sequences
    # This function trims the attention mask and the kv cache from the left whenever possible to remove excess padding
    def maybe_trim_cache_padding(self, current_inputs: ContinuousBatchInput):
        attention_mask = current_inputs.attention_mask
        active_idxs = [k for k, v in self.batch_prompt_mapping.items() if v is not None]

        # No more samples running
        if not active_idxs:
            return current_inputs

        active_attention_mask = attention_mask[active_idxs]
        first_non_padding_idx = (active_attention_mask == 1).to(torch.int).argmax(dim=1)
        trim_start = first_non_padding_idx.min().item()

        if trim_start == 0:
            return current_inputs

        trimmed_attention_mask = attention_mask[:, trim_start:]
        current_inputs.attention_mask = trimmed_attention_mask

        # Trim the cache accordingly
        if self.kv_cache:
            self.kv_cache.trim_left(trim_start)

        return current_inputs

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
        sort_lines: bool = True,
        math_mode: bool = True,
    ) -> List[OCRResult]:
        allowed_tasks = self.tasks.keys()
        if task_names is None:
            task_names = [TaskNames.ocr_with_boxes] * len(images)

        assert all([task_name in allowed_tasks for task_name in task_names]), (
            f"One or more tasks in {task_names} is not supported. Supported tasks are {allowed_tasks}"
        )
        assert len(images) == len(task_names), (
            "You need to pass in one task name for each image"
        )

        images = convert_if_not_rgb(images)
        if highres_images is not None:
            assert len(images) == len(highres_images), (
                "You need to pass in one highres image for each image"
            )

        highres_images = (
            convert_if_not_rgb(highres_images)
            if highres_images is not None
            else [None] * len(images)
        )

        if bboxes is None and polygons is None:
            assert det_predictor is not None, (
                "You need to pass in a detection predictor if you don't provide bboxes or polygons"
            )

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
                assert len(images) == len(bboxes), (
                    "You need to pass in one list of bboxes for each image"
                )
            if polygons is not None:
                assert len(images) == len(polygons), (
                    "You need to pass in one list of polygons for each image"
                )

            flat = self.slice_bboxes(
                images,
                bboxes=bboxes,
                polygons=polygons,
                input_text=input_text,
                task_names=task_names,
            )

        # No images passed, or no boxes passed, or no text detected in the images
        if len(flat["slices"]) == 0:
            return []

        # Sort by line widths. Negative so that longer images come first, fits in with continuous batching better
        sorted_pairs = sorted(enumerate(flat["slices"]), key=lambda x: -x[1].width)
        indices, sorted_slices = zip(*sorted_pairs)

        # Reorder input_text and task_names based on the new order
        flat["slices"] = list(sorted_slices)
        flat["input_text"] = [flat["input_text"][i] for i in indices]
        flat["task_names"] = [flat["task_names"][i] for i in indices]

        predicted_tokens = [[] for _ in range(len(flat["slices"]))]
        predicted_boxes = [[] for _ in range(len(flat["slices"]))]
        scores = [[] for _ in range(len(flat["slices"]))]

        if recognition_batch_size is None:
            recognition_batch_size = self.get_batch_size()
        current_inputs = None
        self.setup_cache(recognition_batch_size)

        batch_max_tokens = {}
        for idx, (img, txt, task) in enumerate(
            zip(flat["slices"], flat["input_text"], flat["task_names"])
        ):
            self.prompt_queue.append(
                RecognitionPrompt(
                    id=idx, task_name=task, text=txt, image=img, math_mode=math_mode
                )
            )
            batch_max_tokens[idx] = (
                settings.RECOGNITION_MAX_TOKENS or self.tasks[task]["max_tokens"]
            )

        pbar = tqdm(total=len(self.prompt_queue), desc="Recognizing Text")
        while self.prompt_queue or self.num_active_slots > 0:
            if (
                self.num_empty_slots / recognition_batch_size
            ) > self.min_prefill_ratio and self.prompt_queue:
                updated_inputs, outputs, merge_idxs = self.prefill(current_inputs)

                for temp_idx, b_idx in enumerate(merge_idxs):
                    if self.batch_prompt_mapping[b_idx] is not None:
                        p_idx = self.batch_prompt_mapping[b_idx]
                        predicted_tokens[p_idx].append(
                            outputs.preds[temp_idx].cpu().item()
                        )
                        predicted_boxes[p_idx].append(
                            outputs.bbox_preds[temp_idx].cpu()[0]
                        )
                        scores[p_idx].append(outputs.scores[temp_idx].cpu().item())

                        if predicted_tokens[p_idx][-1] in [
                            self.processor.eos_token_id,
                            self.processor.no_output_token,
                        ]:
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)
            else:
                updated_inputs, outputs = self.decode(current_inputs)

                # TODO Find a cleaner way of popping from the dict
                for b_idx, p_idx in self.batch_prompt_mapping.items():
                    if p_idx is not None:
                        predicted_tokens[p_idx].append(
                            outputs.preds[b_idx].cpu().item()
                        )
                        predicted_boxes[p_idx].append(
                            outputs.bbox_preds[b_idx].cpu()[0]
                        )
                        scores[p_idx].append(outputs.scores[b_idx].cpu().item())

                        if (
                            predicted_tokens[p_idx][-1]
                            in [
                                self.processor.eos_token_id,
                                self.processor.pad_token_id,
                            ]
                            or len(predicted_tokens[p_idx]) >= batch_max_tokens[p_idx]
                        ):
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)

            # Update inputs and mark XLA step
            current_inputs = updated_inputs
            current_inputs = self.maybe_trim_cache_padding(current_inputs)
            mark_step()
        pbar.close()

        char_predictions = []
        needs_boxes = [
            self.tasks[task_name]["needs_bboxes"] for task_name in flat["task_names"]
        ]
        bbox_size = self.model.config.bbox_size

        for slice_idx, (
            slice_image,
            image_tokens,
            image_boxes,
            image_scores,
            needs_box,
        ) in enumerate(
            zip(flat["slices"], predicted_tokens, predicted_boxes, scores, needs_boxes)
        ):
            image_polygons = [
                prediction_to_polygon(bbox, slice_image.size, bbox_size, bbox_size // 2)
                for bbox in image_boxes
            ]

            if self.processor.no_output_token in image_tokens:
                char_predictions.append(None)
                continue

            detokenize_sequences = []
            detokenize_sequence = []
            past_char_qwen_token = False

            def _add_detokenize_sequence(
                qwen_token: bool,
                past_char_qwen_token: bool,
                special_token: bool,
                past_special_token: bool,
                force: bool = False,
            ):
                nonlocal detokenize_sequence, detokenize_sequences

                if (
                    qwen_token != past_char_qwen_token
                    or force
                    or special_token
                    or past_special_token
                ) and detokenize_sequence:
                    chars = [dt[0] for dt in detokenize_sequence]
                    scores = [dt[1] for dt in detokenize_sequence]
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
            for bbox, char_id, score in zip(image_polygons, image_tokens, image_scores):
                if char_id in [
                    self.processor.eos_token_id,
                    self.processor.pad_token_id,
                ]:
                    break

                qwen_token = char_id < self.processor.ocr_tokenizer.qwen_offset
                special_token = (
                    self.processor.ocr_tokenizer.qwen_offset
                    <= char_id
                    < self.processor.ocr_tokenizer.special_token_offset
                )
                _add_detokenize_sequence(
                    qwen_token, past_char_qwen_token, special_token, past_special_token
                )
                detokenize_sequence.append((char_id, score, bbox))
                past_char_qwen_token = qwen_token
                past_special_token = special_token

            _add_detokenize_sequence(
                False, past_char_qwen_token, False, past_special_token, force=True
            )

            img_chars = []
            for sequence in detokenize_sequences:
                token_ids, seq_score, bboxes, token_type = sequence
                blank_bbox = [[0, 0], [0, 1], [1, 1], [1, 0]]
                if token_type == "ocr":
                    text = self.processor.ocr_tokenizer.decode(
                        token_ids, task=TaskNames.ocr_with_boxes
                    )
                    bboxes = clean_close_polygons(
                        bboxes
                    )  # clean out bboxes that are close, like what happens with multiple utf-16 tokens per char
                    bbox_idx = 0
                    for text_idx, text_line in enumerate(text):
                        img_chars.append(
                            TextChar(
                                text=text_line,
                                polygon=bboxes[bbox_idx],
                                confidence=seq_score[bbox_idx],
                                bbox_valid=True,
                            )
                        )

                        # Ensure we don't exceed the bbox count
                        if bbox_idx < len(bboxes) - 1:
                            bbox_idx += 1
                elif token_type == "special":
                    text = self.processor.ocr_tokenizer.decode(
                        token_ids, task="ocr_without_boxes"
                    )
                    if text in [NOMATH_TOKEN]:
                        continue

                    img_chars.append(
                        TextChar(
                            text=text,
                            polygon=blank_bbox,
                            confidence=seq_score[0],
                            bbox_valid=False,
                        )
                    )
                else:
                    text = self.processor.ocr_tokenizer.decode(
                        token_ids, task=TaskNames.block_without_boxes
                    )
                    img_chars.append(
                        TextChar(
                            text=text,
                            polygon=blank_bbox,
                            confidence=seq_score[0],
                            bbox_valid=False,
                        )
                    )

            char_predictions.append(img_chars)

        char_predictions = sorted(zip(indices, char_predictions), key=lambda x: x[0])
        char_predictions = [pred for _, pred in char_predictions]

        predictions_by_image = []
        slice_start = 0
        for idx, image in enumerate(images):
            slice_end = slice_start + flat["slice_map"][idx]
            image_lines = char_predictions[slice_start:slice_end]
            polygons = flat["polygons"][slice_start:slice_end]
            res_scales = flat["res_scales"][slice_start:slice_end]
            slice_start = slice_end

            lines = []
            for text_line, polygon, res_scale in zip(image_lines, polygons, res_scales):
                # Special case when input text is good
                if not text_line:
                    lines.append(
                        TextLine(
                            text="",
                            polygon=polygon,
                            chars=[],
                            confidence=1,
                            original_text_good=True,
                        )
                    )
                else:
                    text = "".join([char.text for char in text_line])
                    confidence = float(np.mean([char.confidence for char in text_line]))
                    poly_box = PolygonBox(polygon=polygon)
                    for char in text_line:
                        char.rescale(
                            res_scale, (1, 1)
                        )  # Rescale from highres if needed
                        char.shift(
                            poly_box.bbox[0], poly_box.bbox[1]
                        )  # Ensure character boxes match line boxes (relative to page)

                    text_line = fix_unbalanced_tags(
                        text_line, self.processor.ocr_tokenizer.special_tokens
                    )
                    lines.append(
                        TextLine(
                            text=text,
                            polygon=polygon,
                            chars=text_line,
                            confidence=confidence,
                            words=words_from_chars(text_line),
                        )
                    )

            if sort_lines:
                lines = sort_text_lines(lines)
            predictions_by_image.append(
                OCRResult(
                    text_lines=lines, image_bbox=[0, 0, image.size[0], image.size[1]]
                )
            )

        return predictions_by_image
