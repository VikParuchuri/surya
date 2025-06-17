from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import QuantizedCacheConfig

from surya.common.surya import SuryaModelOutput
from surya.common.util import mark_step
from surya.common.predictor import BasePredictor

from surya.foundation.loader import FoundationModelLoader
from surya.foundation.util import (
    detect_repeat_token,
)
from surya.common.surya.schema import TaskNames
from surya.foundation.cache import (
    ContinuousBatchingCache,
    ContinuousBatchingQuantizedCache,
)
from surya.settings import settings
from surya.logging import get_logger, configure_logging

configure_logging()
logger = get_logger()


@dataclass
class ContinuousBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


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
    image: np.ndarray
    text: str
    math_mode: bool


class FoundationPredictor(BasePredictor):
    model_loader_cls = FoundationModelLoader
    batch_size = settings.RECOGNITION_BATCH_SIZE        # Default to the recognition batch size
    torch_dtype = None  # No default, loader picks the dtype based on device properties - bf16/fp16
    default_batch_sizes = {"cpu": 32, "mps": 64, "cuda": 256, "xla": 128}
    encoder_chunk_size: int = 4096  # Default chunk size
    encoder_chunk_sizes = {"cpu": 4096, "mps": 4096, "cuda": 32768, "xla": 32768}
    min_prefill_ratio: int = 0.2
    min_trim_length: int = 50
    tasks = {
        TaskNames.ocr_with_boxes: {
            "needs_bboxes": True,
            "img_size": (1024, 256),  # 370 max tokens
            "max_tokens": 224,
        },
        TaskNames.ocr_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 256),  # 370 max tokens
            "max_tokens": 224,
        },
        TaskNames.block_without_boxes: {
            "needs_bboxes": False,
            "img_size": (1024, 512),  # 703 max tokens
            "max_tokens": 768,
        },
    }

    def __init__(self, checkpoint=None, device=settings.TORCH_DEVICE_MODEL, dtype=None):
        super().__init__(checkpoint, device, dtype)
        self.kv_cache = None
        self.prompt_queue = deque()
        self.batch_prompt_mapping = None

        # Setup various tokens on-device
        self.device_pad_token = torch.tensor(
            self.processor.pad_token_id, device=self.model.device, dtype=torch.long
        )

    def get_encoder_chunk_size(self) -> int:
        if settings.RECOGNITION_CHUNK_SIZE is not None:
            return settings.RECOGNITION_CHUNK_SIZE

        chunk_size = self.encoder_chunk_size
        if settings.TORCH_DEVICE_MODEL in self.encoder_chunk_sizes:
            if settings.TORCH_DEVICE_MODEL in self.encoder_chunk_sizes:
                chunk_size = self.encoder_chunk_sizes[settings.TORCH_DEVICE_MODEL]
        return chunk_size

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

            try:
                image = self.processor.scale_to_fit(
                    image, image_size
                )  # Only resizes if out of bounds (max/min)
            except cv2.error:
                # The image is empty if it can't be resized, so just make a blank image
                image = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)

            # Task input is the same for all tasks for now
            text = text or ""

            # Remove input text that exceeds max generation tokens (likely invalid)
            if len(text) > self.tasks[task_name]["max_tokens"]:
                text = ""
            inputs = [
                {"type": "image", "image": image, "rotated": False},
                {"type": "text", "text": text.strip(), "math": math_mode},
            ]
            batch.append({"task": task_name, "inputs": inputs})

        return batch

    def process_outputs(self, outputs: SuryaModelOutput) -> ContinuousBatchOutput:
        # Get logits and initial preds
        next_token_logits = outputs["lm_logits"][:, -1:, :].clone().float()
        next_bbox_logits = outputs["bbox_logits"][:, -1:, :].clone().float()
        preds = torch.argmax(next_token_logits, dim=-1)

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
        input_boxes = box_preds.to(torch.long)

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

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=self.kv_cache,
                logits_to_keep=1,
            )

        processed_output: ContinuousBatchOutput = self.process_outputs(outputs)

        attention_mask = F.pad(attention_mask, (0, 1), mode="constant", value=1)

        position_ids = position_ids[:, -1:] + 1
        new_input = ContinuousBatchInput(
            input_ids=processed_output.input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        return new_input, processed_output

    def prefill(self, current_inputs: Optional[ContinuousBatchInput] = None):
        logger.debug(f"Prefilling {self.num_empty_slots} slots")
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
        processed_inputs = self.processor(
            batch_input, padding_side="left", device=self.model.device
        ).to(device=self.model.device)

        input_ids = processed_inputs["input_ids"].to(dtype=torch.long)
        image_tiles = processed_inputs["image_tiles"].to(dtype=self.model.dtype)
        grid_thw = processed_inputs["grid_thw"].to(dtype=torch.long)
        attention_mask = processed_inputs["attention_mask"].to(dtype=torch.long)
        position_ids = processed_inputs["position_ids"].to(dtype=torch.long)

        if settings.FOUNDATION_MODEL_QUANTIZE:
            try:
                import hqq  # noqa: F401
            except Exception:
                raise ImportError(
                    "Please install hqq to use quantized recognition model"
                )

        # Use quantized cache if setting activated
        cache_config = QuantizedCacheConfig(
            "HQQ", 8, 1, 1, device=self.model.device, compute_dtype=self.model.dtype
        )
        prefill_cache = (
            ContinuousBatchingCache()
            if not settings.FOUNDATION_MODEL_QUANTIZE
            else ContinuousBatchingQuantizedCache(cache_config)
        )

        with settings.INFERENCE_MODE():
            outputs = self.model(
                input_ids=input_ids,
                image_tiles=image_tiles,
                grid_thw=grid_thw,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                past_key_values=prefill_cache,
                use_cache=True,
                logits_to_keep=1,
                encoder_chunk_size=self.get_encoder_chunk_size(),
            )

        # Process outputs
        processed_outputs = self.process_outputs(outputs)

        # Merge new kv cache with existing, update batch mapping
        non_active_idxs = [k for k, v in self.batch_prompt_mapping.items() if v is None]
        idxs_to_merge = non_active_idxs[: len(prompts)]

        assert len(idxs_to_merge) == len(prompts), (
            "Number of prompts should match number of empty slots"
        )
        for i, prompt in zip(idxs_to_merge, prompts):
            self.batch_prompt_mapping[i] = prompt.id

        if self.kv_cache:
            offset = self.kv_cache.merge(
                prefill_cache, idxs_to_merge, self.model.device
            )
        else:
            self.kv_cache = prefill_cache
            offset = 0

        # Adjust attention mask and position ids to account for the newly generated tokens
        attention_mask = F.pad(attention_mask, (0, 1), mode="constant", value=1)
        position_ids = position_ids[:, -1:] + 1

        if current_inputs is None:
            new_input = ContinuousBatchInput(
                input_ids=processed_outputs.input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
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

        new_input = ContinuousBatchInput(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids,
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
        trim_start = first_non_padding_idx.min()

        # Trimming too much slows things down
        if trim_start < self.min_trim_length:
            return current_inputs

        logger.debug(f"Trimming cache from left by {trim_start} tokens.")
        trimmed_attention_mask = attention_mask[:, trim_start:]
        current_inputs.attention_mask = trimmed_attention_mask

        # Trim the cache accordingly
        if self.kv_cache:
            self.kv_cache.trim_left(trim_start)

        return current_inputs

    def prediction_loop(
        self,
        images: List[Image.Image],
        input_texts: List[str],
        task_names: List[TaskNames],
        batch_size: int | None = None,
        math_mode: bool = True,
        drop_repeated_tokens: bool = True,
    ) -> tuple:
        allowed_tasks = self.tasks.keys()
        assert all([task_name in allowed_tasks for task_name in task_names]), (
            f"One or more tasks in {task_names} is not supported. Supported tasks are {allowed_tasks}"
        )

        predicted_tokens = [[] for _ in range(len(images))]
        scores = [[] for _ in range(len(images))]

        if batch_size is None:
            batch_size = self.get_batch_size()
        current_inputs = None
        self.setup_cache(batch_size)

        batch_max_tokens = {}
        for idx, (img, txt, task) in enumerate(
            zip(images, input_texts, task_names)
        ):
            self.prompt_queue.append(
                RecognitionPrompt(
                    id=idx, task_name=task, text=txt, image=img, math_mode=math_mode
                )
            )
            batch_max_tokens[idx] = (
                settings.FOUNDATION_MAX_TOKENS or self.tasks[task]["max_tokens"]
            )

        overall_max_tokens = max(batch_max_tokens.values())

        pbar = tqdm(
            total=len(self.prompt_queue),
            desc="Recognizing Text",
            disable=self.disable_tqdm,
        )

        batch_bboxes = torch.zeros(len(images), overall_max_tokens, 6)
        batch_pos = [0] * len(images)

        while self.prompt_queue or self.num_active_slots > 0:
            if (
                self.num_empty_slots / batch_size
            ) > self.min_prefill_ratio and self.prompt_queue:
                updated_inputs, outputs, merge_idxs = self.prefill(current_inputs)

                predicted_tokens_cpu = outputs.preds.cpu()
                scores_cpu = outputs.scores.cpu()
                for temp_idx, b_idx in enumerate(merge_idxs):
                    if self.batch_prompt_mapping[b_idx] is not None:
                        p_idx = self.batch_prompt_mapping[b_idx]
                        predicted_tokens[p_idx].append(
                            predicted_tokens_cpu[temp_idx].item()
                        )
                        batch_bboxes[p_idx, batch_pos[p_idx]] = outputs.bbox_preds[
                            temp_idx
                        ][0]
                        batch_pos[p_idx] += 1
                        scores[p_idx].append(scores_cpu[temp_idx].item())

                        if predicted_tokens[p_idx][-1] in [
                            self.processor.eos_token_id,
                            self.processor.no_output_token,
                        ]:
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)
            else:
                updated_inputs, outputs = self.decode(current_inputs)
                # TODO Find a cleaner way of popping from the dict
                predicted_tokens_cpu = outputs.preds.cpu()
                scores_cpu = outputs.scores.cpu()

                for b_idx, p_idx in self.batch_prompt_mapping.items():
                    if p_idx is not None:
                        predicted_tokens[p_idx].append(
                            predicted_tokens_cpu[b_idx].item()
                        )
                        batch_bboxes[p_idx, batch_pos[p_idx]] = outputs.bbox_preds[
                            b_idx
                        ][0]
                        batch_pos[p_idx] += 1

                        scores[p_idx].append(scores_cpu[b_idx].item())

                        repeats = (
                            len(predicted_tokens[p_idx]) >= batch_max_tokens[p_idx]
                            or (drop_repeated_tokens and detect_repeat_token(predicted_tokens[p_idx]))
                        )
                        if (
                            predicted_tokens[p_idx][-1]
                            in [
                                self.processor.eos_token_id,
                                self.processor.pad_token_id,
                            ]
                            or repeats
                        ):
                            self.batch_prompt_mapping[b_idx] = None
                            pbar.update(1)

            # Update inputs and mark XLA step
            current_inputs = updated_inputs
            current_inputs = self.maybe_trim_cache_padding(current_inputs)
            mark_step()
        pbar.close()

        del self.kv_cache
        self.kv_cache = None
        torch.cuda.empty_cache()

        return predicted_tokens, batch_bboxes, scores