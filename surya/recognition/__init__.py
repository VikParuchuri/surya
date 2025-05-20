from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

import numpy as np
from PIL import Image

from surya.common.polygon import PolygonBox
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.input.processing import (
    convert_if_not_rgb,
    slice_polys_from_image,
    slice_bboxes_from_image,
)

from surya.recognition.postprocessing import fix_unbalanced_tags
from surya.recognition.util import (
    sort_text_lines,
    clean_close_polygons,
    words_from_chars,
)
from surya.foundation.util import prediction_to_polygon_batch
from surya.recognition.schema import TextLine, OCRResult, TextChar
from surya.common.surya.schema import TaskNames
from surya.common.surya.processor import NOMATH_TOKEN
from surya.settings import settings

class RecognitionPredictor(FoundationPredictor):
    batch_size = settings.RECOGNITION_BATCH_SIZE
    default_batch_sizes = {"cpu": 32, "mps": 64, "cuda": 256, "xla": 128}

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
                highres_image = self.processor.image_processor(highres_image)
                slices = slice_polys_from_image(highres_image, scaled_polygons)
                res_scales = [(width_scaler, height_scaler) for _ in range(len(slices))]
            else:
                image = self.processor.image_processor(image)
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
            image = self.processor.image_processor(image)
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

    def get_bboxes_text(
        self, flat: dict, predicted_tokens: list, scores: list, predicted_polygons: list
    ) -> list:
        char_predictions = []
        needs_boxes = [
            self.tasks[task_name]["needs_bboxes"] for task_name in flat["task_names"]
        ]

        for slice_idx, (
            slice_image,
            image_tokens,
            image_polygons,
            image_scores,
            needs_box,
        ) in enumerate(
            zip(
                flat["slices"],
                predicted_tokens,
                predicted_polygons,
                scores,
                needs_boxes,
            )
        ):
            if self.processor.no_output_token in image_tokens:
                char_predictions.append(None)
                continue

            image_polygons = image_polygons[: len(image_tokens)].cpu().numpy().tolist()

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
                        # Use the last bbox for the rest of the text
                        if bbox_idx < len(bboxes) - 1:
                            bbox_idx += 1
                elif token_type == "special":
                    text = self.processor.ocr_tokenizer.decode(
                        token_ids, task="ocr_without_boxes"
                    )
                    if text in [NOMATH_TOKEN] or re.match(r"<SCRIPT-\w+>", text):
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

        return char_predictions

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
        sort_lines: bool = False,
        math_mode: bool = True,
        return_words: bool = False,
    ) -> List[OCRResult]:
        allowed_tasks = self.tasks.keys()
        if task_names is None:
            task_names = [TaskNames.ocr_with_boxes] * len(images)

        if recognition_batch_size is None:
            recognition_batch_size = self.get_batch_size()

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
        sorted_pairs = sorted(enumerate(flat["slices"]), key=lambda x: -x[1].shape[1])
        indices, sorted_slices = zip(*sorted_pairs)

        # Reorder input_text and task_names based on the new order
        flat["slices"] = list(sorted_slices)
        flat["input_text"] = [flat["input_text"][i] for i in indices]
        flat["task_names"] = [flat["task_names"][i] for i in indices]

        # Make predictions
        predicted_tokens, batch_bboxes, scores = self.prediction_loop(
            processed_images=flat['slices'],
            input_texts=flat['input_text'],
            task_names=flat['task_names'],
            batch_size=recognition_batch_size,
            math_mode=math_mode
        )

        # Get text and bboxes in structured form
        bbox_size = self.model.config.bbox_size
        image_sizes = [img.shape for img in flat["slices"]]
        predicted_polygons = prediction_to_polygon_batch(
            batch_bboxes, image_sizes, bbox_size, bbox_size // 2
        )
        char_predictions = self.get_bboxes_text(
            flat, predicted_tokens, scores, predicted_polygons
        )

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
                    confidence = float(np.mean([char.confidence for char in text_line]))
                    poly_box = PolygonBox(polygon=polygon)
                    for char in text_line:
                        char.rescale(
                            res_scale, (1, 1)
                        )  # Rescale from highres if needed
                        char.shift(
                            poly_box.bbox[0], poly_box.bbox[1]
                        )  # Ensure character boxes match line boxes (relative to page)
                        char.clamp(poly_box.bbox)

                    text_line = fix_unbalanced_tags(
                        text_line, self.processor.ocr_tokenizer.special_tokens
                    )
                    text = "".join([char.text for char in text_line])
                    lines.append(
                        TextLine(
                            text=text,
                            polygon=polygon,
                            chars=text_line,
                            confidence=confidence,
                            words=words_from_chars(text_line, poly_box)
                            if return_words
                            else [],
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
