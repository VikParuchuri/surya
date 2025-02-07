from copy import deepcopy
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from surya.common.util import mark_step
from surya.common.predictor import BasePredictor
from surya.detection import DetectionPredictor
from surya.input.processing import convert_if_not_rgb, slice_polys_from_image, slice_bboxes_from_image
from surya.recognition.loader import RecognitionModelLoader
from surya.recognition.postprocessing import truncate_repetitions
from surya.recognition.processor import SuryaProcessor
from surya.recognition.util import sort_text_lines
from surya.recognition.schema import TextLine, OCRResult
from surya.settings import settings


class RecognitionPredictor(BasePredictor):
    model_loader_cls = RecognitionModelLoader
    batch_size = settings.RECOGNITION_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 32,
        "mps": 64,
        "cuda": 256,
        "xla": 128
    }

    def __call__(
            self,
            images: List[Image.Image],
            langs: List[List[str] | None],
            det_predictor: DetectionPredictor | None = None,
            detection_batch_size: int | None = None,
            recognition_batch_size: int | None = None,
            highres_images: List[Image.Image] | None = None,
            bboxes: List[List[List[int]]] | None = None,
            polygons: List[List[List[List[int]]]] | None = None,
            sort_lines: bool = True
    ) -> List[OCRResult]:
            assert len(images) == len(langs), "You need to pass in one list of languages for each image"
            images = convert_if_not_rgb(images)
            if highres_images is not None:
                assert len(images) == len(highres_images), "You need to pass in one highres image for each image"

            highres_images = convert_if_not_rgb(highres_images) if highres_images is not None else [None] * len(images)

            if bboxes is None and polygons is None:
                assert det_predictor is not None, "You need to pass in a detection predictor if you don't provide bboxes or polygons"

                # Detect then slice
                flat = self.detect_and_slice_bboxes(
                    images,
                    langs,
                    det_predictor,
                    detection_batch_size=detection_batch_size,
                    highres_images=highres_images
                )
            else:
                if bboxes is not None:
                    assert len(images) == len(bboxes), "You need to pass in one list of bboxes for each image"
                if polygons is not None:
                    assert len(images) == len(polygons), "You need to pass in one list of polygons for each image"

                flat = self.slice_bboxes(
                    images,
                    langs,
                    bboxes=bboxes,
                    polygons=polygons
                )

            rec_predictions, confidence_scores = self.batch_recognition(
                flat["slices"],
                flat["langs"],
                batch_size=recognition_batch_size
            )

            predictions_by_image = []
            slice_start = 0
            for idx, (image, lang) in enumerate(zip(images, langs)):
                slice_end = slice_start + flat["slice_map"][idx]
                image_lines = rec_predictions[slice_start:slice_end]
                line_confidences = confidence_scores[slice_start:slice_end]
                polygons = flat["polygons"][slice_start:slice_end]
                slice_start = slice_end

                lines = []
                for text_line, confidence, polygon in zip(image_lines, line_confidences, polygons):
                    lines.append(TextLine(
                        text=text_line,
                        polygon=polygon,
                        confidence=confidence
                    ))

                if sort_lines:
                    lines = sort_text_lines(lines)
                predictions_by_image.append(OCRResult(
                    text_lines=lines,
                    languages=lang,
                    image_bbox=[0, 0, image.size[0], image.size[1]]
                ))

            return predictions_by_image

    def detect_and_slice_bboxes(
            self,
            images: List[Image.Image],
            langs: List[List[str] | None],
            det_predictor: DetectionPredictor,
            detection_batch_size: int | None = None,
            highres_images: List[Image.Image] | None = None,
    ):
        det_predictions = det_predictor(images, batch_size=detection_batch_size)

        all_slices = []
        slice_map = []
        all_langs = []
        all_polygons = []

        for idx, (det_pred, image, highres_image, lang) in enumerate(zip(det_predictions, images, highres_images, langs)):
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
            all_langs.extend([lang] * len(slices))
            all_slices.extend(slices)
            all_polygons.extend(polygons)

        assert len(all_slices) == sum(slice_map) == len(all_langs) == len(all_polygons)

        return {
            "slices": all_slices,
            "slice_map": slice_map,
            "langs": all_langs,
            "polygons": all_polygons
        }

    def slice_bboxes(
            self,
            images: List[Image.Image],
            langs: List[List[str] | None],
            bboxes: List[List[List[int]]] | None = None,
            polygons: List[List[List[List[int]]]] | None = None
    ):
        assert bboxes is not None or polygons is not None
        slice_map = []
        all_slices = []
        all_langs = []
        all_polygons = []
        for idx, (image, lang) in enumerate(zip(images, langs)):
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
            all_langs.extend([deepcopy(lang)] * len(slices))
            all_polygons.extend(polys)

        assert len(all_slices) == sum(slice_map) == len(all_langs) == len(all_polygons)

        return {
            "slices": all_slices,
            "slice_map": slice_map,
            "langs": all_langs,
            "polygons": all_polygons
        }

    def prepare_input(self, batch_langs, batch_pixel_values, batch_size):
        batch_decoder_input = [[self.model.config.decoder_start_token_id] + lang for lang in batch_langs]
        max_input_length = max(len(tokens) for tokens in batch_decoder_input)

        # Pad decoder input to max length if needed, to ensure we can convert to a tensor
        for idx, tokens in enumerate(batch_decoder_input):
            if len(tokens) < max_input_length:
                padding_length = max_input_length - len(tokens)
                batch_decoder_input[idx] = [self.processor.tokenizer.pad_id] * padding_length + tokens

        batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=self.model.dtype)
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long)
        if settings.RECOGNITION_STATIC_CACHE:
            batch_pixel_values = self.pad_to_batch_size(batch_pixel_values, batch_size)
            batch_decoder_input = self.pad_to_batch_size(batch_decoder_input, batch_size)

        # Moving this after the padding fixes XLA recompilation issues
        batch_pixel_values = batch_pixel_values.to(self.model.device)
        batch_decoder_input = batch_decoder_input.to(self.model.device)

        return batch_pixel_values, batch_decoder_input

    def batch_recognition(
            self,
            images: List[Image.Image],
            languages: List[List[str] | None],
            batch_size=None
    ):
        assert all(isinstance(image, Image.Image) for image in images)
        assert len(images) == len(languages)

        if len(images) == 0:
            return [], []

        if batch_size is None:
            batch_size = self.get_batch_size()

        # Sort images by width, so similar length ones go together
        sorted_pairs = sorted(enumerate(images), key=lambda x: x[1].width, reverse=False)
        indices, images = zip(*sorted_pairs)
        indices = list(indices)
        images = list(images)

        output_text = []
        confidences = []
        for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text", disable=self.disable_tqdm):
            batch_images = images[i:i + batch_size]
            batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images
            current_batch_size = len(batch_images)
            batch_langs = languages[i:i + current_batch_size]
            processed_batch = self.processor(text=[""] * len(batch_images), images=batch_images, langs=batch_langs)

            batch_pixel_values = processed_batch["pixel_values"]
            batch_langs = processed_batch["langs"]
            batch_pixel_values, batch_decoder_input = self.prepare_input(
                batch_langs,
                batch_pixel_values,
                batch_size
            )

            token_count = 0
            inference_token_count = batch_decoder_input.shape[-1]

            decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64,
                                                   device=self.model.device).cumsum(0) - 1
            self.model.decoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)
            self.model.text_encoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)

            # Batch pixel values is the real current batch size
            sequence_scores = torch.zeros(batch_pixel_values.shape[0], dtype=torch.bool, device=self.model.device).unsqueeze(1)
            all_done = torch.zeros(batch_pixel_values.shape[0], dtype=torch.bool, device=self.model.device)
            batch_predictions = torch.zeros(batch_pixel_values.shape[0], dtype=torch.int64, device=self.model.device).unsqueeze(1)
            device_pad_token = torch.tensor(self.processor.tokenizer.pad_token_id, device=self.model.device)

            with settings.INFERENCE_MODE():
                encoder_hidden_states = self.model.encoder(pixel_values=batch_pixel_values).last_hidden_state

                text_encoder_input_ids = torch.arange(
                    self.model.text_encoder.config.query_token_count,
                    device=encoder_hidden_states.device,
                    dtype=torch.long
                ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)

                encoder_text_hidden_states = self.model.text_encoder(
                    input_ids=text_encoder_input_ids,
                    cache_position=None,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    use_cache=False
                ).hidden_states

                while token_count < settings.RECOGNITION_MAX_TOKENS - 1:
                    is_prefill = token_count == 0
                    # TODO: add attention mask
                    return_dict = self.model.decoder(
                        input_ids=batch_decoder_input,
                        encoder_hidden_states=encoder_text_hidden_states,
                        cache_position=decoder_position_ids,
                        use_cache=True,
                        prefill=is_prefill
                    )

                    decoder_position_ids = decoder_position_ids[-1:] + 1
                    logits = return_dict["logits"]  # Ignore batch padding

                    preds = torch.argmax(logits[:, -1], dim=-1)
                    scores = torch.max(F.softmax(logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)
                    done = (preds == self.processor.tokenizer.eos_id) | (preds == self.processor.tokenizer.pad_id)
                    all_done = all_done | done
                    all_done_cpu = all_done.cpu()

                    # Confidence score for the current token
                    scores = scores.masked_fill(all_done, 0)
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                    # Account for possible padding
                    if all_done_cpu[:current_batch_size].all():
                        break

                    batch_decoder_input = preds.unsqueeze(1)

                    # If this batch item is done, input a pad token
                    batch_decoder_input = torch.where(all_done.unsqueeze(1), device_pad_token, batch_decoder_input)

                    batch_predictions = torch.cat([batch_predictions, batch_decoder_input], dim=1)
                    token_count += inference_token_count

                    inference_token_count = batch_decoder_input.shape[-1]
                    mark_step()

            sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)

            sequence_scores = sequence_scores.cpu()[:current_batch_size]
            batch_predictions = batch_predictions.cpu()[:current_batch_size, 1:] # Remove the start token
            detected_text = self.processor.tokenizer.batch_decode(batch_predictions)

            # Convert sequence_scores to list for the current batch
            batch_confidences = sequence_scores.tolist()

            output_text.extend(detected_text)
            confidences.extend(batch_confidences)

        output_text = sorted(zip(indices, output_text), key=lambda x: x[0])
        confidences = sorted(zip(indices, confidences), key=lambda x: x[0])
        output_text = [text for _, text in output_text]
        confidences = [conf for _, conf in confidences]
        return output_text, confidences