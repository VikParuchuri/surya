from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from surya.common.predictor import BasePredictor
from surya.layout.loader import LayoutModelLoader
from surya.layout.model.config import ID_TO_LABEL
from surya.layout.slicer import ImageSlicer
from surya.layout.util import prediction_to_polygon
from surya.common.util import clean_boxes
from surya.layout.schema import LayoutBox, LayoutResult
from surya.settings import settings


class LayoutPredictor(BasePredictor):
    model_loader_cls = LayoutModelLoader
    batch_size = settings.LAYOUT_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 4,
        "mps": 4,
        "cuda": 32
    }

    def __call__(
            self,
            images: List[Image.Image],
            batch_size: int | None = None,
            top_k: int = 5
    ) -> List[LayoutResult]:
        return self.batch_layout_detection(
            images,
            top_k=top_k,
            batch_size=batch_size
        )

    def batch_layout_detection(
            self,
            images: List[Image.Image],
            batch_size: int | None = None,
            top_k: int = 5
    ) -> List[LayoutResult]:
        assert all([isinstance(image, Image.Image) for image in images])
        if batch_size is None:
            batch_size = self.get_batch_size()

        slicer = ImageSlicer(settings.LAYOUT_SLICE_MIN, settings.LAYOUT_SLICE_SIZE)

        batches = []
        img_counts = [slicer.slice_count(image) for image in images]

        start_idx = 0
        end_idx = 1
        while end_idx < len(img_counts):
            if any([
                sum(img_counts[start_idx:end_idx]) >= batch_size,
                sum(img_counts[start_idx:end_idx + 1]) > batch_size,
            ]):
                batches.append((start_idx, end_idx))
                start_idx = end_idx
            end_idx += 1

        if start_idx < len(img_counts):
            batches.append((start_idx, len(img_counts)))

        results = []
        for (start_idx, end_idx) in tqdm(batches, desc="Recognizing layout"):
            batch_results = []
            batch_images = images[start_idx:end_idx]
            batch_images = [image.convert("RGB") for image in batch_images]  # also copies the image
            batch_images, tile_positions = slicer.slice(batch_images)
            current_batch_size = len(batch_images)

            orig_sizes = [image.size for image in batch_images]
            model_inputs = self.processor(batch_images)

            batch_pixel_values = model_inputs["pixel_values"]
            batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=self.model.dtype).to(self.model.device)

            pause_token = [self.model.config.decoder.pause_token_id] * 7
            start_token = [self.model.config.decoder.bos_token_id] * 7
            batch_decoder_input = [
                [start_token] + [pause_token] * self.model.config.decoder.pause_token_count
                for _ in range(current_batch_size)
            ]
            batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long,
                                               device=self.model.device)
            inference_token_count = batch_decoder_input.shape[1]

            decoder_position_ids = torch.ones_like(batch_decoder_input[0, :, 0], dtype=torch.int64,
                                                   device=self.model.device).cumsum(0) - 1
            self.model.decoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)

            batch_predictions = [[] for _ in range(current_batch_size)]

            with torch.inference_mode():
                encoder_hidden_states = self.model.encoder(pixel_values=batch_pixel_values)[0]

                token_count = 0
                all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device)

                while token_count < settings.LAYOUT_MAX_BOXES:
                    is_prefill = token_count == 0
                    return_dict = self.model.decoder(
                        input_boxes=batch_decoder_input,
                        encoder_hidden_states=encoder_hidden_states,
                        cache_position=decoder_position_ids,
                        use_cache=True,
                        prefill=is_prefill
                    )

                    decoder_position_ids = decoder_position_ids[-1:] + 1
                    box_logits = return_dict["bbox_logits"][:current_batch_size, -1, :].detach()
                    class_logits = return_dict["class_logits"][:current_batch_size, -1, :].detach()

                    class_preds = class_logits.argmax(-1)
                    box_preds = box_logits * self.model.config.decoder.bbox_size

                    done = (class_preds == self.model.decoder.config.eos_token_id) | (
                                class_preds == self.model.decoder.config.pad_token_id)

                    all_done = all_done | done
                    if all_done.all():
                        break

                    batch_decoder_input = torch.cat([box_preds.unsqueeze(1), class_preds.unsqueeze(1).unsqueeze(1)],
                                                    dim=-1)

                    for j, (pred, status) in enumerate(zip(batch_decoder_input, all_done)):
                        if not status:
                            preds = pred[0].detach().cpu()
                            prediction = {
                                "preds": preds,
                                "token": preds,
                                "polygon": prediction_to_polygon(
                                    preds,
                                    orig_sizes[j],
                                    self.model.config.decoder.bbox_size,
                                    self.model.config.decoder.skew_scaler
                                ),
                                "label": preds[6].item() - self.model.decoder.config.special_token_count,
                                "class_logits": class_logits[j].detach().cpu(),
                                "orig_size": orig_sizes[j]
                            }
                            prediction["text_label"] = ID_TO_LABEL.get(int(prediction["label"]), None)
                            if all([
                                prediction["text_label"] in ["PageHeader", "PageFooter"],
                                prediction["polygon"][0][1] < prediction["orig_size"][1] * .8,
                                prediction["polygon"][2][1] > prediction["orig_size"][1] * .2,
                                prediction["polygon"][0][0] < prediction["orig_size"][0] * .8,
                                prediction["polygon"][2][0] > prediction["orig_size"][0] * .2
                            ]):
                                # Ensure page footers only occur at the bottom of the page, headers only at top
                                prediction["class_logits"][int(preds[6].item())] = 0
                                new_prediction = prediction["class_logits"].argmax(-1).item()
                                prediction["label"] = new_prediction - self.model.decoder.config.special_token_count
                                prediction["token"][6] = new_prediction
                                batch_decoder_input[j, -1, 6] = new_prediction

                            prediction["top_k_probs"], prediction["top_k_indices"] = torch.topk(
                                torch.nn.functional.softmax(prediction["class_logits"], dim=-1), k=top_k, dim=-1)
                            del prediction["class_logits"]
                            batch_predictions[j].append(prediction)

                    token_count += inference_token_count
                    inference_token_count = batch_decoder_input.shape[1]
                    batch_decoder_input = batch_decoder_input.to(torch.long)

            for j, (pred_dict, orig_size) in enumerate(zip(batch_predictions, orig_sizes)):
                boxes = []
                preds = [p for p in pred_dict if
                         p["token"][6] > self.model.decoder.config.special_token_count]  # Remove special tokens, like pause
                if len(preds) > 0:
                    polygons = [p["polygon"] for p in preds]
                    labels = [p["label"] for p in preds]
                    top_k_probs = [p["top_k_probs"] for p in preds]
                    top_k_indices = [p["top_k_indices"] - self.model.decoder.config.special_token_count for p in preds]

                    for z, (poly, label, top_k_prob, top_k_index) in enumerate(
                            zip(polygons, labels, top_k_probs, top_k_indices)):
                        top_k_dict = {
                            ID_TO_LABEL.get(int(l)): prob.item()
                            for (l, prob) in zip(top_k_index, top_k_prob) if l > 0
                        }
                        l = ID_TO_LABEL[int(label)]
                        lb = LayoutBox(
                            polygon=poly,
                            label=l,
                            position=z,
                            top_k=top_k_dict,
                            confidence=top_k_dict[l]
                        )
                        boxes.append(lb)
                boxes = clean_boxes(boxes)
                result = LayoutResult(
                    bboxes=boxes,
                    image_bbox=[0, 0, orig_size[0], orig_size[1]]
                )
                batch_results.append(result)

            assert len(batch_results) == len(tile_positions)
            batch_results = slicer.join(batch_results, tile_positions)
            results.extend(batch_results)

        assert len(results) == len(images)
        return results