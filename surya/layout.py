from typing import List
import torch
from PIL import Image

from surya.model.layout.config import SPECIAL_TOKENS, ID_TO_LABEL
from surya.postprocessing.math.latex import fix_math, contains_math
from surya.postprocessing.text import truncate_repetitions
from surya.postprocessing.util import bbox_to_polygon
from surya.schema import LayoutResult, LayoutBox
from surya.settings import settings
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def get_batch_size():
    batch_size = settings.RECOGNITION_BATCH_SIZE
    if batch_size is None:
        batch_size = 32
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 64 # 12GB RAM max
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 512
    return batch_size


def batch_layout_detection(images: List, model, processor, batch_size=None) -> List[LayoutResult]:
    assert all([isinstance(image, Image.Image) for image in images])

    if len(images) == 0:
        return []

    images = [image.convert("RGB") for image in images]  # also copies the images
    if batch_size is None:
        batch_size = get_batch_size()

    layout_results = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_images = images[i:i+batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images
        orig_sizes = [image.size for image in batch_images]

        processed_batch = processor(images=batch_images)
        batch_pixel_values = processed_batch["pixel_values"]
        current_batch_size = len(batch_pixel_values)
        batch_decoder_input = [[[model.config.decoder_start_token_id] * 5] for _ in range(current_batch_size)]

        batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=model.dtype, device=model.device)
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)

        token_count = 0
        inference_token_count = batch_decoder_input.shape[-1]
        batch_predictions = [[] for _ in range(current_batch_size)]

        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :, 0], dtype=torch.int64, device=model.device).cumsum(0) - 1
        model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)
        model.text_encoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)

        sequence_scores = None
        all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)

        with torch.no_grad(): # inference_mode doesn't work with torch.compile
            encoder_hidden_states = model.encoder(pixel_values=batch_pixel_values).last_hidden_state

            text_encoder_input_ids = torch.arange(
                model.text_encoder.config.query_token_count,
                device=encoder_hidden_states.device,
                dtype=torch.long
            ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)

            encoder_text_hidden_states = model.text_encoder(
                input_ids=text_encoder_input_ids,
                cache_position=None,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                use_cache=False
            ).hidden_states
            del encoder_hidden_states

            while token_count < settings.RECOGNITION_MAX_TOKENS - 1:
                is_prefill = token_count == 0
                #TODO: add attention mask
                return_dict = model.decoder(
                    input_boxes=batch_decoder_input,
                    encoder_hidden_states=encoder_text_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True,
                    prefill=is_prefill
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1
                bbox_logits = return_dict["bbox_logits"][:current_batch_size] # Ignore batch padding
                class_logits = return_dict["class_logits"][:current_batch_size] # Ignore batch padding
                bbox_preds = bbox_logits[:, -1]
                class_preds = torch.argmax(class_logits[:, -1], dim=-1)
                scores = torch.max(F.softmax(class_logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)
                done = (class_preds == processor.eos_id) | (class_preds == processor.pad_id)
                all_done = all_done | done

                if is_prefill:
                    sequence_scores = scores
                else:
                    scores = scores.masked_fill(all_done, 0)
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                if all_done.all():
                    break

                for j, (bbox, class_, status) in enumerate(zip(bbox_preds, class_preds, all_done)):
                    if not status:
                        label = ID_TO_LABEL[int(class_) - SPECIAL_TOKENS] # Map from output class id to label
                        bbox = [b - SPECIAL_TOKENS for b in bbox.tolist()] # Map from output bbox id to bbox
                        bbox = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] # cx,cy to x0,y0, etc.
                        width_scaler = orig_sizes[j][0] / settings.LAYOUT_IMAGE_SIZE["width"]
                        height_scaler = orig_sizes[j][1] / settings.LAYOUT_IMAGE_SIZE["height"]
                        fixed_bbox = [
                            bbox[0] * width_scaler,
                            bbox[1] * height_scaler,
                            bbox[2] * width_scaler,
                            bbox[3] * height_scaler
                        ]
                        poly = bbox_to_polygon(fixed_bbox)
                        batch_predictions[j].append(LayoutBox(
                            polygon=poly,
                            label=label,
                        ))

                token_count += inference_token_count
                inference_token_count = batch_decoder_input.shape[-1]

                bbox_preds = torch.round(bbox_preds).to(torch.long) # Round to nearest int for embedding
                batch_decoder_input = torch.cat([bbox_preds, class_preds.unsqueeze(1)], dim=-1).unsqueeze(1)
                max_position_id = torch.max(decoder_position_ids).item()
                decoder_position_ids = torch.ones_like(batch_decoder_input[0, :, 0], dtype=torch.int64, device=model.device).cumsum(0) - 1 + max_position_id

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)

        for j in range(len(batch_predictions)):
            layout_results.append(LayoutResult(
                bboxes=batch_predictions[j],
                image_bbox=[0, 0, orig_sizes[j][0], orig_sizes[j][1]],
                confidence=int(sequence_scores[j].item())
            ))

        del encoder_text_hidden_states

    return layout_results






