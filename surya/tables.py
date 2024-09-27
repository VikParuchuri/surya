from copy import deepcopy
from typing import List
import torch
from PIL import Image

from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.schema import TableResult, TableCell
from surya.settings import settings
from tqdm import tqdm
import numpy as np
from surya.model.table_rec.config import SPECIAL_TOKENS


def get_batch_size():
    batch_size = settings.ORDER_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 32
    return batch_size


def sort_bboxes(bboxes, tolerance=1):
    vertical_groups = {}
    for block in bboxes:
        group_key = round(block[1] / tolerance) * tolerance
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(block)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_page_blocks = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(group, key=lambda x: x[0])
        sorted_page_blocks.extend(sorted_group)

    return sorted_page_blocks


def batch_table_recognition(images: List, bboxes: List[List[List[float]]], model: OrderVisionEncoderDecoderModel, processor, batch_size=None) -> List[TableResult]:
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(bboxes)
    if batch_size is None:
        batch_size = get_batch_size()

    output_order = []
    for i in tqdm(range(0, len(images), batch_size), desc="Finding reading order"):
        batch_list_bboxes = deepcopy(bboxes[i:i+batch_size])
        batch_list_bboxes = [sort_bboxes(page_bboxes) for page_bboxes in batch_list_bboxes] # Sort bboxes before passing in

        batch_images = images[i:i+batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images
        current_batch_size = len(batch_images)

        orig_sizes = [image.size for image in batch_images]
        model_inputs = processor(images=batch_images, boxes=deepcopy(batch_list_bboxes))

        batch_pixel_values = model_inputs["pixel_values"]
        batch_bboxes = model_inputs["input_boxes"]
        batch_bbox_mask = model_inputs["input_boxes_mask"]
        batch_bbox_counts = model_inputs["input_boxes_counts"]

        batch_bboxes = torch.from_numpy(np.array(batch_bboxes, dtype=np.int32)).to(model.device)
        batch_bbox_mask = torch.from_numpy(np.array(batch_bbox_mask, dtype=np.int32)).to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_bbox_counts = torch.tensor(np.array(batch_bbox_counts), dtype=torch.long).to(model.device)

        # Setup inputs for the decoder
        batch_decoder_input = [[[model.config.decoder.bos_token_id] * 2] for _ in range(current_batch_size)]
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)
        inference_token_count = batch_decoder_input.shape[1]

        col_predictions = None
        row_predictions = None
        max_tokens = min(batch_bbox_counts[:, 1].max().item(), settings.TABLE_REC_MAX_BOXES)
        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :, 0], dtype=torch.int64, device=model.device).cumsum(0) - 1
        model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)
        model.text_encoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)

        with torch.inference_mode():
            encoder_hidden_states = model.encoder(pixel_values=batch_pixel_values).last_hidden_state
            text_encoder_hidden_states = model.text_encoder(
                input_boxes=batch_bboxes,
                input_boxes_counts=batch_bbox_counts,
                cache_position=None,
                attention_mask=batch_bbox_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                use_cache=False
            ).hidden_states

            token_count = 0
            while token_count < max_tokens:
                is_prefill = token_count == 0
                return_dict = model.decoder(
                    input_ids=batch_decoder_input,
                    encoder_hidden_states=text_encoder_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True,
                    prefill=is_prefill
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1
                row_logits = return_dict["row_logits"].detach()
                col_logits = return_dict["col_logits"].detach()

                row_preds = torch.argmax(row_logits[:, -1], dim=-1).unsqueeze(1)
                col_preds = torch.argmax(col_logits[:, -1], dim=-1).unsqueeze(1)

                if col_predictions is None:
                    col_predictions = col_preds
                else:
                    col_predictions = torch.cat([col_predictions, col_preds], dim=1)

                if row_predictions is None:
                    row_predictions = row_preds
                else:
                    row_predictions = torch.cat([row_predictions, row_preds], dim=1)

                batch_decoder_input = torch.cat([row_preds, col_preds], dim=1).unsqueeze(1)

                token_count += inference_token_count
                inference_token_count = batch_decoder_input.shape[1]

            # Get raw values without special tokens
            row_predictions -= SPECIAL_TOKENS
            col_predictions -= SPECIAL_TOKENS

        ls_row_predictions = []
        ls_col_predictions = []
        max_rows = []
        max_cols = []
        for z in range(current_batch_size):
            box_end_idx = batch_bbox_counts[z][1]
            row_preds = row_predictions[z][:box_end_idx]
            max_row = row_preds[0].item()
            row_preds = row_preds[1:].tolist()

            col_preds = col_predictions[z][:box_end_idx]
            max_col = col_preds[0].item()
            col_preds = col_preds[1:].tolist()

            ls_row_predictions.append(row_preds)
            ls_col_predictions.append(col_preds)
            max_rows.append(max_row)
            max_cols.append(max_col)

        assert len(ls_row_predictions) == len(ls_col_predictions) == len(max_rows) == len(max_cols) == len(batch_images)
        for j, (row_pred, col_pred, max_row, max_col, row_bboxes) in enumerate(zip(ls_row_predictions, ls_col_predictions, max_rows, max_cols, batch_list_bboxes)):
            orig_size = orig_sizes[j]
            out_data = []
            # They either match up, or there are too many bboxes passed in
            assert (len(row_pred) == len(col_pred) == len(row_bboxes)) or len(row_bboxes) > settings.TABLE_REC_MAX_BOXES
            for z, (row_idx, col_idx, bbox) in enumerate(zip(row_pred, col_pred, row_bboxes)):
                    cell = TableCell(
                        bbox=bbox,
                        col_id=col_idx,
                        row_id=row_idx
                    )
                    out_data.append(cell)

            result = TableResult(
                cells=out_data,
                image_bbox=[0, 0, orig_size[0], orig_size[1]],
            )
            output_order.append(result)
    return output_order