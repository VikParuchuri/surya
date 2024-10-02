from collections import defaultdict
from copy import deepcopy
from typing import List
import torch
from PIL import Image

from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.schema import TableResult, TableCell, Bbox
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


def cx_cy_to_corners(pred):
    w = pred[2] / 2
    h = pred[3] / 2
    x1 = pred[0] - w
    y1 = pred[1] - h
    x2 = pred[0] + w
    y2 = pred[1] + h

    return [x1, y1, x2, y2]


def corners_to_cx_cy(pred):
    x = (pred[0] + pred[2]) / 2
    y = (pred[1] + pred[3]) / 2
    w = pred[2] - pred[0]
    h = pred[3] - pred[1]

    return [x, y, w, h]


def snap_to_bboxes(rc_box, input_boxes, used_cells_row, used_cells_col, row=True, row_threshold=.2, col_threshold=.2):
    sel_bboxes = []
    for cell_idx, cell in enumerate(input_boxes):
        rc_corner_bbox = cx_cy_to_corners(rc_box)
        intersection_pct = Bbox(bbox=cell).intersection_pct(Bbox(bbox=rc_corner_bbox))

        if row:
            if cell_idx not in used_cells_row:
                if intersection_pct > row_threshold:
                    sel_bboxes.append(cell)
                    used_cells_row.add(cell_idx)
        else:
            if cell_idx not in used_cells_col:
                if intersection_pct > col_threshold:
                    sel_bboxes.append(cell)
                    used_cells_col.add(cell_idx)

    if len(sel_bboxes) == 0:
        return rc_box, used_cells_row, used_cells_col

    new_bbox = [
        min([b[0] for b in sel_bboxes]),
        min([b[1] for b in sel_bboxes]),
        max([b[2] for b in sel_bboxes]),
        max([b[3] for b in sel_bboxes])
    ]
    new_bbox = [
        max(new_bbox[0], rc_corner_bbox[0]),
        max(new_bbox[1], rc_corner_bbox[1]),
        min(new_bbox[2], rc_corner_bbox[2]),
        min(new_bbox[3], rc_corner_bbox[3])
    ]
    cx_cy_box = corners_to_cx_cy(new_bbox)
    return cx_cy_box, used_cells_row, used_cells_col



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
        batch_decoder_input = [[[model.config.decoder.bos_token_id] * 5] for _ in range(current_batch_size)]
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)
        inference_token_count = batch_decoder_input.shape[1]

        max_tokens = min(batch_bbox_counts[:, 1].max().item(), settings.TABLE_REC_MAX_BOXES)
        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :, 0], dtype=torch.int64, device=model.device).cumsum(0) - 1
        model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)
        model.text_encoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)

        batch_predictions = [[] for _ in range(current_batch_size)]
        used_cells_row = [set() for _ in range(current_batch_size)]
        used_cells_col = [set() for _ in range(current_batch_size)]

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
            all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)

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
                box_logits = return_dict["bbox_logits"][:, -1, :].detach()
                rowcol_logits = return_dict["class_logits"][:, -1, :].detach()

                rowcol_preds = torch.argmax(rowcol_logits, dim=-1)
                box_preds = torch.argmax(box_logits, dim=-1)

                done = (rowcol_preds == processor.tokenizer.eos_id) | (rowcol_preds == processor.tokenizer.pad_id)
                done = done
                all_done = all_done | done

                if all_done.all():
                    break

                for batch_idx, (box_pred, class_pred, ucr, ucc, bboxes, status) in enumerate(zip(box_preds, rowcol_preds, used_cells_row, used_cells_col, batch_list_bboxes, all_done)):
                    if status:
                        continue
                    class_pred = class_pred.item() - SPECIAL_TOKENS
                    nb = processor.resize_boxes(batch_images[batch_idx], deepcopy(bboxes))
                    is_row = class_pred == 0
                    new_bbox, ucr, ucc = snap_to_bboxes(box_pred.tolist(), nb, ucr, ucc, row=is_row)
                    new_bbox = torch.tensor(new_bbox, dtype=torch.long, device=model.device)
                    box_preds[batch_idx] = new_bbox

                    used_cells_row[batch_idx] = ucr
                    used_cells_col[batch_idx] = ucc

                batch_decoder_input = torch.cat([box_preds.unsqueeze(1), rowcol_preds.unsqueeze(1).unsqueeze(1)], dim=-1)

                for j, (pred, status) in enumerate(zip(batch_decoder_input, all_done)):
                    if not status:
                        batch_predictions[j].append(pred[0].tolist())

                token_count += inference_token_count
                inference_token_count = batch_decoder_input.shape[1]
        """
        for j, (preds, bboxes, orig_size) in enumerate(zip(batch_predictions, batch_list_bboxes, orig_sizes)):
            out_data = []
            # They either match up, or there are too many bboxes passed in
            img_w, img_h = orig_size
            # cx, cy to corners
            for i, pred in enumerate(preds):
                scale_w = img_w / model.config.decoder.out_box_size
                scale_h = img_h / model.config.decoder.out_box_size
                class_ = int(pred[4] - SPECIAL_TOKENS)
                pred = cx_cy_to_corners(pred)

                preds[i] = [pred[0] * scale_w, pred[1] * scale_h, pred[2] * scale_w, pred[3] * scale_h, class_]

            rows = [p[:4] for p in preds if p[4] == 0]
            cols = [p[:4] for p in preds if p[4] == 1]

            for cell in bboxes:
                max_intersection = 0
                row_pred = -1
                for row_idx, row in enumerate(rows):
                    intersection_pct = Bbox(bbox=cell).intersection_pct(Bbox(bbox=row))
                    if intersection_pct > max_intersection:
                        max_intersection = intersection_pct
                        row_pred = row_idx

                max_intersection = 0
                col_pred = -1
                for col_idx, col in enumerate(cols):
                    intersection_pct = Bbox(bbox=cell).intersection_pct(Bbox(bbox=col))
                    if intersection_pct > max_intersection:
                        max_intersection = intersection_pct
                        col_pred = col_idx

                cell = TableCell(
                    bbox=cell,
                    col_id=col_pred,
                    row_id=row_pred
                )
                out_data.append(cell)

            result = TableResult(
                cells=out_data,
                image_bbox=[0, 0, img_w, img_h],
            )

            output_order.append(result)
        """
        for j, (preds, bboxes, orig_size) in enumerate(zip(batch_predictions, batch_list_bboxes, orig_sizes)):
            out_data = []
            # They either match up, or there are too many bboxes passed in
            img_w, img_h = orig_size
            # cx, cy to corners
            for i, pred in enumerate(preds):
                width_scaler = img_w / model.config.decoder.out_box_size
                height_scaler = img_h / model.config.decoder.out_box_size
                w = pred[2] / 2
                h = pred[3] / 2
                x1 = pred[0] - w
                y1 = pred[1] - h
                x2 = pred[0] + w
                y2 = pred[1] + h
                class_ = int(pred[4] - SPECIAL_TOKENS)

                preds[i] = [x1 * width_scaler, y1 * height_scaler, x2 * width_scaler, y2 * height_scaler, class_]

            rows = [p[:4] for p in preds if p[4] == 0]
            cols = [p[:4] for p in preds if p[4] == 1]
            for row_idx, row in enumerate(rows):
                cell = TableCell(
                    bbox=row,
                    col_id=-1,
                    row_id=row_idx
                )
                out_data.append(cell)

            for col_idx, col in enumerate(cols):
                cell = TableCell(
                    bbox=col,
                    col_id=col_idx,
                    row_id=-1
                )
                out_data.append(cell)

            result = TableResult(
                cells=out_data,
                image_bbox=[0, 0, img_w, img_h],
            )

            output_order.append(result)

    return output_order