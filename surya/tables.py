from copy import deepcopy
from typing import List
import torch
from PIL import Image

from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.schema import TableResult, TableCell
from surya.settings import settings
from tqdm import tqdm
import numpy as np


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

        col_predictions = []
        row_predictions = []
        max_rows = []
        max_cols = []
        with torch.inference_mode():
            return_dict = model(
                pixel_values=batch_pixel_values,
                decoder_input_boxes=batch_bboxes,
                decoder_input_boxes_mask=batch_bbox_mask,
                decoder_input_boxes_counts=batch_bbox_counts,
                encoder_outputs=None,
                past_key_values=None,
            )
            row_logits = return_dict["row_logits"].detach()
            col_logits = return_dict["col_logits"].detach()

            for z in range(len(batch_images)):
                box_start_idx = batch_bbox_counts[z][0]
                row_preds = row_logits[z][box_start_idx:].argmax(dim=-1)
                max_row = row_preds[0]
                row_preds = row_preds[1:]

                col_preds = col_logits[z][box_start_idx:].argmax(dim=-1)
                max_col = col_preds[0]
                col_preds = col_preds[1:]

                row_predictions.append(row_preds)
                col_predictions.append(col_preds)
                max_rows.append(max_row)
                max_cols.append(max_col)

        assert len(row_predictions) == len(col_predictions) == len(max_rows) == len(max_cols) == len(batch_images)
        for j, (row_pred, col_pred, max_row, max_col, row_bboxes) in enumerate(zip(row_predictions, col_predictions, max_rows, max_cols, batch_list_bboxes)):
            orig_size = orig_sizes[j]
            out_data = []
            assert len(row_pred) == len(col_pred) == len(row_bboxes)
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