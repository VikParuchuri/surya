from copy import deepcopy
from typing import List, Dict
import torch
from PIL import Image

from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.schema import TableResult, TableCell, Bbox, TableCol, TableRow
from surya.settings import settings
from tqdm import tqdm
import numpy as np
from surya.model.table_rec.config import SPECIAL_TOKENS


def get_batch_size():
    batch_size = settings.TABLE_REC_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 64
    return batch_size


def sort_bboxes(bboxes, tolerance=1):
    vertical_groups = {}
    for block in bboxes:
        group_key = round(block["bbox"][1] / tolerance) * tolerance
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(block)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_page_blocks = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(group, key=lambda x: x["bbox"][0])
        sorted_page_blocks.extend(sorted_group)

    return sorted_page_blocks


def batch_table_recognition(images: List, table_cells: List[List[Dict]], model: OrderVisionEncoderDecoderModel, processor, batch_size=None) -> List[TableResult]:
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(table_cells)
    if batch_size is None:
        batch_size = get_batch_size()

    output_order = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing tables"):
        batch_table_cells = deepcopy(table_cells[i:i+batch_size])
        batch_table_cells = [sort_bboxes(page_bboxes) for page_bboxes in batch_table_cells] # Sort bboxes before passing in
        batch_list_bboxes = [[block["bbox"] for block in page] for page in batch_table_cells]

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

                batch_decoder_input = torch.cat([box_preds.unsqueeze(1), rowcol_preds.unsqueeze(1).unsqueeze(1)], dim=-1)

                for j, (pred, status) in enumerate(zip(batch_decoder_input, all_done)):
                    if not status:
                        batch_predictions[j].append(pred[0].tolist())

                token_count += inference_token_count
                inference_token_count = batch_decoder_input.shape[1]

        for j, (preds, input_cells, orig_size) in enumerate(zip(batch_predictions, batch_table_cells, orig_sizes)):
            img_w, img_h = orig_size
            width_scaler = img_w / model.config.decoder.out_box_size
            height_scaler = img_h / model.config.decoder.out_box_size

            # cx, cy to corners
            for i, pred in enumerate(preds):
                w = pred[2] / 2
                h = pred[3] / 2
                x1 = pred[0] - w
                y1 = pred[1] - h
                x2 = pred[0] + w
                y2 = pred[1] + h
                class_ = int(pred[4] - SPECIAL_TOKENS)

                preds[i] = [x1 * width_scaler, y1 * height_scaler, x2 * width_scaler, y2 * height_scaler, class_]

            # Get rows and columns
            bb_rows = [p[:4] for p in preds if p[4] == 0]
            bb_cols = [p[:4] for p in preds if p[4] == 1]

            rows = []
            cols = []
            for row_idx, row in enumerate(bb_rows):
                rows.append(TableRow(
                    bbox=row,
                    row_id=row_idx
                ))

            for col_idx, col in enumerate(bb_cols):
                cols.append(TableCol(
                    bbox=col,
                    col_id=col_idx,
                ))

            # Assign cells to rows/columns
            cells = []
            for cell in input_cells:
                cells.append(
                    TableCell(
                        bbox=cell["bbox"],
                        text=cell.get("text"),
                    )
                )

            result = TableResult(
                cells=cells,
                rows=rows,
                cols=cols,
                image_bbox=[0, 0, img_w, img_h],
            )

            output_order.append(result)

    return output_order