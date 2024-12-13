from typing import List
import torch
from PIL import Image
from torch import nn

from surya.model.table_rec.columns import find_columns
from surya.model.table_rec.encoderdecoder import TableRecEncoderDecoderModel
from surya.model.table_rec.shaper import LabelShaper
from surya.schema import TableResult, TableCell, TableRow
from surya.settings import settings
from tqdm import tqdm
import numpy as np
from surya.model.table_rec.config import SPECIAL_TOKENS, CATEGORY_TO_ID, BOX_PROPERTIES, BOX_DIM, MERGE_KEYS


def get_batch_size():
    batch_size = settings.TABLE_REC_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 128
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


def pad_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    repeats = (pad_size + current_batch_size - 1) // current_batch_size
    repeated_rows = tensor.repeat((repeats, *[1] * (tensor.dim() - 1)))
    pad_tensor = repeated_rows[:pad_size]

    return torch.cat([tensor, pad_tensor], dim=0)


def inference_loop(
        model: nn.Module,
        encoder_hidden_states: torch.Tensor,
        batch_input_ids: torch.Tensor,
        current_batch_size: int,
        batch_size: int
):
    shaper = LabelShaper()
    batch_predictions = [[] for _ in range(current_batch_size)]
    max_tokens = settings.TABLE_REC_MAX_BOXES
    decoder_position_ids = torch.ones_like(batch_input_ids[0, :, 0], dtype=torch.int64, device=model.device).cumsum(
        0) - 1
    inference_token_count = batch_input_ids.shape[1]

    if settings.TABLE_REC_STATIC_CACHE:
        encoder_hidden_states = pad_to_batch_size(encoder_hidden_states, batch_size)
        batch_input_ids = pad_to_batch_size(batch_input_ids, batch_size)

    model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)

    print(batch_input_ids)
    with torch.inference_mode():
        token_count = 0
        all_done = torch.zeros(current_batch_size, dtype=torch.bool)

        while token_count < max_tokens:
            is_prefill = token_count == 0
            return_dict = model.decoder(
                input_ids=batch_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                cache_position=decoder_position_ids,
                use_cache=True,
                prefill=is_prefill
            )

            decoder_position_ids = decoder_position_ids[-1:] + 1

            # Get predictions for each box element
            box_properties = []
            done = []
            for j in range(current_batch_size):
                box_property = {}
                for (k, kcount, mode) in BOX_PROPERTIES:
                    k_logits = return_dict["box_property_logits"][k][j, -1, :]
                    if mode == "classification":
                        item = int(torch.argmax(k_logits, dim=-1).item())
                        if k == "category":
                            done.append(item == model.decoder.config.eos_token_id or item == model.decoder.config.pad_token_id)
                        item -= SPECIAL_TOKENS
                        box_property[k] = item
                    elif mode == "regression":
                        if k == "bbox":
                            k_logits *= BOX_DIM
                        elif k == "colspan":
                            k_logits = k_logits.clamp(min=1)
                        box_property[k] = k_logits.tolist()
                box_properties.append(box_property)

            print(box_properties[0])
            all_done = all_done | torch.tensor(done, dtype=torch.bool)

            if all_done.all():
                break

            batch_input_ids = torch.tensor(shaper.dict_to_labels(box_properties), dtype=torch.long).to(model.device)
            batch_input_ids = batch_input_ids.unsqueeze(1) # Add sequence length dimension

            for j, (box_property, status) in enumerate(zip(box_properties, all_done)):
                if not status:
                    batch_predictions[j].append(box_property)

            token_count += inference_token_count
            inference_token_count = batch_input_ids.shape[1]

            if settings.TABLE_REC_STATIC_CACHE:
                batch_input_ids = pad_to_batch_size(batch_input_ids, batch_size)
    return batch_predictions


def batch_table_recognition(images: List, model: TableRecEncoderDecoderModel, processor, batch_size=None) -> List[TableResult]:
    assert all([isinstance(image, Image.Image) for image in images])
    if batch_size is None:
        batch_size = get_batch_size()

    if len(images) == 0:
        return []

    query_items = []
    for image in images:
        query_items.append({
            "polygon": [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]],
            "category": CATEGORY_TO_ID["Table"],
            "colspan": 0,
            "merges": 0,
        })

    output_order = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing tables"):
        batch_query_items = query_items[i:i+batch_size]

        batch_images = images[i:i+batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images

        current_batch_size = len(batch_images)

        orig_sizes = [image.size for image in batch_images]
        model_inputs = processor(images=batch_images, query_items=batch_query_items)

        batch_pixel_values = model_inputs["pixel_values"]

        batch_input_ids = model_inputs["input_ids"].to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)

        shaper = LabelShaper()

        # We only need to process each image once
        with torch.inference_mode():
            encoder_hidden_states = model.encoder(pixel_values=batch_pixel_values).last_hidden_state

        row_predictions = inference_loop(model, encoder_hidden_states, batch_input_ids, current_batch_size, batch_size)

        row_query_items = []
        row_encoder_hidden_states = []
        idx_map = []
        for j, img_predictions in enumerate(row_predictions):
            for row_prediction in img_predictions:
                polygon = shaper.convert_bbox_to_polygon(row_prediction["bbox"])
                row_query_items.append({
                    "polygon": polygon,
                    "category": CATEGORY_TO_ID["Table-row"],
                    "colspan": 0,
                    "merges": 0,
                })
                row_encoder_hidden_states.append(encoder_hidden_states[j])
                idx_map.append(j)

        row_encoder_hidden_states = torch.stack(row_encoder_hidden_states)
        row_inputs = processor(images=None, query_items=row_query_items, convert_images=False)
        row_input_ids = row_inputs["input_ids"].to(model.device)
        cell_predictions = []
        """
        for j in tqdm(range(0, len(row_input_ids), batch_size), desc="Recognizing tables"):
            cell_batch_hidden_states = row_encoder_hidden_states[j:j+batch_size]
            cell_batch_input_ids = row_input_ids[j:j+batch_size]
            cell_batch_size = len(cell_batch_input_ids)
            cell_predictions.extend(
                inference_loop(model, cell_batch_hidden_states, cell_batch_input_ids, cell_batch_size, batch_size)
            )
        """

        for j, (img_predictions, orig_size) in enumerate(zip(row_predictions, orig_sizes)):
            row_cell_predictions = [c for i,c in enumerate(cell_predictions) if idx_map[i] == j]
            # Each row prediction matches a cell prediction
            #assert len(img_predictions) == len(row_cell_predictions)
            rows = []
            cells = []
            for z, row_prediction in enumerate(img_predictions):
                polygon = shaper.convert_bbox_to_polygon(row_prediction["bbox"])
                polygon = processor.resize_polygon(polygon, (BOX_DIM, BOX_DIM), orig_size)
                rows.append(TableRow(
                    polygon=polygon,
                    row_id=z
                ))
                """
                for l, cell in enumerate(row_cell_predictions[z]):
                    polygon = shaper.convert_bbox_to_polygon(cell["bbox"])
                    polygon = processor.resize_polygon(polygon, (BOX_DIM, BOX_DIM), orig_size)
                    cells.append(
                        TableCell(
                            polygon=polygon,
                            row_id=z,
                            within_row_id=l,
                            colspan=max(1, int(cell["colspan"])),
                            merge_up=cell["merges"] in [MERGE_KEYS["merge_up"], MERGE_KEYS["merge_both"]],
                            merge_down=cell["merges"] in [MERGE_KEYS["merge_down"], MERGE_KEYS["merge_both"]],
                        )
                    )
                """
            columns = find_columns(rows, cells)

            result = TableResult(
                cells=cells,
                rows=rows,
                cols=columns,
                image_bbox=[0, 0, orig_size[0], orig_size[1]],
            )
            output_order.append(result)

    return output_order