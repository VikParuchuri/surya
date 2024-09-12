from copy import deepcopy
from typing import List
import torch
from PIL import Image

from surya.input.processing import convert_if_not_rgb
from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.schema import TableBox, TableResult
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


def batch_table_recognition(images: List, bboxes: List[List[List[float]]], model: OrderVisionEncoderDecoderModel, processor, batch_size=None) -> List[TableResult]:
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(bboxes)
    if batch_size is None:
        batch_size = get_batch_size()


    output_order = []
    for i in tqdm(range(0, len(images), batch_size), desc="Finding reading order"):
        batch_bboxes = deepcopy(bboxes[i:i+batch_size])
        batch_images = images[i:i+batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images

        orig_sizes = [image.size for image in batch_images]
        model_inputs = processor(images=batch_images, boxes=batch_bboxes)

        batch_pixel_values = model_inputs["pixel_values"]
        batch_bboxes = model_inputs["input_boxes"]
        batch_bbox_mask = model_inputs["input_boxes_mask"]
        batch_bbox_counts = model_inputs["input_boxes_counts"]

        batch_bboxes = torch.from_numpy(np.array(batch_bboxes, dtype=np.int32)).to(model.device)
        batch_bbox_mask = torch.from_numpy(np.array(batch_bbox_mask, dtype=np.int32)).to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_bbox_counts = torch.tensor(np.array(batch_bbox_counts), dtype=torch.long).to(model.device)

        token_count = 0
        past_key_values = None
        encoder_outputs = None
        batch_predictions = [[] for _ in range(len(batch_images))]
        done = torch.zeros(len(batch_images), dtype=torch.bool, device=model.device)

        with torch.inference_mode():
            while not done.all() and token_count < settings.TABLE_REC_MAX_BOXES:
                return_dict = model(
                    pixel_values=batch_pixel_values,
                    decoder_input_boxes=batch_bboxes,
                    decoder_input_boxes_mask=batch_bbox_mask,
                    decoder_input_boxes_counts=batch_bbox_counts,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                )
                logits = return_dict["logits"].detach()

                last_tokens = []
                last_token_mask = []
                for j in range(logits.shape[0]):
                    # Get token label prediction
                    pred = int(torch.argmax(logits[j, -1], dim=-1).item())

                    # Add one to avoid colliding with the 1000 height/width token for bboxes
                    token_done = pred == processor.token_eos_id

                    if token_done:
                        last_token_mask.append([0])
                        last_tokens.append([[pred + processor.box_size["height"]] * 4])
                        done[j] = True
                    else:
                        out_token = pred + processor.box_size["height"] + 1
                        if pred in [processor.token_row_id, processor.token_cell_id]:
                            out_token -= 1

                        last_tokens.append([[out_token] * 4])
                        last_token_mask.append([1])
                        batch_predictions[j].append(pred)  # Get rank prediction for given position

                past_key_values = return_dict["past_key_values"]
                encoder_outputs = (return_dict["encoder_last_hidden_state"],)

                batch_bboxes = torch.tensor(last_tokens, dtype=torch.long).to(model.device)
                token_bbox_mask = torch.tensor(last_token_mask, dtype=torch.long).to(model.device)
                batch_bbox_mask = torch.cat([batch_bbox_mask, token_bbox_mask], dim=1)
                token_count += 1

        for j, row_pred in enumerate(batch_predictions):
            row_bboxes = bboxes[i+j]

            orig_size = orig_sizes[j]
            out_data = []
            row_idx = 0
            col_idx = -1
            for z, token in enumerate(row_pred):
                if token == processor.token_cell_id:
                    col_idx += 1
                elif token == processor.token_row_id:
                    row_idx += 1
                    col_idx = -1
                elif 0 <= token < len(row_bboxes):
                    cell = TableBox(
                        bbox=row_bboxes[token],
                        col_id=col_idx,
                        row_id=row_idx
                    )
                    out_data.append(cell)

            result = TableResult(
                bboxes=out_data,
                image_bbox=[0, 0, orig_size[0], orig_size[1]],
            )
            output_order.append(result)
    return output_order






