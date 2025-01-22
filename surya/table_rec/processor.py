from typing import List

import PIL
import torch
from transformers import ProcessorMixin

from surya.common.donut.processor import SuryaEncoderImageProcessor
from surya.table_rec.shaper import LabelShaper
from surya.settings import settings
from surya.table_rec.model.config import BOX_DIM, SPECIAL_TOKENS


class SuryaProcessor(ProcessorMixin):
    attributes = ["image_processor"]
    image_processor_class = "AutoImageProcessor"

    def __init__(self, checkpoint, revision, **kwargs):
        image_processor = SuryaEncoderImageProcessor.from_pretrained(checkpoint, revision=revision)
        image_processor.do_align_long_axis = False
        image_processor.max_size = settings.TABLE_REC_IMAGE_SIZE
        self.image_processor = image_processor
        super().__init__(image_processor)

        self.box_size = (BOX_DIM, BOX_DIM)
        self.special_token_count = SPECIAL_TOKENS
        self.shaper = LabelShaper()

    def resize_polygon(self, polygon, orig_size, new_size):
        w_scaler = new_size[0] / orig_size[0]
        h_scaler = new_size[1] / orig_size[1]

        for corner in polygon:
            corner[0] = corner[0] * w_scaler
            corner[1] = corner[1] * h_scaler

            if corner[0] < 0:
                corner[0] = 0
            if corner[1] < 0:
                corner[1] = 0
            if corner[0] > new_size[0]:
                corner[0] = new_size[0]
            if corner[1] > new_size[1]:
                corner[1] = new_size[1]

        return polygon

    def __call__(
            self,
            images: List[PIL.Image.Image] | None,
            query_items: List[dict],
            columns: List[dict] | None = None,
            convert_images: bool = True,
            *args,
            **kwargs
    ):
        if convert_images:
            assert len(images) == len(query_items)
            assert len(images) > 0

            # Resize input query items
            for image, query_item in zip(images, query_items):
                query_item["polygon"] = self.resize_polygon(query_item["polygon"], image.size, self.box_size)

        query_items = self.shaper.convert_polygons_to_bboxes(query_items)
        query_labels = self.shaper.dict_to_labels(query_items)

        decoder_input_boxes = []
        col_count = len(query_labels[0])
        for label in query_labels:
            decoder_input_boxes.append([
                [self.token_bos_id] * col_count,
                label,
                [self.token_query_end_id] * col_count
            ])

        # Add columns to end of decoder input
        if columns:
            columns = self.shaper.convert_polygons_to_bboxes(columns)
            column_labels = self.shaper.dict_to_labels(columns)
            for decoder_box in decoder_input_boxes:
                decoder_box += column_labels

        input_boxes = torch.tensor(decoder_input_boxes, dtype=torch.long)
        input_boxes_mask = torch.ones_like(input_boxes, dtype=torch.long)

        inputs = {
            "input_ids": input_boxes,
            "attention_mask": input_boxes_mask
        }
        if convert_images:
            inputs["pixel_values"] = self.image_processor(images, *args, **kwargs)["pixel_values"]
        return inputs
