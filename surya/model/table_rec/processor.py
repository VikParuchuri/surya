import math
import random
import torch
from transformers import DonutProcessor

from surya.model.common.donut.processor import SuryaEncoderImageProcessor
from surya.model.recognition.tokenizer import Byt5LangTokenizer
from surya.settings import settings
from surya.model.table_rec.config import BOX_DIM, SPECIAL_TOKENS


def load_processor():
    processor = SuryaProcessor()

    processor.token_pad_id = 0
    processor.token_eos_id = 1
    processor.token_bos_id = 2
    processor.token_row_id = 3
    processor.token_unused_id = 4
    processor.box_size = (BOX_DIM, BOX_DIM)
    processor.special_token_count = SPECIAL_TOKENS
    return processor


class SuryaProcessor(DonutProcessor):
    def __init__(self, image_processor=None, tokenizer=None, train=False, **kwargs):
        image_processor = SuryaEncoderImageProcessor.from_pretrained(settings.RECOGNITION_MODEL_CHECKPOINT)
        image_processor.do_align_long_axis = False
        image_processor.max_size = settings.TABLE_REC_IMAGE_SIZE
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")

        tokenizer = Byt5LangTokenizer()
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False
        self.max_input_boxes = kwargs.get("max_input_boxes", settings.TABLE_REC_MAX_ROWS)
        self.extra_input_boxes = kwargs.get("extra_input_boxes", 32)

    def resize_boxes(self, img, boxes):
        width, height = img.size
        box_width, box_height = self.box_size
        for box in boxes:
            # Rescale to 0-1024
            box[0] = math.ceil(box[0] / width * box_width)
            box[1] = math.ceil(box[1] / height * box_height)
            box[2] = math.floor(box[2] / width * box_width)
            box[3] = math.floor(box[3] / height * box_height)

            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > box_width:
                box[2] = box_width
            if box[3] > box_height:
                box[3] = box_height

        boxes = [b for b in boxes if b[3] > b[1] and b[2] > b[0]]
        boxes = [b for b in boxes if (b[3] - b[1]) * (b[2] - b[0]) > 10]

        return boxes

    def __call__(self, *args, **kwargs):
        images = kwargs.pop("images", [])
        boxes = kwargs.pop("boxes", [])
        assert len(images) == len(boxes)

        if len(args) > 0:
            images = args[0]
            args = args[1:]

        for i in range(len(boxes)):
            random.seed(1)
            if len(boxes[i]) > self.max_input_boxes:
                downsample_ratio = self.max_input_boxes / len(boxes[i])
                boxes[i] = [b for b in boxes[i] if random.random() < downsample_ratio]
            boxes[i] = boxes[i][:self.max_input_boxes]

        new_boxes = []
        max_len = self.max_input_boxes + self.extra_input_boxes
        box_masks = []
        box_ends = []
        for i in range(len(boxes)):
            nb = self.resize_boxes(images[i], boxes[i])
            nb = [[b + self.special_token_count for b in box] for box in nb] # shift up
            nb = nb[:self.max_input_boxes - 1]

            nb.insert(0, [self.token_row_id] * 4) # Insert special token for max rows/cols
            for _ in range(self.extra_input_boxes):
                nb.append([self.token_unused_id] * 4)

            pad_length = max_len - len(nb)
            box_mask = [1] * len(nb) + [1] * (pad_length)
            box_ends.append(len(nb))
            nb = nb + [[self.token_unused_id] * 4] * pad_length

            new_boxes.append(nb)
            box_masks.append(box_mask)

        box_ends = torch.tensor(box_ends, dtype=torch.long)
        box_starts = torch.tensor([0] * len(boxes), dtype=torch.long)
        box_ranges = torch.stack([box_starts, box_ends], dim=1)

        inputs = self.image_processor(images, *args, **kwargs)
        inputs["input_boxes"] = torch.tensor(new_boxes, dtype=torch.long)
        inputs["input_boxes_mask"] = torch.tensor(box_masks, dtype=torch.long)
        inputs["input_boxes_counts"] = box_ranges
        return inputs
