import math
import random
from typing import Dict, Union, Optional, List, Iterable

import cv2
import torch
from torch import TensorType
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import pad, normalize
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, get_image_size
import numpy as np
from PIL import Image
import PIL
from surya.model.recognition.tokenizer import Byt5LangTokenizer
from surya.settings import settings
from surya.model.table_rec.config import BOX_DIM, SPECIAL_TOKENS


def load_processor():
    processor = SuryaProcessor()
    processor.image_processor.train = False
    processor.image_processor.max_size = settings.TABLE_REC_IMAGE_SIZE

    processor.token_pad_id = 0
    processor.token_eos_id = 1
    processor.token_bos_id = 2
    processor.token_row_id = 3
    processor.token_unused_id = 4
    processor.box_size = (BOX_DIM, BOX_DIM)
    processor.special_token_count = SPECIAL_TOKENS
    return processor


class SuryaImageProcessor(DonutImageProcessor):
    def __init__(self, *args, max_size=None, train=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs.get("patch_size", (4, 4))
        self.max_size = max_size
        self.train = train

    @classmethod
    def numpy_resize(cls, image: np.ndarray, size, interpolation=cv2.INTER_LANCZOS4):
        max_width, max_height = size["width"], size["height"]

        resized_image = cv2.resize(image, (max_width, max_height), interpolation=interpolation)
        resized_image = resized_image.transpose(2, 0, 1)

        return resized_image

    def process_inner(self, images: List[np.ndarray]):
        assert images[0].shape[2] == 3 # RGB input images, channel dim last

        # This also applies the right channel dim format, to channel x height x width
        images = [SuryaImageProcessor.numpy_resize(img, self.max_size, self.resample) for img in images]
        assert images[0].shape[0] == 3 # RGB input images, channel dim first

        # Convert to float32 for rescale/normalize
        images = [img.astype(np.float32) for img in images]

        # Pads with 255 (whitespace)
        # Pad to max size to improve performance
        max_size = self.max_size
        images = [
            SuryaImageProcessor.pad_image(
                image=image,
                size=max_size,
                input_data_format=ChannelDimension.FIRST,
                pad_value=settings.RECOGNITION_PAD_VALUE
            )
            for image in images
        ]
        # Rescale and normalize
        for idx in range(len(images)):
            images[idx] = images[idx] * self.rescale_factor
        images = [
            SuryaImageProcessor.normalize(img, mean=self.image_mean, std=self.image_std, input_data_format=ChannelDimension.FIRST)
            for img in images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        images = make_list_of_images(images)

        # Convert to numpy for later processing steps
        images = [np.array(img) for img in images]
        images = self.process_inner(images)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def pad_image(
        cls,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        assert delta_width >= 0 and delta_height >= 0

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format, constant_values=pad_value)

    @classmethod
    def align_long_axis(
        cls,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        input_height, input_width = image.shape[:2]
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        return image

    @classmethod
    def normalize(
        cls,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        return normalize(
            image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs
        )


class SuryaProcessor(DonutProcessor):
    def __init__(self, image_processor=None, tokenizer=None, train=False, **kwargs):
        image_processor = SuryaImageProcessor.from_pretrained(settings.RECOGNITION_MODEL_CHECKPOINT)
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
