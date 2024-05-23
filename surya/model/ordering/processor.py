from copy import deepcopy
from typing import Dict, Union, Optional, List, Tuple

import torch
from torch import TensorType
from transformers import DonutImageProcessor, DonutProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, \
    valid_images, to_numpy_array
import numpy as np
from PIL import Image
import PIL
from surya.settings import settings


def load_processor(checkpoint=settings.ORDER_MODEL_CHECKPOINT):
    processor = OrderImageProcessor.from_pretrained(checkpoint)
    processor.size = settings.ORDER_IMAGE_SIZE
    box_size = 1024
    max_tokens = 256
    processor.token_sep_id = max_tokens + box_size + 1
    processor.token_pad_id = max_tokens + box_size + 2
    processor.max_boxes = settings.ORDER_MAX_BOXES - 1
    processor.box_size = {"height": box_size, "width": box_size}
    return processor


class OrderImageProcessor(DonutImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs.get("patch_size", (4, 4))

    def process_inner(self, images: List[np.ndarray]):
        images = [img.transpose(2, 0, 1) for img in images] # convert to CHW format

        assert images[0].shape[0] == 3 # RGB input images, channel dim last

        # Convert to float32 for rescale/normalize
        images = [img.astype(np.float32) for img in images]

        # Rescale and normalize
        images = [
            self.rescale(img, scale=self.rescale_factor, input_data_format=ChannelDimension.FIRST)
            for img in images
        ]
        images = [
            self.normalize(img, mean=self.image_mean, std=self.image_std, input_data_format=ChannelDimension.FIRST)
            for img in images
        ]

        return images

    def process_boxes(self, boxes):
        padded_boxes = []
        box_masks = []
        box_counts = []
        for b in boxes:
            # Left pad for generation
            padded_b = deepcopy(b)
            padded_b.append([self.token_sep_id] * 4) # Sep token to indicate start of label predictions
            padded_boxes.append(padded_b)

        max_boxes = max(len(b) for b in padded_boxes)
        for i in range(len(padded_boxes)):
            pad_len = max_boxes - len(padded_boxes[i])
            box_len = len(padded_boxes[i])
            box_mask = [0] * pad_len + [1] * box_len
            padded_box = [[self.token_pad_id] * 4] * pad_len + padded_boxes[i]
            padded_boxes[i] = padded_box
            box_masks.append(box_mask)
            box_counts.append([pad_len, max_boxes])

        return padded_boxes, box_masks, box_counts

    def resize_img_and_boxes(self, img, boxes):
        orig_dim = img.size
        new_size = (self.size["width"], self.size["height"])
        img.thumbnail(new_size, Image.Resampling.LANCZOS)  # Shrink largest dimension to fit new size
        img = img.resize(new_size, Image.Resampling.LANCZOS)  # Stretch smaller dimension to fit new size

        img = np.asarray(img, dtype=np.uint8)

        width, height = orig_dim
        box_width, box_height = self.box_size["width"], self.box_size["height"]
        for box in boxes:
            # Rescale to 0-1024
            box[0] = box[0] / width * box_width
            box[1] = box[1] / height * box_height
            box[2] = box[2] / width * box_width
            box[3] = box[3] / height * box_height

            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > box_width:
                box[2] = box_width
            if box[3] > box_height:
                box[3] = box_height

        return img, boxes

    def preprocess(
        self,
        images: ImageInput,
        boxes: List[List[int]],
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

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        new_images = []
        new_boxes = []
        for img, box in zip(images, boxes):
            if len(box) > self.max_boxes:
                raise ValueError(f"Too many boxes, max is {self.max_boxes}")
            img, box = self.resize_img_and_boxes(img, box)
            new_images.append(img)
            new_boxes.append(box)

        images = new_images
        boxes = new_boxes

        # Convert to numpy for later processing steps
        images = [np.array(image) for image in images]

        images = self.process_inner(images)
        boxes, box_mask, box_counts = self.process_boxes(boxes)
        data = {
            "pixel_values": images,
            "input_boxes": boxes,
            "input_boxes_mask": box_mask,
            "input_boxes_counts": box_counts,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)