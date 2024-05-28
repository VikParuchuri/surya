from typing import Dict, Union, Optional, List, Iterable

import cv2
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


def load_processor():
    processor = SuryaProcessor()
    processor.image_processor.train = False
    processor.image_processor.max_size = settings.RECOGNITION_IMAGE_SIZE
    processor.tokenizer.model_max_length = settings.RECOGNITION_MAX_TOKENS
    return processor


class SuryaImageProcessor(DonutImageProcessor):
    def __init__(self, *args, max_size=None, train=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs.get("patch_size", (4, 4))
        self.max_size = max_size
        self.train = train

    @classmethod
    def numpy_resize(cls, image: np.ndarray, size, interpolation=cv2.INTER_LANCZOS4):
        height, width = image.shape[:2]
        max_width, max_height = size["width"], size["height"]

        if (height == max_height and width <= max_width) or (width == max_width and height <= max_height):
            image = image.transpose(2, 0, 1)
            return image

        scale = min(max_width / width, max_height / height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        resized_image = resized_image.transpose(2, 0, 1)

        return resized_image

    def process_inner(self, images: List[np.ndarray]):
        assert images[0].shape[2] == 3 # RGB input images, channel dim last

        # Rotate if the bbox is wider than it is tall
        images = [SuryaImageProcessor.align_long_axis(image, size=self.max_size, input_data_format=ChannelDimension.LAST) for image in images]

        # Verify that the image is wider than it is tall
        for img in images:
            assert img.shape[1] >= img.shape[0]

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
        tokenizer = Byt5LangTokenizer()
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        lang = kwargs.pop("lang", None)

        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)

        if text is not None:
            encodings = self.tokenizer(text, lang, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            inputs["langs"] = encodings["langs"]
            return inputs