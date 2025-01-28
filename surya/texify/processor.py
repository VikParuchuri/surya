from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast, ProcessorMixin

from surya.common.donut.processor import SuryaEncoderImageProcessor
from surya.settings import settings


class TexifyProcessor(ProcessorMixin):
    attributes = ["image_processor"]
    image_processor_class = "AutoImageProcessor"

    def __init__(self, checkpoint, revision, **kwargs):
        image_processor = SuryaEncoderImageProcessor.from_pretrained(checkpoint, revision=revision)
        image_processor.do_align_long_axis = False
        image_processor.max_size = settings.TEXIFY_IMAGE_SIZE
        self.image_processor = image_processor

        tokenizer = TexifyTokenizer.from_pretrained(checkpoint, revision=revision)
        tokenizer.model_max_length = settings.TEXIFY_MAX_TOKENS
        self.tokenizer = tokenizer

        super().__init__(image_processor)

    def __call__(
            self,
            images: List[Image.Image] | None,
            *args,
            **kwargs
    ):
        input_ids = [[self.tokenizer.bos_token_id]] * len(images)
        input_ids = torch.tensor(input_ids)

        pixel_values = self.image_processor(images, **kwargs)["pixel_values"]
        pixel_values = torch.tensor(np.array(pixel_values))

        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values
        }
        return inputs



class TexifyTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )