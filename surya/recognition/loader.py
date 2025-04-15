from typing import Optional

import torch
from transformers import AutoImageProcessor

from surya.common.load import ModelLoader
from surya.common.surya.config import SuryaModelConfig
from surya.common.surya import SuryaModel
from surya.common.surya.processor import SuryaOCRProcessor
from surya.common.surya.processor.tokenizer import SuryaOCRTokenizer
from surya.settings import settings


class RecognitionModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.RECOGNITION_MODEL_CHECKPOINT

    def model(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE_BFLOAT
    ) -> SuryaModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE_BFLOAT

        model = SuryaModel.from_pretrained(self.checkpoint, torch_dtype=dtype).to(
            device
        )
        model = model.eval()

        model.config.decoder._attn_implementation = "sdpa"

        if settings.COMPILE_ALL or settings.COMPILE_RECOGNITION:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            print(
                f"Compiling recognition model {self.checkpoint} on device {device} with dtype {dtype}"
            )
            compile_args = {"backend": "openxla"} if device == "xla" else {}
            model.vision_encoder = torch.compile(model.vision_encoder, **compile_args)
            model.decoder = torch.compile(model.decoder, **compile_args)

        print(
            f"Loaded recognition model {self.checkpoint} on device {model.device} with dtype {dtype}, using attention mechanism {model.config.decoder._attn_implementation}.  Quantizing kv cache: {settings.RECOGNITION_MODEL_QUANTIZE}."
        )
        return model

    def processor(self) -> SuryaOCRProcessor:
        config: SuryaModelConfig = SuryaModelConfig.from_pretrained(self.checkpoint)

        # Workaround since load_pretrained isn't working for our processor - TODO Fix
        image_processor = AutoImageProcessor.from_pretrained(
            self.checkpoint, use_fast=False
        )
        ocr_tokenizer = SuryaOCRTokenizer(
            special_tokens=config.special_ocr_tokens, model_checkpoint=self.checkpoint
        )

        processor = SuryaOCRProcessor(
            image_processor=image_processor,
            ocr_tokenizer=ocr_tokenizer,
            tile_size=config.tile_size,
            image_tokens_per_tile=config.vision_encoder.num_patches,
            blank_bbox_token_id=config.blank_bbox_token_id,
            num_register_tokens=config.num_register_tokens,
            sequence_length=None,
        )
        config.eos_token_id = processor.eos_token_id
        config.pad_token_id = processor.pad_token_id
        config.bos_token_id = processor.bos_token_id

        return processor
