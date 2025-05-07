from typing import Optional

import torch

from surya.common.donut.processor import SuryaEncoderImageProcessor
from surya.common.load import ModelLoader
from surya.layout.model.config import (
    SuryaLayoutConfig,
    SuryaLayoutDecoderConfig,
    DonutSwinLayoutConfig,
)
from surya.layout.model.encoderdecoder import SuryaLayoutModel
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


class LayoutModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.LAYOUT_MODEL_CHECKPOINT

    def model(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> SuryaLayoutModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = SuryaLayoutConfig.from_pretrained(self.checkpoint)
        decoder_config = config.decoder
        decoder = SuryaLayoutDecoderConfig(**decoder_config)
        config.decoder = decoder

        encoder_config = config.encoder
        encoder = DonutSwinLayoutConfig(**encoder_config)
        config.encoder = encoder

        model = SuryaLayoutModel.from_pretrained(
            self.checkpoint, config=config, torch_dtype=dtype
        )
        model = model.to(device)
        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_LAYOUT:
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            logger.info(
                f"Compiling layout model {self.checkpoint} on device {device} with dtype {dtype}"
            )
            compile_args = {"backend": "openxla"} if device == "xla" else {}
            model.encoder = torch.compile(model.encoder, **compile_args)
            model.decoder = torch.compile(model.decoder, **compile_args)

        logger.debug(
            f"Loaded layout model {self.checkpoint} from {SuryaLayoutModel.get_local_path(self.checkpoint)} onto device {device} with dtype {dtype}"
        )
        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> SuryaEncoderImageProcessor:
        processor = SuryaEncoderImageProcessor(max_size=settings.LAYOUT_IMAGE_SIZE)
        return processor
