from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.settings import settings
from surya.texify.model.config import TexifyConfig, TexifyDecoderConfig, TexifyEncoderConfig

from surya.texify.model.encoderdecoder import TexifyModel
from surya.texify.processor import TexifyProcessor


class TexifyModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.TEXIFY_MODEL_CHECKPOINT

        self.checkpoint, self.revision = self.split_checkpoint_revision(self.checkpoint)

    def model(
            self,
            device=settings.TORCH_DEVICE_MODEL,
            dtype=settings.MODEL_DTYPE
    ) -> TexifyModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = TexifyConfig.from_pretrained(self.checkpoint, revision=self.revision)
        decoder_config = config.decoder
        decoder = TexifyDecoderConfig(**decoder_config)
        config.decoder = decoder

        encoder_config = config.encoder
        encoder = TexifyEncoderConfig(**encoder_config)
        config.encoder = encoder

        model = TexifyModel.from_pretrained(self.checkpoint, config=config, torch_dtype=dtype, revision=self.revision)

        model = model.to(device)
        model = model.eval()

        if settings.TABLE_REC_STATIC_CACHE:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling texify model {self.checkpoint} on device {device} with dtype {dtype}")
            model.encoder = torch.compile(model.encoder)
            model.decoder = torch.compile(model.decoder)

        print(f"Loaded texify model {self.checkpoint} on device {device} with dtype {dtype}")
        return model

    def processor(self) -> TexifyProcessor:
        return TexifyProcessor(self.checkpoint, self.revision)