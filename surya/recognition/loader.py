from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.recognition.model.config import SuryaOCRConfig, SuryaOCRDecoderConfig, DonutSwinConfig, SuryaOCRTextEncoderConfig
from surya.recognition.model.encoderdecoder import OCREncoderDecoderModel
from surya.recognition.processor import SuryaProcessor
from surya.settings import settings

torch.backends.cuda.enable_cudnn_sdp(settings.ENABLE_CUDNN_ATTENTION)
if not settings.ENABLE_EFFICIENT_ATTENTION:
    print("Efficient attention is disabled. This will use significantly more VRAM.")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

class RecognitionModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.RECOGNITION_MODEL_CHECKPOINT

        self.checkpoint, self.revision = self.split_checkpoint_revision(self.checkpoint)

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=settings.MODEL_DTYPE
    ) -> OCREncoderDecoderModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = SuryaOCRConfig.from_pretrained(self.checkpoint, revision=self.revision)
        decoder_config = config.decoder
        decoder = SuryaOCRDecoderConfig(**decoder_config)
        config.decoder = decoder

        encoder_config = config.encoder
        encoder = DonutSwinConfig(**encoder_config)
        config.encoder = encoder

        text_encoder_config = config.text_encoder
        text_encoder = SuryaOCRTextEncoderConfig(**text_encoder_config)
        config.text_encoder = text_encoder

        model = OCREncoderDecoderModel.from_pretrained(self.checkpoint, config=config, torch_dtype=dtype, revision=self.revision)
        model = model.to(device)
        model = model.eval()

        if settings.RECOGNITION_STATIC_CACHE:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling recognition model {self.checkpoint} on device {device} with dtype {dtype}")
            model.encoder = torch.compile(model.encoder)
            model.decoder = torch.compile(model.decoder)
            model.text_encoder = torch.compile(model.text_encoder)

        print(f"Loaded recognition model {self.checkpoint} on device {device} with dtype {dtype}")
        return model

    def processor(self) -> SuryaProcessor:
        return SuryaProcessor(self.checkpoint, self.revision)

