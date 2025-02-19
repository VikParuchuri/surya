from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.detection.processor import SegformerImageProcessor

from surya.detection.model.config import EfficientViTConfig
from surya.detection.model.encoderdecoder import EfficientViTForSemanticSegmentation
from surya.settings import settings


class DetectionModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.DETECTOR_MODEL_CHECKPOINT

    def model(
            self,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype | str] = None
    ) -> EfficientViTForSemanticSegmentation:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = EfficientViTConfig.from_pretrained(self.checkpoint)
        model = EfficientViTForSemanticSegmentation.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            config=config,
        )
        model = model.to(device)
        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_DETECTOR:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling detection model {self.checkpoint} on device {device} with dtype {dtype}")
            compile_args = {'backend': 'openxla'} if device == 'xla' else {}
            model = torch.compile(model, **compile_args)

        print(f"Loaded detection model {self.checkpoint} on device {device} with dtype {dtype}")
        return model

    def processor(self) -> SegformerImageProcessor:
        return SegformerImageProcessor.from_pretrained(self.checkpoint)

class InlineDetectionModelLoader(DetectionModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        if checkpoint is None:
            checkpoint = settings.INLINE_MATH_MODEL_CHECKPOINT
        super().__init__(checkpoint)