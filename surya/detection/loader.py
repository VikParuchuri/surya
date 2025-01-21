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

        self.checkpoint, self.revision = self.split_checkpoint_revision(self.checkpoint)

    def model(
            self,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype | str] = None
    ) -> EfficientViTForSemanticSegmentation:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = EfficientViTConfig.from_pretrained(self.checkpoint, revision=self.revision)
        model = EfficientViTForSemanticSegmentation.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            config=config,
            revision=self.revision
        )
        model = model.to(device)
        model = model.eval()

        if settings.DETECTOR_STATIC_CACHE:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling detection model {self.checkpoint} on device {device} with dtype {dtype}")
            model = torch.compile(model)

        print(f"Loaded detection model {self.checkpoint} on device {device} with dtype {dtype}")
        return model

    def processor(self) -> SegformerImageProcessor:
        return SegformerImageProcessor.from_pretrained(self.checkpoint, revision=self.revision)