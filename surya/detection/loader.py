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
            self.checkpoint = [settings.DETECTOR_MODEL_CHECKPOINT, settings.INLINE_MATH_MODEL_CHECKPOINT]
        
        checkpoints, revisions = [], []
        for checkpoint in self.checkpoint:
            ckpt, rev = self.split_checkpoint_revision(checkpoint)
            checkpoints.append(ckpt)
            revisions.append(rev)

        self.checkpoint = checkpoints
        self.revision = revisions

    def model(
            self,
            device: Optional[torch.device | str] = None,
            dtype: Optional[torch.dtype | str] = None
    ) -> EfficientViTForSemanticSegmentation:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        #Loading Text Detection Model
        config = EfficientViTConfig.from_pretrained(self.checkpoint[0], revision=self.revision[0])
        model = EfficientViTForSemanticSegmentation.from_pretrained(
            self.checkpoint[0],
            torch_dtype=dtype,
            config=config,
            revision=self.revision[0]
        )
        model = model.to(device)
        model = model.eval()

        if settings.DETECTOR_STATIC_CACHE:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling detection model {self.checkpoint[0]} on device {device} with dtype {dtype}")
            model = torch.compile(model)

        print(f"Loaded detection model {self.checkpoint[0]} on device {device} with dtype {dtype}")

        #Loading Inline Math Model - Will be merged into common model
        inline_config = EfficientViTConfig.from_pretrained(self.checkpoint[1], revision=self.revision[1])
        inline_model = EfficientViTForSemanticSegmentation.from_pretrained(
            self.checkpoint[1],
            torch_dtype=dtype,
            config=inline_config,
            revision=self.revision[1]
        )
        inline_model = inline_model.to(device)
        inline_model = inline_model.eval()

        if settings.DETECTOR_STATIC_CACHE:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling detection model {self.checkpoint[1]} on device {device} with dtype {dtype}")
            inline_model = torch.compile(inline_model)

        print(f"Loaded inline math detection model {self.checkpoint[1]} on device {device} with dtype {dtype}")

        return {
            'text': model,
            'inline': inline_model
        }

    def processor(self) -> SegformerImageProcessor:
        return SegformerImageProcessor.from_pretrained(self.checkpoint[0], revision=self.revision[0])