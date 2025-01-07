from typing import Optional
import torch

from surya.common.load import ModelLoader
from surya.settings import settings


class BasePredictor:
    model_loader_cls = ModelLoader
    def __init__(self, checkpoint: Optional[str] = None, device: torch.device | str | None = settings.TORCH_DEVICE_MODEL, dtype: Optional[torch.dtype | str] = settings.MODEL_DTYPE):
        self.model = None
        self.processor = None
        loader = self.model_loader_cls(checkpoint)

        self.model = loader.model(device, dtype)
        self.processor = loader.processor()

    def to(self, device_dtype: torch.device | str | None = None):
        if self.model:
            self.model.to(device_dtype)
        else:
            raise ValueError("Model not loaded")

    @staticmethod
    def get_batch_size():
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()