from typing import Optional
import torch

from surya.settings import settings


class BasePredictor:
    def __init__(self, checkpoint: Optional[str] = None, device: torch.device | str | None = settings.TORCH_DEVICE_MODEL, dtype: Optional[torch.dtype | str] = settings.MODEL_DTYPE):
        self.model = None
        self.processor = None
        self.load_model(checkpoint, device, dtype)
        self.load_processor(checkpoint)

    def load_model(self, checkpoint: Optional[str] = None, device: torch.device | str | None = settings.TORCH_DEVICE_MODEL, dtype: Optional[torch.dtype | str] = settings.MODEL_DTYPE):
        raise NotImplementedError()

    def load_processor(self, checkpoint: Optional[str] = None):
        raise NotImplementedError()

    def to(self, device_dtype: torch.device | str | None = None):
        if self.model:
            self.model.to(device_dtype)
        else:
            raise ValueError("Model not loaded")

    def get_batch_size(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()