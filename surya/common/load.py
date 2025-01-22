from typing import Optional, Any

import torch

from surya.settings import settings


class ModelLoader:
    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint

    def model(
            self,
            device: torch.device | str | None = settings.TORCH_DEVICE_MODEL,
            dtype: Optional[torch.dtype | str] = settings.MODEL_DTYPE) -> Any:
        raise NotImplementedError()

    def processor(
            self
    ) -> Any:
        raise NotImplementedError()

    @staticmethod
    def split_checkpoint_revision(checkpoint: str) -> tuple[str, str | None]:
        parts = checkpoint.rsplit("@", 1)
        if len(parts) == 1:
            return parts[0], "main" # Default revision is main
        return parts[0], parts[1]