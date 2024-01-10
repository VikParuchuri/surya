from typing import Dict, Optional

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    MODEL_CHECKPOINT: str = "vikp/line_detector4"
    IMAGE_DPI: int = 96

    # Paths
    BASE_DIR: str = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RESULT_DIR: str = os.path.join(BASE_DIR, "results")

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        # MPS returns garbled results for some reason
        # Maybe related to https://github.com/pytorch/pytorch/issues/84936
        #if torch.backends.mps.is_available():
        #    return "mps"

        return "cpu"

    @computed_field
    @property
    def CUDA(self) -> bool:
        return "cuda" in self.TORCH_DEVICE_MODEL

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    BATCH_SIZE: int = 2 if TORCH_DEVICE_MODEL == "cpu" else 16


    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()