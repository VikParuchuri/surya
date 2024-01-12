from typing import Dict, Optional

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    # Text detection
    DETECTOR_BATCH_SIZE: int = 2 if TORCH_DEVICE_MODEL == "cpu" else 64
    DETECTOR_MODEL_CHECKPOINT: str = "vikp/line_detector"
    BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1200 # Height at which to slice images vertically
    DETECTOR_TEXT_THRESHOLD: float = 0.6 # Threshold for text detection
    DETECTOR_NMS_THRESHOLD: float = 0.35 # Threshold for non-maximum suppression

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()