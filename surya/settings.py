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

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @computed_field
    @property
    def TORCH_DEVICE_DETECTION(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        # Does not work with mps
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    # Text detection
    DETECTOR_BATCH_SIZE: int = 2 if TORCH_DEVICE_DETECTION == "cpu" else 32
    DETECTOR_MODEL_CHECKPOINT: str = "vikp/line_detector"
    BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1200 # Height at which to slice images vertically
    DETECTOR_TEXT_THRESHOLD: float = 0.6 # Threshold for text detection
    DETECTOR_NMS_THRESHOLD: float = 0.35 # Threshold for non-maximum suppression

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = "vikp/rec_test_gqa"
    RECOGNITION_MAX_TOKENS: int = 512
    RECOGNITION_BATCH_SIZE: int = 8 if TORCH_DEVICE_MODEL in ["cpu", "mps"] else 128
    RECOGNITION_IMAGE_SIZE: Dict = {"height": 196, "width": 896}
    RECOGNITION_RENDER_FONT: str = os.path.join(FONT_DIR, "GoNotoKurrent-Regular.ttf")

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    @computed_field
    @property
    def MODEL_DTYPE_DETECTION(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_DETECTION == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()