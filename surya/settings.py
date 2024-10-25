from typing import Dict, Optional

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96 # Used for detection, layout, reading order
    IMAGE_DPI_HIGHRES: int = 192  # Used for OCR, table rec
    IN_STREAMLIT: bool = False # Whether we're running in streamlit
    ENABLE_EFFICIENT_ATTENTION: bool = True # Usually keep True, but if you get CUDA errors, setting to False can help
    ENABLE_CUDNN_ATTENTION: bool = False # Causes issues on many systems when set to True, but can improve performance on certain GPUs
    FLATTEN_PDF: bool = True # Flatten PDFs by merging form fields before processing

    # Paths
    DATA_DIR: str = "data"
    RESULT_DIR: str = "results"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    # Text detection
    DETECTOR_BATCH_SIZE: Optional[int] = None # Defaults to 2 for CPU/MPS, 32 otherwise
    DETECTOR_MODEL_CHECKPOINT: str = "vikp/surya_det3"
    DETECTOR_BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1400 # Height at which to slice images vertically
    DETECTOR_TEXT_THRESHOLD: float = 0.6 # Threshold for text detection (above this is considered text)
    DETECTOR_BLANK_THRESHOLD: float = 0.35 # Threshold for blank space (below this is considered blank)
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(8, os.cpu_count()) # Number of workers for postprocessing
    DETECTOR_MIN_PARALLEL_THRESH: int = 3 # Minimum number of images before we parallelize

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = "vikp/surya_rec2"
    RECOGNITION_MAX_TOKENS: int = 175
    RECOGNITION_BATCH_SIZE: Optional[int] = None # Defaults to 8 for CPU/MPS, 256 otherwise
    RECOGNITION_IMAGE_SIZE: Dict = {"height": 256, "width": 896}
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }
    RECOGNITION_FONT_DL_BASE: str = "https://github.com/satbyy/go-noto-universal/releases/download/v7.0"
    RECOGNITION_BENCH_DATASET_NAME: str = "vikp/rec_bench"
    RECOGNITION_PAD_VALUE: int = 255 # Should be 0 or 255
    RECOGNITION_STATIC_CACHE: bool = False # Static cache for torch compile
    RECOGNITION_ENCODER_BATCH_DIVISOR: int = 2 # Divisor for batch size in decoder

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = "vikp/surya_layout3"
    LAYOUT_BENCH_DATASET_NAME: str = "vikp/publaynet_bench"

    # Ordering
    ORDER_MODEL_CHECKPOINT: str = "vikp/surya_order"
    ORDER_IMAGE_SIZE: Dict = {"height": 1024, "width": 1024}
    ORDER_MAX_BOXES: int = 256
    ORDER_BATCH_SIZE: Optional[int] = None  # Defaults to 4 for CPU/MPS, 32 otherwise
    ORDER_BENCH_DATASET_NAME: str = "vikp/order_bench"

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "vikp/surya_tablerec"
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 640, "width": 640}
    TABLE_REC_MAX_BOXES: int = 512
    TABLE_REC_MAX_ROWS: int = 384
    TABLE_REC_BATCH_SIZE: Optional[int] = None
    TABLE_REC_BENCH_DATASET_NAME: str = "vikp/fintabnet_bench"

    # Tesseract (for benchmarks only)
    TESSDATA_PREFIX: Optional[str] = None

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()