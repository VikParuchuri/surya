import os
from typing import Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    __ENABLED_FLAG: list[str] = ["true", "yes", "y", "on", "1"]

    # General
    TORCH_DEVICE: Optional[str] = os.getenv("SURYA_TORCH_DEVICE", None)
    IMAGE_DPI: int = int(os.getenv("SURYA_IMAGE_DPI", 96))  # Used for detection, layout, reading order
    IMAGE_DPI_HIGHRES: int = int(os.getenv("SURYA_IMAGE_DPI_HIGHRES", 192))  # Used for OCR, table rec
    IN_STREAMLIT: bool = os.getenv("SURYA_IN_STREAMLIT", "False").lower() in __ENABLED_FLAG  # Whether we're running in streamlit
    ENABLE_EFFICIENT_ATTENTION: bool = os.getenv("SURYA_ENABLE_EFFICIENT_ATTENTION", "True").lower() in __ENABLED_FLAG  # Usually keep True, but if you get CUDA errors, setting to False can help
    ENABLE_CUDNN_ATTENTION: bool = os.getenv("SURYA_ENABLE_CUDNN_ATTENTION", "False").lower() in __ENABLED_FLAG  # Causes issues on many systems when set to True, but can improve performance on certain GPUs
    FLATTEN_PDF: bool = os.getenv("SURYA_FLATTEN_PDF", "True").lower() in __ENABLED_FLAG  # Flatten PDFs by merging form fields before processing

    # Paths
    DATA_DIR: str = os.getenv("SURYA_DATA_DIR", "data")
    RESULT_DIR: str = os.environ.get("SURYA_RESULT_DIR", "results")
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
    DETECTOR_BATCH_SIZE: Optional[int] = int(os.getenv("SURYA_DETECTOR_BATCH_SIZE")) if os.getenv("SURYA_DETECTOR_BATCH_SIZE") else None  # Defaults to 2 for CPU/MPS, 32 otherwise
    DETECTOR_MODEL_CHECKPOINT: str = os.getenv("SURYA_DETECTOR_MODEL_CHECKPOINT", "vikp/surya_det3")
    DETECTOR_BENCH_DATASET_NAME: str = os.getenv("SURYA_DETECTOR_BENCH_DATASET_NAME", "vikp/doclaynet_bench")
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = int(os.getenv("SURYA_DETECTOR_IMAGE_CHUNK_HEIGHT", 1400))  # Height at which to slice images vertically
    DETECTOR_TEXT_THRESHOLD: float = float(os.getenv("SURYA_DETECTOR_IMAGE_CHUNK_HEIGHT", 0.6))  # Threshold for text detection (above this is considered text)
    DETECTOR_BLANK_THRESHOLD: float = float(os.getenv("SURYA_DETECTOR_BLANK_THRESHOLD", 0.35))  # Threshold for blank space (below this is considered blank)
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(8, os.cpu_count())  # Number of workers for postprocessing
    DETECTOR_MIN_PARALLEL_THRESH: int = int(os.getenv("SURYA_DETECTOR_MIN_PARALLEL_THRESH", 3))  # Minimum number of images before we parallelize
    COMPILE_DETECTOR: bool = os.getenv("SURYA_COMPILE_DETECTOR", "False").lower() in __ENABLED_FLAG

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = os.getenv("SURYA_RECOGNITION_MODEL_CHECKPOINT", "vikp/surya_rec2")
    RECOGNITION_MAX_TOKENS: int = int(os.getenv("SURYA_RECOGNITION_MAX_TOKENS", 175))
    RECOGNITION_BATCH_SIZE: Optional[int] = int(os.getenv("SURYA_RECOGNITION_BATCH_SIZE")) if os.getenv("SURYA_RECOGNITION_BATCH_SIZE") else None  # Defaults to 8 for CPU/MPS, 256 otherwise
    RECOGNITION_IMAGE_SIZE: Dict = {"height": 256, "width": 896}

    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }
    RECOGNITION_FONT_DL_BASE: str = os.getenv("SURYA_RECOGNITION_FONT_DL_BASE", "https://github.com/satbyy/go-noto-universal/releases/download/v7.0")
    RECOGNITION_BENCH_DATASET_NAME: str = os.getenv("SURYA_RECOGNITION_BENCH_DATASET_NAME", "vikp/rec_bench")
    RECOGNITION_PAD_VALUE: int = int(os.getenv("SURYA_RECOGNITION_PAD_VALUE", 255))  # Should be 0 or 255
    COMPILE_RECOGNITION: bool = os.getenv("SURYA_COMPILE_RECOGNITION", "False").lower() in __ENABLED_FLAG  # Static cache for torch compile
    RECOGNITION_ENCODER_BATCH_DIVISOR: int = int(os.getenv("SURYA_RECOGNITION_ENCODER_BATCH_DIVISOR"), 1)  # Divisor for batch size in decoder

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = os.getenv("SURYA_LAYOUT_MODEL_CHECKPOINT", "datalab-to/surya_layout0")
    LAYOUT_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    LAYOUT_SLICE_MIN: Dict = {"height": 1500, "width": 1500}  # When to start slicing images
    LAYOUT_SLICE_SIZE: Dict = {"height": 1200, "width": 1200}  # Size of slices
    LAYOUT_BATCH_SIZE: Optional[int] = int(os.getenv("SURYA_LAYOUT_BATCH_SIZE")) if os.getenv("SURYA_LAYOUT_BATCH_SIZE") else None
    LAYOUT_BENCH_DATASET_NAME: str = os.getenv("SURYA_LAYOUT_BENCH_DATASET_NAME", "vikp/publaynet_bench")
    LAYOUT_MAX_BOXES: int = int(os.getenv("SURYA_LAYOUT_MAX_BOXES", 100))
    COMPILE_LAYOUT: bool = os.getenv("SURYA_COMPILE_LAYOUT", "False").lower() in __ENABLED_FLAG
    ORDER_BENCH_DATASET_NAME: str = os.getenv("SURYA_ORDER_BENCH_DATASET_NAME", "vikp/order_bench")

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = os.getenv("SURYA_TABLE_REC_MODEL_CHECKPOINT", "vikp/surya_tablerec")
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 640, "width": 640}
    TABLE_REC_MAX_BOXES: int = int(os.getenv("SURYA_TABLE_REC_MAX_BOXES", 512))
    TABLE_REC_MAX_ROWS: int = int(os.getenv("SURYA_TABLE_REC_MAX_ROWS", 384))
    TABLE_REC_BATCH_SIZE: Optional[int] = int(os.getenv("SURYA_TABLE_REC_BATCH_SIZE")) if os.getenv("SURYA_TABLE_REC_BATCH_SIZE") else None
    TABLE_REC_BENCH_DATASET_NAME: str = os.getenv("SURYA_TABLE_REC_BENCH_DATASET_NAME", "vikp/fintabnet_bench")
    COMPILE_TABLE_REC: bool = os.getenv("SURYA_COMPILE_TABLE_REC", "False").lower() in __ENABLED_FLAG

    # OCR Error Detection
    OCR_ERROR_MODEL_CHECKPOINT: str = os.getenv("SURYA_OCR_ERROR_MODEL_CHECKPOINT", "datalab-to/ocr_error_detection")
    OCR_ERROR_BATCH_SIZE: Optional[int] = int(os.getenv("SURYA_OCR_ERROR_BATCH_SIZE")) if os.getenv("SURYA_OCR_ERROR_BATCH_SIZE") else None
    COMPILE_OCR_ERROR: bool = os.getenv("SURYA_COMPILE_OCR_ERROR", "False").lower() in __ENABLED_FLAG

    # Tesseract (for benchmarks only)
    TESSDATA_PREFIX: Optional[str] = os.getenv("SURYA_TESSDATA_PREFIX", None)

    COMPILE_ALL: bool = os.getenv("SURYA_COMPILE_ALL", "False").lower() in __ENABLED_FLAG

    @computed_field
    def DETECTOR_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_DETECTOR

    @computed_field
    def RECOGNITION_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_RECOGNITION

    @computed_field
    def LAYOUT_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_LAYOUT

    @computed_field
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_TABLE_REC

    @computed_field
    def OCR_ERROR_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_OCR_ERROR

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
