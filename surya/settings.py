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
    DETECTOR_MODEL_CHECKPOINT: str = "vikp/surya_det3@467ee9ec33e6e6c5f73e57dbc1415b14032f5b95"
    DETECTOR_BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1400 # Height at which to slice images vertically
    DETECTOR_TEXT_THRESHOLD: float = 0.6 # Threshold for text detection (above this is considered text)
    DETECTOR_BLANK_THRESHOLD: float = 0.35 # Threshold for blank space (below this is considered blank)
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(8, os.cpu_count()) # Number of workers for postprocessing
    DETECTOR_MIN_PARALLEL_THRESH: int = 3 # Minimum number of images before we parallelize
    COMPILE_DETECTOR: bool = False

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = "vikp/surya_rec2@6611509b2c3a32c141703ce19adc899d9d0abf41"
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
    COMPILE_RECOGNITION: bool = False # Static cache for torch compile

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = "datalab-to/surya_layout@7ac8e390226ee5fa2125dd303d827f79d31d1a1f"
    LAYOUT_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    LAYOUT_SLICE_MIN: Dict = {"height": 1500, "width": 1500} # When to start slicing images
    LAYOUT_SLICE_SIZE: Dict = {"height": 1200, "width": 1200} # Size of slices
    LAYOUT_BATCH_SIZE: Optional[int] = None
    LAYOUT_BENCH_DATASET_NAME: str = "vikp/publaynet_bench"
    LAYOUT_MAX_BOXES: int = 100
    COMPILE_LAYOUT: bool = False
    ORDER_BENCH_DATASET_NAME: str = "vikp/order_bench"

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "datalab-to/surya_tablerec@7327dac38c300b2f6cd0501ebc2347dd3ef7fcf2"
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    TABLE_REC_MAX_BOXES: int = 150
    TABLE_REC_BATCH_SIZE: Optional[int] = None
    TABLE_REC_BENCH_DATASET_NAME: str = "datalab-to/fintabnet_bench"
    COMPILE_TABLE_REC: bool = False

    # Texify
    TEXIFY_MODEL_CHECKPOINT: str = "datalab-to/texify@8f1d761762b3e977e9e62cebfca487d489556abc"
    TEXIFY_BENCHMARK_DATASET: str = "datalab-to/texify_bench"
    TEXIFY_IMAGE_SIZE: Dict = {"height": 480, "width": 480}
    TEXIFY_MAX_TOKENS: int = 768
    TEXIFY_BATCH_SIZE: Optional[int] = None
    COMPILE_TEXIFY: bool = False

    # OCR Error Detection
    OCR_ERROR_MODEL_CHECKPOINT: str = "datalab-to/ocr_error_detection@c1cbda3757670fd520553eaa5197656d331de414"
    OCR_ERROR_BATCH_SIZE: Optional[int] = None
    COMPILE_OCR_ERROR: bool = False

    # Tesseract (for benchmarks only)
    TESSDATA_PREFIX: Optional[str] = None
    
    COMPILE_ALL: bool = False

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
    def TEXIFY_STATIC_CACHE(self) -> bool:
        return self.COMPILE_ALL or self.COMPILE_TEXIFY

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
