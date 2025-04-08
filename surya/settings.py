import os
from typing import Callable, Dict, Optional

import torch
from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None
    IMAGE_DPI: int = 96  # Used for detection, layout, reading order
    IMAGE_DPI_HIGHRES: int = 192  # Used for OCR, table rec
    IN_STREAMLIT: bool = False  # Whether we're running in streamlit
    ENABLE_EFFICIENT_ATTENTION: bool = (
        True  # Usually keep True, but if you get CUDA errors, setting to False can help
    )
    ENABLE_CUDNN_ATTENTION: bool = False  # Causes issues on many systems when set to True, but can improve performance on certain GPUs
    FLATTEN_PDF: bool = True  # Flatten PDFs by merging form fields before processing
    DISABLE_TQDM: bool = False  # Disable tqdm progress bars
    S3_BASE_URL: str = "https://models.datalab.to"
    PARALLEL_DOWNLOAD_WORKERS: int = (
        10  # Number of workers for parallel model downloads
    )

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

        try:
            import torch_xla

            if len(torch_xla.devices()) > 0:
                return "xla"
        except Exception:
            pass

        return "cpu"

    # Text detection
    DETECTOR_BATCH_SIZE: Optional[int] = None  # Defaults to 2 for CPU/MPS, 32 otherwise
    DETECTOR_MODEL_CHECKPOINT: str = "s3://text_detection/2025_02_28"
    DETECTOR_BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = (
        1400  # Height at which to slice images vertically
    )
    DETECTOR_TEXT_THRESHOLD: float = (
        0.6  # Threshold for text detection (above this is considered text)
    )
    DETECTOR_BLANK_THRESHOLD: float = (
        0.35  # Threshold for blank space (below this is considered blank)
    )
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(
        8, os.cpu_count()
    )  # Number of workers for postprocessing
    DETECTOR_MIN_PARALLEL_THRESH: int = (
        3  # Minimum number of images before we parallelize
    )
    DETECTOR_BOX_Y_EXPAND_MARGIN: float = (
        0.025  # Margin by which to expand detected boxes vertically
    )
    COMPILE_DETECTOR: bool = False

    # Inline math detection
    INLINE_MATH_MODEL_CHECKPOINT: str = "s3://inline_math_detection/2025_02_24"
    INLINE_MATH_THRESHOLD: float = 0.9  # Threshold for inline math detection (above this is considered inline-math)
    INLINE_MATH_BLANK_THRESHOLD: float = (
        0.35  # Threshold for blank space (below this is considered blank)
    )
    INLINE_MATH_BENCH_DATASET_NAME: str = "datalab-to/inline_detection_bench"
    INLINE_MATH_TEXT_BLANK_PX: int = (
        2  # How many pixels to blank out at the botton of each text line
    )
    INLINE_MATH_MIN_AREA: int = 100  # Minimum area for inline math detection

    # Text recognition
    RECOGNITION_MODEL_CHECKPOINT: str = "datalab-to/surya-alpha"
    RECOGNITION_MODEL_QUANTIZE: bool = False
    RECOGNITION_MAX_TOKENS: Optional[int] = None
    RECOGNITION_BATCH_SIZE: Optional[int] = (
        None  # Defaults to 8 for CPU/MPS, 256 otherwise
    )
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),
    }
    RECOGNITION_FONT_DL_BASE: str = (
        "https://github.com/satbyy/go-noto-universal/releases/download/v7.0"
    )
    RECOGNITION_BENCH_DATASET_NAME: str = "vikp/rec_bench"
    RECOGNITION_PAD_VALUE: int = 255  # Should be 0 or 255
    COMPILE_RECOGNITION: bool = False  # Static cache for torch compile

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = "s3://layout/2025_02_18"
    LAYOUT_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    LAYOUT_SLICE_MIN: Dict = {
        "height": 1500,
        "width": 1500,
    }  # When to start slicing images
    LAYOUT_SLICE_SIZE: Dict = {"height": 1200, "width": 1200}  # Size of slices
    LAYOUT_BATCH_SIZE: Optional[int] = None
    LAYOUT_BENCH_DATASET_NAME: str = "vikp/publaynet_bench"
    LAYOUT_MAX_BOXES: int = 100
    COMPILE_LAYOUT: bool = False
    ORDER_BENCH_DATASET_NAME: str = "vikp/order_bench"

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "s3://table_recognition/2025_02_18"
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 768, "width": 768}
    TABLE_REC_MAX_BOXES: int = 150
    TABLE_REC_BATCH_SIZE: Optional[int] = None
    TABLE_REC_BENCH_DATASET_NAME: str = "datalab-to/fintabnet_bench"
    COMPILE_TABLE_REC: bool = False

    # Texify
    TEXIFY_BENCHMARK_DATASET: str = "datalab-to/texify_bench"

    # OCR Error Detection
    OCR_ERROR_MODEL_CHECKPOINT: str = "s3://ocr_error_detection/2025_02_18"
    OCR_ERROR_BATCH_SIZE: Optional[int] = None
    COMPILE_OCR_ERROR: bool = False

    # Tesseract (for benchmarks only)
    TESSDATA_PREFIX: Optional[str] = None

    COMPILE_ALL: bool = False

    @computed_field
    def DETECTOR_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_DETECTOR
            or self.TORCH_DEVICE_MODEL == "xla"
        )  # We need to static cache and pad to batch size for XLA, since it will recompile otherwise

    @computed_field
    def RECOGNITION_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_RECOGNITION
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def LAYOUT_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL or self.COMPILE_LAYOUT or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_TABLE_REC
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def OCR_ERROR_STATIC_CACHE(self) -> bool:
        return (
            self.COMPILE_ALL
            or self.COMPILE_OCR_ERROR
            or self.TORCH_DEVICE_MODEL == "xla"
        )

    @computed_field
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.bfloat16
        return torch.float16

    @computed_field
    def MODEL_DTYPE_BFLOAT(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        if self.TORCH_DEVICE_MODEL == "mps":
            return torch.float16
        return torch.bfloat16

    @computed_field
    def INFERENCE_MODE(self) -> Callable:
        if self.TORCH_DEVICE_MODEL == "xla":
            return torch.no_grad
        return torch.inference_mode

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
