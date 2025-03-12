from typing import Dict

import torch

from surya.common.predictor import BasePredictor
from surya.detection import DetectionPredictor, InlineDetectionPredictor
from surya.layout import LayoutPredictor
from surya.ocr_error import OCRErrorPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor


def load_predictors(
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None
) -> Dict[str, BasePredictor]:
    return {
        "layout": LayoutPredictor(device=device, dtype=dtype),
        "ocr_error": OCRErrorPredictor(device=device, dtype=dtype),
        "recognition": RecognitionPredictor(device=device, dtype=dtype),
        "detection": DetectionPredictor(device=device, dtype=dtype),
        "inline_detection": InlineDetectionPredictor(device=device, dtype=dtype),
        "table_rec": TableRecPredictor(device=device, dtype=dtype)
    }