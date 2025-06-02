import math
import numpy as np
from typing import Optional, List

from pydantic import BaseModel, field_validator

from surya.common.polygon import PolygonBox


class BaseChar(PolygonBox):
    text: str
    confidence: Optional[float] = 0

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v is None:
            return 0
        elif math.isnan(v) or np.isnan(v):
            return 0
        return v


class TextChar(BaseChar):
    bbox_valid: bool = True  # This is false when the given bbox is not valid


class TextWord(BaseChar):
    bbox_valid: bool = True


class TextLine(BaseChar):
    chars: List[TextChar]  # Individual characters in the line
    original_text_good: bool = False
    words: List[TextWord] | None = None


class OCRResult(BaseModel):
    text_lines: List[TextLine]
    image_bbox: List[float]
