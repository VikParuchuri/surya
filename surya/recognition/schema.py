from typing import Optional, List

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class TextChar(PolygonBox):
    text: str
    bbox_valid: bool = True  # This is false when the given bbox is not valid
    confidence: Optional[float] = None


class TextWord(PolygonBox):
    text: str
    bbox_valid: bool = True
    confidence: Optional[float] = None


class TextLine(PolygonBox):
    text: str
    chars: List[TextChar]  # Individual characters in the line
    confidence: Optional[float] = None
    original_text_good: bool = False
    words: List[TextWord] | None = None


class OCRResult(BaseModel):
    text_lines: List[TextLine]
    image_bbox: List[float]
