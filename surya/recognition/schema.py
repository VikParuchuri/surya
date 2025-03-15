from typing import Optional, List, Literal

from pydantic import BaseModel

from surya.common.polygon import PolygonBox

class TaskNames:
    block_without_boxes = "block_without_boxes"
    ocr_with_boxes = "ocr_with_boxes"
    ocr_without_boxes = "ocr_without_boxes"

class TextChar(PolygonBox):
    text: str
    bbox_valid: bool = True # This is false when the given bbox is not valid
    confidence: Optional[float] = None

class TextSpan(PolygonBox):
    text: str
    confidence: Optional[float] = None
    formats: Literal["bold", "italic", "underline", "strikethrough", "code", "superscript", "subscript", "highlight", "math"] | None = None

class TextLine(PolygonBox):
    text: str
    chars: List[TextChar] # Individual characters in the line
    confidence: Optional[float] = None
    original_text_good: bool = False


class OCRResult(BaseModel):
    text_lines: List[TextLine]
    image_bbox: List[float]
