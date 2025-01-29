from typing import List, Optional, Any

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class ColumnLine(PolygonBox):
    vertical: bool
    horizontal: bool

class TextBox(PolygonBox):
    math: bool = False
    def __hash__(self):
        return hash(tuple(self.bbox))

class TextDetectionResult(BaseModel):
    bboxes: List[TextBox]
    vertical_lines: List[ColumnLine]
    heatmap: Optional[Any]
    affinity_map: Optional[Any]
    image_bbox: List[float]
