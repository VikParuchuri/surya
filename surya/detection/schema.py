from typing import List, Optional, Any

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class ColumnLine(PolygonBox):
    vertical: bool
    horizontal: bool


class TextDetectionResult(BaseModel):
    bboxes: List[PolygonBox]
    vertical_lines: List[ColumnLine]
    heatmap: Optional[Any]
    affinity_map: Optional[Any]
    image_bbox: List[float]
