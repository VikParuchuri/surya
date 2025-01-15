from typing import Optional, Dict, List

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class LayoutBox(PolygonBox):
    label: str
    position: int
    top_k: Optional[Dict[str, float]] = None


class LayoutResult(BaseModel):
    bboxes: List[LayoutBox]
    image_bbox: List[float]
    sliced: bool = False  # Whether the image was sliced and reconstructed
