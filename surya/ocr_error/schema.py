from typing import List

from pydantic import BaseModel


class OCRErrorDetectionResult(BaseModel):
    texts: List[str]
    labels: List[str]
