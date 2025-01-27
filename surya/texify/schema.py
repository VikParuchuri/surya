from typing import Optional

from pydantic import BaseModel


class TexifyResult(BaseModel):
    text: str
    confidence: Optional[float] = None
