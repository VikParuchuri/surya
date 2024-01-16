import copy
from typing import List, Tuple

from pydantic import BaseModel, field_validator, computed_field


class PolygonBox(BaseModel):
    corners: List[List[float]]

    @field_validator('corners')
    @classmethod
    def check_elements(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4:
            raise ValueError('corner must have 4 elements')

        for corner in v:
            if len(corner) != 2:
                raise ValueError('corner must have 2 elements')
        return v

    @property
    def height(self):
        return self.corners[1][1] - self.corners[0][1]

    @property
    def width(self):
        return self.corners[1][0] - self.corners[0][0]

    @property
    def area(self):
        return self.width * self.height

    @computed_field
    @property
    def bbox(self) -> List[float]:
        box = [self.corners[0][0], self.corners[0][1], self.corners[1][0], self.corners[2][1]]
        if box[0] > box[2]:
            box[0], box[2] = box[2], box[0]
        if box[1] > box[3]:
            box[1], box[3] = box[3], box[1]
        return box


    def rescale(self, processor_size, image_size):
        # Point is in x, y format
        page_width, page_height = processor_size

        img_width, img_height = image_size
        width_scaler = img_width / page_width
        height_scaler = img_height / page_height

        new_corners = copy.deepcopy(self.corners)
        for corner in new_corners:
            corner[0] = int(corner[0] * width_scaler)
            corner[1] = int(corner[1] * height_scaler)
        self.corners = new_corners



class Bbox(BaseModel):
    bbox: List[float]

    @field_validator('bbox')
    @classmethod
    def check_4_elements(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError('bbox must have 4 elements')
        return v

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self):
        return self.width * self.height