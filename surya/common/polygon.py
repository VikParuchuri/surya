import copy
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, field_validator, computed_field
import numbers


class PolygonBox(BaseModel):
    polygon: List[List[float]]
    confidence: Optional[float] = None

    @field_validator('polygon', mode='before')
    @classmethod
    def convert_bbox_to_polygon(cls, value):
        if isinstance(value, (list, tuple)) and len(value) == 4:
            if all(isinstance(x, numbers.Number) for x in value):
                value = [float(v) for v in value]
                x_min, y_min, x_max, y_max = value
                polygon = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ]
                return polygon
            elif all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value):
                value = [[float(v) for v in point] for point in value]
                return value
        elif isinstance(value, np.ndarray):
            if value.shape == (4, 2):
                return value.tolist()

        raise ValueError(
            f"Input must be either a bbox [x_min, y_min, x_max, y_max] or a polygon with 4 corners [(x,y), (x,y), (x,y), (x,y)].  All values must be numeric. You passed {value} of type {type(value)}.  The first value is of type {type(value[0])}.")

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self):
        return self.width * self.height

    @computed_field
    @property
    def bbox(self) -> List[float]:
        x_coords = [point[0] for point in self.polygon]
        y_coords = [point[1] for point in self.polygon]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def rescale(self, processor_size, image_size):
        # Point is in x, y format
        page_width, page_height = processor_size

        img_width, img_height = image_size
        width_scaler = img_width / page_width
        height_scaler = img_height / page_height

        new_corners = copy.deepcopy(self.polygon)
        for corner in new_corners:
            corner[0] = int(corner[0] * width_scaler)
            corner[1] = int(corner[1] * height_scaler)
        self.polygon = new_corners

    def round(self, divisor):
        for corner in self.polygon:
            corner[0] = int(corner[0] / divisor) * divisor
            corner[1] = int(corner[1] / divisor) * divisor

    def fit_to_bounds(self, bounds):
        new_corners = copy.deepcopy(self.polygon)
        for corner in new_corners:
            corner[0] = max(min(corner[0], bounds[2]), bounds[0])
            corner[1] = max(min(corner[1], bounds[3]), bounds[1])
        self.polygon = new_corners

    def merge(self, other):
        x1 = min(self.bbox[0], other.bbox[0])
        y1 = min(self.bbox[1], other.bbox[1])
        x2 = max(self.bbox[2], other.bbox[2])
        y2 = max(self.bbox[3], other.bbox[3])
        self.polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def intersection_polygon(self, other) -> List[List[float]]:
        new_poly = []
        for i in range(4):
            if i == 0:
                new_corner = [max(self.polygon[0][0], other.polygon[0][0]), max(self.polygon[0][1], other.polygon[0][1])]
            elif i == 1:
                new_corner = [min(self.polygon[1][0], other.polygon[1][0]), max(self.polygon[1][1], other.polygon[1][1])]
            elif i == 2:
                new_corner = [min(self.polygon[2][0], other.polygon[2][0]), min(self.polygon[2][1], other.polygon[2][1])]
            elif i == 3:
                new_corner = [max(self.polygon[3][0], other.polygon[3][0]), min(self.polygon[3][1], other.polygon[3][1])]
            new_poly.append(new_corner)

        return new_poly

    def intersection_area(self, other, x_margin=0, y_margin=0):
        x_overlap = self.x_overlap(other, x_margin)
        y_overlap = self.y_overlap(other, y_margin)
        return x_overlap * y_overlap

    def x_overlap(self, other, x_margin=0):
        return max(0, min(self.bbox[2] + x_margin, other.bbox[2] + x_margin) - max(self.bbox[0] - x_margin, other.bbox[0] - x_margin))

    def y_overlap(self, other, y_margin=0):
        return max(0, min(self.bbox[3] + y_margin, other.bbox[3] + y_margin) - max(self.bbox[1] - y_margin, other.bbox[1] - y_margin))

    def intersection_pct(self, other, x_margin=0, y_margin=0):
        assert 0 <= x_margin <= 1
        assert 0 <= y_margin <= 1
        if self.area == 0:
            return 0

        if x_margin:
            x_margin = int(min(self.width, other.width) * x_margin)
        if y_margin:
            y_margin = int(min(self.height, other.height) * y_margin)

        intersection = self.intersection_area(other, x_margin, y_margin)
        return intersection / self.area

    def shift(self, x_shift: float | None = None, y_shift: float | None = None):
        if x_shift is not None:
            for corner in self.polygon:
                corner[0] += x_shift
        if y_shift is not None:
            for corner in self.polygon:
                corner[1] += y_shift
