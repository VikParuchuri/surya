import math
from typing import List, Dict
import numpy as np

from surya.table_rec.model.config import BOX_PROPERTIES, SPECIAL_TOKENS, BOX_DIM


class LabelShaper:
    def __init__(self):
        self.property_keys = [k for (k, kcount, mode) in BOX_PROPERTIES]

    def dict_to_labels(self, label_components: List[dict]):
        if len(label_components) == 0:
            return []

        out_list = []
        for (k, kcount, mode) in BOX_PROPERTIES:
            for label_component in label_components:
                if k not in label_component:
                    raise ValueError(f"Missing key {k} in label component {label_component}")

                if mode == "classification":
                    assert isinstance(label_component[k], int)
                elif mode == "regression":
                    assert (isinstance(label_component[k], (int, float)) and kcount == 1) or len(label_component[k]) == kcount
                else:
                    raise ValueError(f"Invalid mode {k['mode']} for key {k}")

        for label_component in label_components:
            bbox = label_component["bbox"]
            for i in range(len(bbox)):
                if bbox[i] < 0:
                    bbox[i] = 0
                if bbox[i] > BOX_DIM:
                    bbox[i] = BOX_DIM

            vector = []
            for (k, kcount, mode) in BOX_PROPERTIES:
                item = label_component[k]
                if isinstance(item, (list, tuple)):
                    vector += list(item)
                elif isinstance(item, (float, int)):
                    if mode == "classification":
                        # Shift up for model
                        item += SPECIAL_TOKENS
                    vector.append(item)
                else:
                    raise ValueError(f"Invalid item {item} for key {k}")

            out_list.append(vector)

        return out_list

    def component_idx(self, key):
        idx = 0
        for (k, kcount, mode) in BOX_PROPERTIES:
            if mode == "regression":
                incr = kcount
            elif mode == "classification":
                incr = 1
            else:
                raise ValueError(f"Invalid mode {mode} for key {k}")
            if k == key:
                return (idx, idx + incr)
            idx += incr
        raise ValueError(f"Key {key} not found in properties")

    def get_box_property(self, key, add_special_tokens=True):
        for (k, kcount, mode) in BOX_PROPERTIES:
            if k == key:
                # Add special token count
                if mode == "classification" and add_special_tokens:
                    kcount += SPECIAL_TOKENS
                return (k, kcount, mode)
        raise ValueError(f"Key {key} not found in properties")

    def component_idx_dict(self):
        idx_dict = {}
        for (k, kcount, mode) in BOX_PROPERTIES:
            idx_dict[k] = self.component_idx(k)
        return idx_dict

    def convert_polygons_to_bboxes(self, label_components: List[Dict]):
        for i, label_component in enumerate(label_components):
            poly = label_component["polygon"]
            poly = np.clip(poly, 0, BOX_DIM)

            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = poly
            cx = (x1 + x2 + x3 + x4) / 4
            cy = (y1 + y2 + y3 + y4) / 4
            width = (x2 + x3) / 2 - (x1 + x4) / 2
            height = (y3 + y4) / 2 - (y2 + y1) / 2
            bottom_avg_x = (x3 + x4) / 2
            top_avg_x = (x1 + x2) / 2
            right_avg_y = (y2 + y3) / 2
            left_avg_y = (y1 + y4) / 2

            x_skew = bottom_avg_x - top_avg_x
            y_skew = right_avg_y - left_avg_y
            x_skew += BOX_DIM // 2 # Shift up into positive space
            y_skew += BOX_DIM // 2 # Shift up into positive space
            new_poly = [
                cx,
                cy,
                width,
                height,
                x_skew,
                y_skew
            ]
            label_component["bbox"] = new_poly

        return label_components

    def convert_bbox_to_polygon(self, box, skew_scaler=BOX_DIM // 2, skew_min=.001):
        cx = box[0]
        cy = box[1]
        width = box[2]
        height = box[3]
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        skew_x = math.floor((box[4] - skew_scaler) / 2)
        skew_y = math.floor((box[5] - skew_scaler) / 2)

        # Ensures we don't get slightly warped boxes
        # Note that the values are later scaled, so this is in 1/1024 space
        if abs(skew_x) < skew_min:
            skew_x = 0

        if abs(skew_y) < skew_min:
            skew_y = 0

        polygon = [x1 - skew_x, y1 - skew_y, x2 - skew_x, y1 + skew_y, x2 + skew_x, y2 + skew_y, x1 + skew_x,
                   y2 - skew_y]
        poly = []
        for i in range(4):
            poly.append([
                polygon[2 * i],
                polygon[2 * i + 1]
            ])
        return poly



