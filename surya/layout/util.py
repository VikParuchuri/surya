from typing import List

import torch

from surya.schema import PolygonBox


def prediction_to_polygon(pred, img_size, bbox_scaler, skew_scaler, skew_min=.001):
    w_scale = img_size[0] / bbox_scaler
    h_scale = img_size[1] / bbox_scaler

    boxes = pred
    cx = boxes[0]
    cy = boxes[1]
    width = boxes[2]
    height = boxes[3]
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2
    skew_x = torch.floor((boxes[4] - skew_scaler) / 2)
    skew_y = torch.floor((boxes[5] - skew_scaler) / 2)

    # Ensures we don't get slightly warped boxes
    # Note that the values are later scaled, so this is in 1/1024 space
    skew_x[torch.abs(skew_x) < skew_min] = 0
    skew_y[torch.abs(skew_y) < skew_min] = 0

    polygon = [x1 - skew_x, y1 - skew_y, x2 - skew_x, y1 + skew_y, x2 + skew_x, y2 + skew_y, x1 + skew_x, y2 - skew_y]
    poly = []
    for i in range(4):
        poly.append([
            polygon[2 * i].item() * w_scale,
            polygon[2 * i + 1].item() * h_scale
        ])
    return poly

def clean_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        xs = [point[0] for point in box_obj.polygon]
        ys = [point[1] for point in box_obj.polygon]
        if max(xs) == min(xs) or max(ys) == min(ys):
            continue

        box = box_obj.bbox
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            if box == other_box:
                continue
            if box[0] >= other_box[0] and box[1] >= other_box[1] and box[2] <= other_box[2] and box[3] <= other_box[3]:
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes