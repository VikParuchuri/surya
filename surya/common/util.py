import copy
from typing import List

from surya.common.polygon import PolygonBox


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


def rescale_bbox(bbox, processor_size, image_size):
    page_width, page_height = processor_size

    img_width, img_height = image_size
    width_scaler = img_width / page_width
    height_scaler = img_height / page_height

    new_bbox = copy.deepcopy(bbox)
    new_bbox[0] = int(new_bbox[0] * width_scaler)
    new_bbox[1] = int(new_bbox[1] * height_scaler)
    new_bbox[2] = int(new_bbox[2] * width_scaler)
    new_bbox[3] = int(new_bbox[3] * height_scaler)
    return new_bbox

def expand_bbox(bbox, expansion_factor=.01):
    expansion_low = 1 - expansion_factor
    expansion_high = 1 + expansion_factor
    return [
        bbox[0] * expansion_low,
        bbox[1] * expansion_low,
        bbox[2] * expansion_high,
        bbox[3] * expansion_high
    ]
