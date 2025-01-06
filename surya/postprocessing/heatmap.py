from typing import List
from PIL import ImageDraw, ImageFont

from surya.postprocessing.fonts import get_font_path
from surya.schema import PolygonBox
from surya.postprocessing.text import get_text_size


def keep_largest_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
        box = box_obj.bbox
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        contained = False
        for other_box_obj in boxes:
            if other_box_obj.polygon == box_obj.polygon:
                continue

            other_box = other_box_obj.bbox
            other_box_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
            if box == other_box:
                continue
            # find overlap percentage
            overlap = box_obj.intersection_pct(other_box_obj)
            if overlap > .9 and box_area < other_box_area:
                contained = True
                break
        if not contained:
            new_boxes.append(box_obj)
    return new_boxes


def intersects_other_boxes(box: List[List[float]], boxes: List[List[List[float]]], thresh=.9) -> bool:
    box = PolygonBox(polygon=box)
    for other_box in boxes:
        # find overlap percentage
        other_box_obj = PolygonBox(polygon=other_box)
        overlap = box.intersection_pct(other_box_obj)
        if overlap > thresh:
            return True
    return False



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


def draw_bboxes_on_image(bboxes, image, labels=None, label_font_size=10, color: str | list = 'red'):
    polys = []
    for bb in bboxes:
        # Clockwise polygon
        poly = [
            [bb[0], bb[1]],
            [bb[2], bb[1]],
            [bb[2], bb[3]],
            [bb[0], bb[3]]
        ]
        polys.append(poly)

    return draw_polys_on_image(polys, image, labels, label_font_size=label_font_size, color=color)


def draw_polys_on_image(corners, image, labels=None, box_padding=-1, label_offset=1, label_font_size=10, color: str | list = 'red'):
    draw = ImageDraw.Draw(image)
    font_path = get_font_path()
    label_font = ImageFont.truetype(font_path, label_font_size)

    for i in range(len(corners)):
        poly = corners[i]
        poly = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(poly, outline=color[i] if isinstance(color, list) else color, width=1)

        if labels is not None:
            label = labels[i]
            text_position = (
                min([p[0] for p in poly]) + label_offset,
                min([p[1] for p in poly]) + label_offset
            )
            text_size = get_text_size(label, label_font)
            box_position = (
                text_position[0] - box_padding + label_offset,
                text_position[1] - box_padding + label_offset,
                text_position[0] + text_size[0] + box_padding + label_offset,
                text_position[1] + text_size[1] + box_padding + label_offset
            )
            draw.rectangle(box_position, fill="white")
            draw.text(
                text_position,
                label,
                fill=color[i] if isinstance(color, list) else color,
                font=label_font
            )

    return image
