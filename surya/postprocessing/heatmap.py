from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import math
from PIL import ImageDraw, ImageFont

from surya.postprocessing.fonts import get_font_path
from surya.postprocessing.util import rescale_bbox
from surya.schema import PolygonBox
from surya.settings import settings
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


def clean_contained_boxes(boxes: List[PolygonBox]) -> List[PolygonBox]:
    new_boxes = []
    for box_obj in boxes:
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


def get_dynamic_thresholds(linemap, text_threshold, low_text, typical_top10_avg=0.7):
    # Find average intensity of top 10% pixels
    flat_map = linemap.flatten()
    top_10_count = int(flat_map.numel() * 0.9)
    avg_intensity = torch.mean(torch.topk(flat_map, k=flat_map.numel() - top_10_count).values)
    scaling_factor = torch.clamp(avg_intensity / typical_top10_avg, 0, 1) ** (1 / 2)

    low_text = torch.clamp(low_text * scaling_factor, 0.1, 0.6)
    text_threshold = torch.clamp(text_threshold * scaling_factor, 0.15, 0.8)

    return text_threshold, low_text


def fast_contours_cumsum(segmap):
    # Nonzero is slow, so use this, then mod and div to get x, y
    # x and y are flipped in the output because openCV uses (y, x) instead of (x, y)
    flat_indices = np.flatnonzero(segmap)
    return np.column_stack((flat_indices % segmap.shape[1], flat_indices // segmap.shape[1]))


def poly_from_region(dilation_map, bbox):
    height, width = dilation_map.shape
    size = height * width
    try:
        niter = int(np.sqrt(size * min(width, height) / size) * 2)
    except ValueError:
        # Overflow when size is too large
        niter = 0

    sx, sy = max(0, bbox[0] - niter), max(0, bbox[1] - niter)
    ex, ey = min(width, bbox[2] + niter + 1), min(height, bbox[3] + niter + 1)

    poly_width = ex - sx
    poly_height = ey - sy
    if poly_height == 0 or poly_width == 0:
        return None

    ksize = 1 + niter
    dilated = F.max_pool2d(dilation_map[sy:ey, sx:ex].unsqueeze(0), ksize, stride=1,
                           padding=ksize // 2).squeeze(0)
    dilation_map[sy:ey, sx:ex] = dilated[:poly_height, :poly_width]

    # Flip to get to x,y order like bboxes
    selected_indices = torch.nonzero(dilation_map).flip(1).cpu().numpy()

    rectangle = cv2.minAreaRect(selected_indices)
    box = cv2.boxPoints(rectangle)
    w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    box_ratio = max(w, h) / (min(w, h) + 1e-5)
    if abs(1 - box_ratio) <= 0.1:
        l, r = int(selected_indices[:, 0].min()), int(selected_indices[:, 0].max())
        t, b = int(selected_indices[:, 1].min()), int(selected_indices[:, 1].max())
        box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

    # make clock-wise order
    startidx = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - startidx, 0)
    box = np.array(box)
    return box


def connected_component_polygons(img: torch.Tensor, low_text, text_threshold, iterations=1000):
    # Initial version from CRAFT - https://github.com/clovaai/CRAFT-pytorch
    # Modified to improve performance and run on GPU

    batch, label_count, img_height, img_width = img.shape
    mask = (img > low_text)

    out = torch.arange(img_height * img_width, device=img.device, dtype=img.dtype).view(1, 1, img_height, img_width).expand(batch, label_count, img_height, img_width)
    out[~mask] = 0

    for _ in range(iterations):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask) # Set everything outside predicted areas to 0

    all_polys = []
    max_confidence = 0

    for batch_item in range(batch):
        batch_polys = []
        for inf_type in range(label_count):
            seg_map = out[batch_item, inf_type]
            linemap = img[batch_item, inf_type]
            dilation_map = torch.zeros_like(linemap, dtype=torch.uint8)
            unique_labels = torch.unique(seg_map)
            polys = []
            for i, label in enumerate(unique_labels.tolist()):
                if label == 0:
                    continue
                label_mask = seg_map == label
                rows, cols = torch.where(label_mask)
                bbox = [
                    cols.min().item(),
                    rows.min().item(),
                    cols.max().item(),
                    rows.max().item()
                ]

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                size = width * height
                if size < 10:
                    continue

                selected_linemap = linemap[label_mask]
                if torch.max(selected_linemap) < text_threshold:
                    continue

                dilation_map.fill_(0)
                dilation_map[label_mask] = 1

                #poly = poly_from_region(dilation_map, bbox)
                poly = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                if poly is None:
                    continue
                confidence = torch.mean(selected_linemap[selected_linemap > low_text]).item()
                max_confidence = max(max_confidence, confidence)
                polys.append(PolygonBox(polygon=poly, confidence=confidence))
            batch_polys.append(polys)
        all_polys.append(batch_polys)

    # Normalize confidences
    if max_confidence > 0:
        for b in all_polys:
            for polys in b:
                for p in polys:
                    p.confidence /= max_confidence
    return all_polys


def get_detected_boxes(textmap, text_threshold=None,  low_text=None) -> List[PolygonBox]:
    if text_threshold is None:
        text_threshold = settings.DETECTOR_TEXT_THRESHOLD

    if low_text is None:
        low_text = settings.DETECTOR_BLANK_THRESHOLD

    text_threshold, low_text = get_dynamic_thresholds(textmap, text_threshold, low_text)

    no_batch = False
    if len(textmap.shape) == 2:
        no_batch = True
        textmap = textmap.unsqueeze(0).unsqueeze(0)

    polygons = connected_component_polygons(textmap, low_text, text_threshold)
    if no_batch:
        return polygons[0][0]
    else:
        return polygons


def get_and_clean_boxes(textmap, processor_size, image_size, text_threshold=None, low_text=None) -> List[PolygonBox]:
    bboxes = get_detected_boxes(textmap, text_threshold, low_text)
    for bbox in bboxes:
        bbox.rescale(processor_size, image_size)
        bbox.fit_to_bounds([0, 0, image_size[0], image_size[1]])

    bboxes = clean_contained_boxes(bboxes)
    return bboxes


def draw_bboxes_on_image(bboxes, image, labels=None):
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=1)

    return image


def draw_polys_on_image(corners, image, labels=None, box_padding=-1, label_offset=1, label_font_size=10):
    draw = ImageDraw.Draw(image)
    font_path = get_font_path()
    label_font = ImageFont.truetype(font_path, label_font_size)

    for i in range(len(corners)):
        poly = corners[i]
        poly = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(poly, outline='red', width=1)

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
                fill="red",
                font=label_font
            )

    return image


