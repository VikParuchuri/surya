from typing import List, Tuple

import numpy as np
import cv2
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
    flat_map = linemap.ravel()
    top_10_count = int(len(flat_map) * 0.9)
    avg_intensity = np.mean(np.partition(flat_map, top_10_count)[top_10_count:])
    scaling_factor = np.clip(avg_intensity / typical_top10_avg, 0, 1) ** (1 / 2)

    low_text = np.clip(low_text * scaling_factor, 0.1, 0.6)
    text_threshold = np.clip(text_threshold * scaling_factor, 0.15, 0.8)

    return text_threshold, low_text


def fast_contours_cumsum(segmap):
    # Nonzero is slow, so use this, then mod and div to get x, y
    # x and y are flipped in the output because openCV uses (y, x) instead of (x, y)
    flat_indices = np.flatnonzero(segmap)
    return np.column_stack((flat_indices % segmap.shape[1], flat_indices // segmap.shape[1]))


def detect_boxes(linemap, text_threshold, low_text):
    # From CRAFT - https://github.com/clovaai/CRAFT-pytorch
    # Modified to return boxes and for speed, accuracy
    img_h, img_w = linemap.shape

    text_threshold, low_text = get_dynamic_thresholds(linemap, text_threshold, low_text)

    text_score_comb = (linemap > low_text).astype(np.uint8)
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb, connectivity=4)

    det = []
    confidences = []
    max_confidence = 0
    segmap = np.zeros_like(labels, dtype=np.uint8)

    for k in range(1, label_count):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        mask = labels == k
        selected_linemap = linemap[mask]

        # thresholding
        if np.max(selected_linemap) < text_threshold:
            continue

        # make segmentation map
        x, y, w, h = stats[k, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]

        try:
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
        except ValueError:
            # Overflow when size is too large
            niter = 0

        sx, sy = max(0, x - niter), max(0, y - niter)
        ex, ey = min(img_w, x + w + niter + 1), min(img_h, y + h + niter + 1)

        segmap.fill(0)
        segmap[mask] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = fast_contours_cumsum(segmap)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        confidence = np.mean(selected_linemap[selected_linemap > low_text])
        max_confidence = max(max_confidence, confidence)

        confidences.append(confidence)
        det.append(box)

    if max_confidence > 0:
        confidences = [c / max_confidence for c in confidences]
    return det, labels, confidences


def get_detected_boxes(textmap, text_threshold=None,  low_text=None) -> List[PolygonBox]:
    if text_threshold is None:
        text_threshold = settings.DETECTOR_TEXT_THRESHOLD

    if low_text is None:
        low_text = settings.DETECTOR_BLANK_THRESHOLD

    textmap = textmap.copy()
    textmap = textmap.astype(np.float32)
    boxes, labels, confidences = detect_boxes(textmap, text_threshold, low_text)
    # From point form to box form
    boxes = [PolygonBox(polygon=box, confidence=confidence) for box, confidence in zip(boxes, confidences)]
    return boxes


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


