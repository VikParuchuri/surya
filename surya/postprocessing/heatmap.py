from typing import List

import numpy as np
import cv2
import math
from PIL import ImageDraw

from surya.postprocessing.util import rescale_bbox
from surya.schema import PolygonBox
from surya.settings import settings


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


def get_dynamic_thresholds(linemap, text_threshold, low_text, typical_top10_avg=.7):
    # Find average intensity of top 10% pixels
    # Do top 10% to account for pdfs that are mostly whitespace, etc.
    flat_map = linemap.flatten()
    sorted_map = np.sort(flat_map)[::-1]
    top_10_count = int(np.ceil(len(flat_map) * 0.1))
    top_10 = sorted_map[:top_10_count]
    avg_intensity = np.mean(top_10)

    # Adjust thresholds based on normalized intensityy
    scaling_factor = min(1, avg_intensity / typical_top10_avg) ** (1 / 2)

    low_text = max(low_text * scaling_factor, 0.1)
    text_threshold = max(text_threshold * scaling_factor, 0.15)

    low_text = min(low_text, 0.6)
    text_threshold = min(text_threshold, 0.8)
    return text_threshold, low_text


def detect_boxes(linemap, text_threshold, low_text):
    # From CRAFT - https://github.com/clovaai/CRAFT-pytorch
    # prepare data
    linemap = linemap.copy()
    img_h, img_w = linemap.shape

    text_threshold, low_text = get_dynamic_thresholds(linemap, text_threshold, low_text)

    ret, text_score = cv2.threshold(linemap, low_text, 1, cv2.THRESH_BINARY)

    text_score_comb = np.clip(text_score, 0, 1)
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    for k in range(1, label_count):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(linemap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(linemap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # boundary checks
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)),1, axis=0).transpose().reshape(-1,2)
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

        det.append(box)

    return det, labels


def get_detected_boxes(textmap, text_threshold=settings.DETECTOR_TEXT_THRESHOLD,  low_text=settings.DETECTOR_BLANK_THRESHOLD) -> List[PolygonBox]:
    textmap = textmap.copy()
    textmap = textmap.astype(np.float32)
    boxes, labels = detect_boxes(textmap, text_threshold, low_text)
    # From point form to box form
    boxes = [PolygonBox(polygon=box) for box in boxes]
    return boxes


def get_and_clean_boxes(textmap, processor_size, image_size) -> List[PolygonBox]:
    bboxes = get_detected_boxes(textmap)
    for bbox in bboxes:
        bbox.rescale(processor_size, image_size)
    bboxes = clean_contained_boxes(bboxes)
    return bboxes


def draw_bboxes_on_image(bboxes, image):
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=1)

    return image


def draw_polys_on_image(corners, image):
    draw = ImageDraw.Draw(image)

    for poly in corners:
        poly = [(p[0], p[1]) for p in poly]
        draw.polygon(poly, outline='red', width=1)

    return image


