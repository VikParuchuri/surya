from typing import List

import cv2
import numpy as np
from PIL import Image

from surya.common.util import clean_boxes
from surya.detection.affinity import get_vertical_lines
from surya.detection import TextDetectionResult
from surya.common.polygon import PolygonBox
from surya.settings import settings


def get_dynamic_thresholds(linemap, text_threshold, low_text, typical_top10_avg=0.7):
    # Find average intensity of top 10% pixels
    flat_map = linemap.ravel()
    top_10_count = int(len(flat_map) * 0.9)
    avg_intensity = np.mean(np.partition(flat_map, top_10_count)[top_10_count:])
    scaling_factor = np.clip(avg_intensity / typical_top10_avg, 0, 1) ** (1 / 2)

    low_text = np.clip(low_text * scaling_factor, 0.1, 0.6)
    text_threshold = np.clip(text_threshold * scaling_factor, 0.15, 0.8)

    return text_threshold, low_text


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

    for k in range(1, label_count):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # make segmentation map
        x, y, w, h = stats[k, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]

        try:
            niter = int(np.sqrt(min(w, h)))
        except ValueError:
            niter = 0

        buffer = 1
        sx, sy = max(0, x - niter - buffer), max(0, y - niter - buffer)
        ex, ey = min(img_w, x + w + niter + buffer), min(img_h, y + h + niter + buffer)

        mask = (labels[sy:ey, sx:ex] == k)
        line_max = np.max(linemap[sy:ey, sx:ex][mask])

        # thresholding
        if line_max < text_threshold:
            continue

        segmap = mask.astype(np.uint8)

        ksize = buffer + niter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        selected_segmap = cv2.dilate(segmap, kernel)

        # make box
        y_inds, x_inds = np.nonzero(selected_segmap)
        x_inds += sx
        y_inds += sy
        np_contours = np.column_stack((x_inds, y_inds))
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = np_contours[:, 0].min(), np_contours[:, 0].max()
            t, b = np_contours[:, 1].min(), np_contours[:, 1].max()
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)

        max_confidence = max(max_confidence, line_max)

        confidences.append(line_max)
        det.append(box)

    if max_confidence > 0:
        confidences = [c / max_confidence for c in confidences]
    return det, confidences


def get_detected_boxes(textmap, text_threshold=None, low_text=None) -> List[PolygonBox]:
    if text_threshold is None:
        text_threshold = settings.DETECTOR_TEXT_THRESHOLD
    if low_text is None:
        low_text = settings.DETECTOR_BLANK_THRESHOLD

    if textmap.dtype != np.float32:
        textmap = textmap.astype(np.float32)

    boxes, confidences = detect_boxes(textmap, text_threshold, low_text)
    # From point form to box form
    return [PolygonBox(polygon=box, confidence=confidence) for box, confidence in zip(boxes, confidences)]


def get_and_clean_boxes(textmap, processor_size, image_size, text_threshold=None, low_text=None) -> List[PolygonBox]:
    bboxes = get_detected_boxes(textmap, text_threshold, low_text)
    for bbox in bboxes:
        bbox.rescale(processor_size, image_size)
        bbox.fit_to_bounds([0, 0, image_size[0], image_size[1]])

    bboxes = clean_boxes(bboxes)
    return bboxes

def parallel_get_lines(preds, orig_sizes, include_maps=False):
    heatmap, affinity_map = preds
    heat_img, aff_img = None, None
    if include_maps:
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        aff_img = Image.fromarray((affinity_map * 255).astype(np.uint8))
    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))
    bboxes = get_and_clean_boxes(heatmap, heatmap_size, orig_sizes)
    vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes)

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]]
    )
    return result