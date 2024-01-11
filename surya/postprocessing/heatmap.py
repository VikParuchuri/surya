import numpy as np
import cv2
import math
from PIL import ImageDraw

from surya.postprocessing.util import rescale_bbox
from surya.settings import settings


def clean_contained_boxes(boxes):
    new_boxes = []
    for box in boxes:
        contained = False
        for other_box in boxes:
            if box == other_box:
                continue
            if box[0] >= other_box[0] and box[1] >= other_box[1] and box[2] <= other_box[2] and box[3] <= other_box[3]:
                contained = True
                break
        if not contained:
            new_boxes.append(box)
    return new_boxes


def detect_boxes(linemap, text_threshold, low_text):
    # From CRAFT - https://github.com/clovaai/CRAFT-pytorch
    # prepare data
    linemap = linemap.copy()
    img_h, img_w = linemap.shape

    ret, text_score = cv2.threshold(linemap, low_text, 1, 0)

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


def get_detected_boxes(textmap, text_threshold=settings.DETECTOR_TEXT_THRESHOLD,  low_text=settings.DETECTOR_NMS_THRESHOLD):
    textmap = textmap.copy()
    textmap = textmap.astype(np.float32)
    boxes, labels = detect_boxes(textmap, text_threshold, low_text)
    # From point form to box form
    boxes = [
        [box[0][0], box[0][1], box[1][0], box[2][1]]
        for box in boxes
    ]

    # Ensure correct box format
    for box in boxes:
        if box[0] > box[2]:
            box[0], box[2] = box[2], box[0]
        if box[1] > box[3]:
            box[1], box[3] = box[3], box[1]
    return boxes


def get_and_clean_boxes(textmap, processor_size, image_size):
    bboxes = get_detected_boxes(textmap)
    bboxes = [rescale_bbox(bbox, processor_size, image_size) for bbox in bboxes]
    bboxes = clean_contained_boxes(bboxes)
    return bboxes


def draw_bboxes_on_image(bboxes, image):
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=1)

    return image

