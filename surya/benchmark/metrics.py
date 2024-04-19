from functools import partial
from itertools import repeat

import numpy as np
from concurrent.futures import ProcessPoolExecutor

def intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def intersection_pixels(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return set()

    x_left, x_right = int(x_left), int(x_right)
    y_top, y_bottom = int(y_top), int(y_bottom)

    coords = np.meshgrid(np.arange(x_left, x_right), np.arange(y_top, y_bottom))
    pixels = set(zip(coords[0].flat, coords[1].flat))

    return pixels


def calculate_coverage(box, other_boxes, penalize_double=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    # find total coverage of the box
    covered_pixels = set()
    double_coverage = list()
    for other_box in other_boxes:
        ia = intersection_pixels(box, other_box)
        double_coverage.append(list(covered_pixels.intersection(ia)))
        covered_pixels = covered_pixels.union(ia)

    # Penalize double coverage - having multiple bboxes overlapping the same pixels
    double_coverage_penalty = len(double_coverage)
    if not penalize_double:
        double_coverage_penalty = 0
    covered_pixels_count = max(0, len(covered_pixels) - double_coverage_penalty)
    return covered_pixels_count / box_area


def calculate_coverage_fast(box, other_boxes, penalize_double=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    total_intersect = 0
    for other_box in other_boxes:
        total_intersect += intersection_area(box, other_box)

    return min(1, total_intersect / box_area)


def precision_recall(preds, references, threshold=.5, workers=8, penalize_double=True):
    if len(references) == 0:
        return {
            "precision": 1,
            "recall": 1,
        }

    if len(preds) == 0:
        return {
            "precision": 0,
            "recall": 0,
        }

    # If we're not penalizing double coverage, we can use a faster calculation
    coverage_func = calculate_coverage_fast
    if penalize_double:
        coverage_func = calculate_coverage

    with ProcessPoolExecutor(max_workers=workers) as executor:
        precision_func = partial(coverage_func, penalize_double=penalize_double)
        precision_iou = executor.map(precision_func, preds, repeat(references))
        reference_iou = executor.map(coverage_func, references, repeat(preds))

    precision_classes = [1 if i > threshold else 0 for i in precision_iou]
    precision = sum(precision_classes) / len(precision_classes)

    recall_classes = [1 if i > threshold else 0 for i in reference_iou]
    recall = sum(recall_classes) / len(recall_classes)

    return {
        "precision": precision,
        "recall": recall,
    }


def mean_coverage(preds, references):
    coverages = []

    for box1 in references:
        coverage = calculate_coverage(box1, preds)
        coverages.append(coverage)

    for box2 in preds:
        coverage = calculate_coverage(box2, references)
        coverages.append(coverage)

    # Calculate the average coverage over all comparisons
    if len(coverages) == 0:
        return 0
    coverage = sum(coverages) / len(coverages)
    return {"coverage": coverage}


def rank_accuracy(preds, references):
    # Preds and references need to be aligned so each position refers to the same bbox
    pairs = []
    for i, pred in enumerate(preds):
        for j, pred2 in enumerate(preds):
            if i == j:
                continue
            pairs.append((i, j, pred > pred2))

    # Find how many of the prediction rankings are correct
    correct = 0
    for i, ref in enumerate(references):
        for j, ref2 in enumerate(references):
            if (i, j, ref > ref2) in pairs:
                correct += 1

    return correct / len(pairs)