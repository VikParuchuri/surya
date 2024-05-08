from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional
from PIL import Image
import numpy as np

from surya.detection import batch_detection
from surya.postprocessing.heatmap import keep_largest_boxes, get_and_clean_boxes, get_detected_boxes
from surya.schema import LayoutResult, LayoutBox, TextDetectionResult
from surya.settings import settings


def get_regions_from_detection_result(detection_result: TextDetectionResult, heatmaps: List[Image.Image], orig_size, id2label, segment_assignment, vertical_line_width=20) -> List[LayoutBox]:
    logits = np.stack(heatmaps, axis=0)
    vertical_line_bboxes = [line for line in detection_result.vertical_lines]
    line_bboxes = detection_result.bboxes

    # Scale back to processor size
    for line in vertical_line_bboxes:
        line.rescale_bbox(orig_size, list(reversed(heatmaps[0].shape)))

    for line in line_bboxes:
        line.rescale(orig_size, list(reversed(heatmaps[0].shape)))

    for bbox in vertical_line_bboxes:
        # Give some width to the vertical lines
        vert_bbox = list(bbox.bbox)
        vert_bbox[2] = min(heatmaps[0].shape[0], vert_bbox[2] + vertical_line_width)

        logits[:, vert_bbox[1]:vert_bbox[3], vert_bbox[0]:vert_bbox[2]] = 0  # zero out where the column lines are

    logits[:, logits[0] >= .5] = 0 # zero out where blanks are

    # Zero out where other segments are
    for i in range(logits.shape[0]):
        logits[i, segment_assignment != i] = 0

    detected_boxes = []
    for heatmap_idx in range(1, len(id2label)):  # Skip the blank class
        heatmap = logits[heatmap_idx]
        bboxes = get_detected_boxes(heatmap)
        bboxes = [bbox for bbox in bboxes if bbox.area > 25]
        for bb in bboxes:
            bb.fit_to_bounds([0, 0, heatmap.shape[1] - 1, heatmap.shape[0] - 1])

        for bbox in bboxes:
            detected_boxes.append(LayoutBox(polygon=bbox.polygon, label=id2label[heatmap_idx], confidence=1))

    detected_boxes = sorted(detected_boxes, key=lambda x: x.confidence, reverse=True)
    # Expand bbox to cover intersecting lines
    box_lines = defaultdict(list)
    used_lines = set()

    # We try 2 rounds of identifying the correct lines to snap to
    # First round is majority intersection, second lowers the threshold
    for thresh in [.5, .4]:
        for bbox_idx, bbox in enumerate(detected_boxes):
            for line_idx, line_bbox in enumerate(line_bboxes):
                if line_bbox.intersection_pct(bbox) > thresh and line_idx not in used_lines:
                    box_lines[bbox_idx].append(line_bbox.bbox)
                    used_lines.add(line_idx)

    new_boxes = []
    for bbox_idx, bbox in enumerate(detected_boxes):
        if bbox.label == "Picture" and bbox.area < 200: # Remove very small figures
            continue

        # Skip if we didn't find any lines to snap to, except for Pictures and Formulas
        if bbox_idx not in box_lines and bbox.label not in ["Picture", "Formula"]:
            continue

        covered_lines = box_lines[bbox_idx]
        # Snap non-picture layout boxes to correct text boundaries
        if len(covered_lines) > 0 and bbox.label not in ["Picture"]:
            min_x = min([line[0] for line in covered_lines])
            min_y = min([line[1] for line in covered_lines])
            max_x = max([line[2] for line in covered_lines])
            max_y = max([line[3] for line in covered_lines])

            # Tables and formulas can contain text, but text isn't the whole area
            if bbox.label in ["Table", "Formula"]:
                min_x_box = min([b[0] for b in bbox.polygon])
                min_y_box = min([b[1] for b in bbox.polygon])
                max_x_box = max([b[0] for b in bbox.polygon])
                max_y_box = max([b[1] for b in bbox.polygon])

                min_x = min(min_x, min_x_box)
                min_y = min(min_y, min_y_box)
                max_x = max(max_x, max_x_box)
                max_y = max(max_y, max_y_box)

            bbox.polygon[0][0] = min_x
            bbox.polygon[0][1] = min_y
            bbox.polygon[1][0] = max_x
            bbox.polygon[1][1] = min_y
            bbox.polygon[2][0] = max_x
            bbox.polygon[2][1] = max_y
            bbox.polygon[3][0] = min_x
            bbox.polygon[3][1] = max_y

        if bbox_idx in box_lines and bbox.label in ["Picture"]:
            bbox.label = "Figure"

        new_boxes.append(bbox)

    # Merge tables together (sometimes one column is detected as a separate table)
    for i in range(5): # Up to 5 rounds of merging
        to_remove = set()
        for bbox_idx, bbox in enumerate(new_boxes):
            if bbox.label != "Table" or bbox_idx in to_remove:
                continue

            for bbox_idx2, bbox2 in enumerate(new_boxes):
                if bbox2.label != "Table" or bbox_idx2 in to_remove or bbox_idx == bbox_idx2:
                    continue

                if bbox.intersection_pct(bbox2) > 0:
                    bbox.merge(bbox2)
                    to_remove.add(bbox_idx2)

        new_boxes = [bbox for idx, bbox in enumerate(new_boxes) if idx not in to_remove]

    # Ensure we account for all text lines in the layout
    unused_lines = [line for idx, line in enumerate(line_bboxes) if idx not in used_lines]
    for bbox in unused_lines:
        new_boxes.append(LayoutBox(polygon=bbox.polygon, label="Text", confidence=.5))

    for bbox in new_boxes:
        bbox.rescale(list(reversed(heatmap.shape)), orig_size)

    detected_boxes = [bbox for bbox in new_boxes if bbox.area > 16]

    # Remove bboxes contained inside others, unless they're captions
    contained_bbox = []
    for i, bbox in enumerate(detected_boxes):
        for j, bbox2 in enumerate(detected_boxes):
            if i == j:
                continue

            if bbox2.intersection_pct(bbox) >= .95 and bbox2.label not in ["Caption"]:
                contained_bbox.append(j)

    detected_boxes = [bbox for idx, bbox in enumerate(detected_boxes) if idx not in contained_bbox]

    return detected_boxes


def get_regions(heatmaps: List[Image.Image], orig_size, id2label, segment_assignment) -> List[LayoutBox]:
    bboxes = []
    for i in range(1, len(id2label)):  # Skip the blank class
        heatmap = heatmaps[i]
        assert heatmap.shape == segment_assignment.shape
        heatmap[segment_assignment != i] = 0  # zero out where another segment is
        bbox = get_and_clean_boxes(heatmap, list(reversed(heatmap.shape)), orig_size)
        for bb in bbox:
            bboxes.append(LayoutBox(polygon=bb.polygon, label=id2label[i]))
        heatmaps.append(heatmap)

    bboxes = keep_largest_boxes(bboxes)
    return bboxes


def parallel_get_regions(heatmaps: List[Image.Image], orig_size, id2label, detection_results=None) -> List[LayoutResult]:
    logits = np.stack(heatmaps, axis=0)
    segment_assignment = logits.argmax(axis=0)
    if detection_results is not None:
        bboxes = get_regions_from_detection_result(detection_results, heatmaps, orig_size, id2label,
                                                   segment_assignment)
    else:
        bboxes = get_regions(heatmaps, orig_size, id2label, segment_assignment)

    segmentation_img = Image.fromarray(segment_assignment.astype(np.uint8))

    result = LayoutResult(
        bboxes=bboxes,
        segmentation_map=segmentation_img,
        heatmaps=heatmaps,
        image_bbox=[0, 0, orig_size[0], orig_size[1]]
    )

    return result


def batch_layout_detection(images: List, model, processor, detection_results: Optional[List[TextDetectionResult]] = None, batch_size=None) -> List[LayoutResult]:
    preds, orig_sizes = batch_detection(images, model, processor, batch_size=batch_size)
    id2label = model.config.id2label

    results = []
    if len(images) == 1: # Ensures we don't parallelize with streamlit
        for i in range(len(images)):
            result = parallel_get_regions(preds[i], orig_sizes[i], id2label, detection_results[i] if detection_results else None)
            results.append(result)
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=settings.DETECTOR_POSTPROCESSING_CPU_WORKERS) as executor:
            for i in range(len(images)):
                future = executor.submit(parallel_get_regions, preds[i], orig_sizes[i], id2label, detection_results[i] if detection_results else None)
                futures.append(future)

            for future in futures:
                results.append(future.result())

    return results