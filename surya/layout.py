from typing import List
from PIL import Image
import numpy as np

from surya.detection import batch_detection
from surya.postprocessing.heatmap import keep_largest_boxes, get_and_clean_boxes
from surya.schema import LayoutResult, LayoutBox


def batch_layout_detection(images: List, model, processor) -> List[LayoutResult]:
    preds, orig_sizes = batch_detection(images, model, processor)
    id2label = model.config.id2label

    results = []
    for i in range(len(images)):
        heatmaps = preds[i]
        orig_size = orig_sizes[i]
        logits = np.stack(heatmaps, axis=0)
        segment_assignment = logits.argmax(dim=0)

        bboxes = []
        for i in range(1, len(id2label)):  # Skip the blank class
            heatmap = heatmaps[i]
            heatmap[segment_assignment != i] = 0  # zero out where another segment is
            bbox = get_and_clean_boxes(heatmap, list(reversed(heatmap.shape)), orig_size, low_text=.6, text_threshold=.8)
            for bb in bbox:
                bboxes.append(LayoutBox(polygon=bb.polygon, label=id2label[i]))
            heatmaps.append(heatmap)

        bboxes = keep_largest_boxes(bboxes)
        segmentation_img = Image.fromarray(segment_assignment.astype(np.uint8))

        result_bboxes = [LayoutBox(polygon=bb[1].polygon, label=bb[0]) for bb in bboxes]
        result = LayoutResult(
            bboxes=result_bboxes,
            segmentation_map=segmentation_img,
            image_bbox=[0, 0, orig_size[0], orig_size[1]]
        )

        results.append(result)

    return results