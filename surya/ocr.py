from collections import defaultdict
from typing import List
from PIL import Image

from surya.detection import batch_detection
from surya.input.processing import slice_polys_from_image
from surya.recognition import batch_recognition


def run_ocr(images: List[Image.Image], langs: List[List[str]], det_model, det_processor, rec_model, rec_processor):
    det_predictions = batch_detection(images, det_model, det_processor)

    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, det_pred, lang) in enumerate(zip(images, det_predictions, langs)):
        slices = slice_polys_from_image(image, det_pred["polygons"])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([lang] * len(slices))

    rec_predictions = batch_recognition(all_slices, all_langs, rec_model, rec_processor)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, det_pred, lang) in enumerate(zip(images, det_predictions, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        slice_start = slice_end

        assert len(image_lines) == len(det_pred["polygons"]) == len(det_pred["bboxes"])
        predictions_by_image.append({
            "text_lines": image_lines,
            "polys": det_pred["polygons"],
            "bboxes": det_pred["bboxes"],
            "language": lang
        })

    return predictions_by_image