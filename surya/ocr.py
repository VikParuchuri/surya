from collections import defaultdict
from typing import List
from tqdm import tqdm

import torch
from PIL import Image

from surya.detection import batch_detection
from surya.input.processing import slice_polys_from_image, slice_bboxes_from_image
from surya.recognition import batch_recognition


def run_recognition(images: List[Image.Image], langs: List[List[str]], rec_model, rec_processor, bboxes: List[List[List[int]]] = None, polygons: List[List[List[List[int]]]] = None):
    # Polygons need to be in corner format - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], bboxes in [x1, y1, x2, y2] format
    assert bboxes is not None or polygons is not None
    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, lang) in tqdm(enumerate(zip(images, langs)), desc="Slicing images"):
        if polygons is not None:
            slices = slice_polys_from_image(image, polygons[idx])
        else:
            slices = slice_bboxes_from_image(image, bboxes[idx])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([lang] * len(slices))

    rec_predictions = batch_recognition(all_slices, all_langs, rec_model, rec_processor)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, lang) in enumerate(zip(images, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        slice_start = slice_end

        pred = {
            "text_lines": image_lines,
            "language": lang
        }

        if polygons is not None:
            pred["polys"] = polygons[idx]
        else:
            pred["bboxes"] = bboxes[idx]

        predictions_by_image.append(pred)

    return predictions_by_image


def run_ocr(images: List[Image.Image], langs: List[List[str]], det_model, det_processor, rec_model, rec_processor):
    det_predictions = batch_detection(images, det_model, det_processor)
    if det_model.device == "cuda":
        torch.cuda.empty_cache() # Empty cache from first model run

    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, det_pred, lang) in tqdm(enumerate(zip(images, det_predictions, langs)), desc="Slicing images"):
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