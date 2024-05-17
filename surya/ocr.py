from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import List
from tqdm import tqdm

import torch
from PIL import Image

from surya.detection import batch_text_detection
from surya.input.processing import slice_polys_from_image, slice_bboxes_from_image
from surya.postprocessing.text import truncate_repetitions, sort_text_lines
from surya.recognition import batch_recognition
from surya.schema import TextLine, OCRResult
from surya.settings import settings


def run_recognition(images: List[Image.Image], langs: List[List[str]], rec_model, rec_processor, bboxes: List[List[List[int]]] = None, polygons: List[List[List[List[int]]]] = None, batch_size=None) -> List[OCRResult]:
    # Polygons need to be in corner format - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], bboxes in [x1, y1, x2, y2] format
    assert bboxes is not None or polygons is not None
    assert len(images) == len(langs), "You need to pass in one list of languages for each image"
    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, lang) in enumerate(zip(images, langs)):
        if polygons is not None:
            slices = slice_polys_from_image(image, polygons[idx])
        else:
            slices = slice_bboxes_from_image(image, bboxes[idx])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([lang] * len(slices))

    rec_predictions, _ = batch_recognition(all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, lang) in enumerate(zip(images, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        slice_start = slice_end

        text_lines = []
        for i in range(len(image_lines)):
            if polygons is not None:
                poly = polygons[idx][i]
            else:
                bbox = bboxes[idx][i]
                poly = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]

            text_lines.append(TextLine(
                text=image_lines[i],
                polygon=poly
            ))

        pred = OCRResult(
            text_lines=text_lines,
            languages=lang,
            image_bbox=[0, 0, image.size[0], image.size[1]]
        )
        predictions_by_image.append(pred)

    return predictions_by_image


def parallel_slice_polys(det_pred, image):
    polygons = [p.polygon for p in det_pred.bboxes]
    slices = slice_polys_from_image(image, polygons)
    return slices


def run_ocr(images: List[Image.Image], langs: List[List[str]], det_model, det_processor, rec_model, rec_processor, batch_size=None) -> List[OCRResult]:
    det_predictions = batch_text_detection(images, det_model, det_processor)
    if det_model.device.type == "cuda":
        torch.cuda.empty_cache() # Empty cache from first model run

    all_slices = []

    if settings.IN_STREAMLIT:
        all_slices = [parallel_slice_polys(det_pred, image) for det_pred, image in zip(det_predictions, images)]
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=settings.DETECTOR_POSTPROCESSING_CPU_WORKERS) as executor:
            for image_idx in range(len(images)):
                future = executor.submit(parallel_slice_polys, det_predictions[image_idx], images[image_idx])
                futures.append(future)

            for future in futures:
                all_slices.append(future.result())

    slice_map = []
    all_langs = []
    for idx, (slice, lang) in enumerate(zip(all_slices, langs)):
        slice_map.append(len(slice))
        all_langs.extend([lang] * len(slice))

    all_slices = [slice for sublist in all_slices for slice in sublist]

    rec_predictions, confidence_scores = batch_recognition(all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, det_pred, lang) in enumerate(zip(images, det_predictions, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        line_confidences = confidence_scores[slice_start:slice_end]
        slice_start = slice_end

        assert len(image_lines) == len(det_pred.bboxes)

        lines = []
        for text_line, confidence, bbox in zip(image_lines, line_confidences, det_pred.bboxes):
            lines.append(TextLine(
                text=text_line,
                polygon=bbox.polygon,
                bbox=bbox.bbox,
                confidence=confidence
            ))

        lines = sort_text_lines(lines)

        predictions_by_image.append(OCRResult(
            text_lines=lines,
            languages=lang,
            image_bbox=det_pred.image_bbox
        ))

    return predictions_by_image
