import numpy as np
import pytesseract
from pytesseract import Output
from surya.settings import settings
import os
from concurrent.futures import ProcessPoolExecutor


def tesseract_bboxes(img):
    arr_img = np.asarray(img, dtype=np.uint8)
    ocr = pytesseract.image_to_data(arr_img, output_type=Output.DICT)

    bboxes = []
    n_boxes = len(ocr['level'])
    for i in range(n_boxes):
        # It is possible to merge by line here with line number, but it gives bad results.
        _, x, y, w, h = ocr['text'][i], ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i]
        bbox = (x, y, x + w, y + h)
        bboxes.append(bbox)

    return bboxes


def tesseract_parallel(imgs):
    # Tesseract uses 4 threads per instance
    tess_parallel_cores = min(len(imgs), settings.DETECTOR_BATCH_SIZE)
    cpus = os.cpu_count()
    tess_parallel_cores = min(tess_parallel_cores, cpus)

    # Tesseract uses 4 threads per instance
    tess_parallel = max(tess_parallel_cores // 4, 1)

    with ProcessPoolExecutor(max_workers=tess_parallel) as executor:
        tess_bboxes = executor.map(tesseract_bboxes, imgs)
        tess_bboxes = list(tess_bboxes)
    return tess_bboxes