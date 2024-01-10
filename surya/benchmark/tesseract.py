import numpy as np
import pytesseract
from pytesseract import Output


def tesseract_bboxes(img):
    arr_img = np.asarray(img, dtype=np.uint8)
    ocr = pytesseract.image_to_data(arr_img, output_type=Output.DICT)

    bboxes = []
    n_boxes = len(ocr['level'])
    for i in range(n_boxes):
        # TODO: it is possible to merge by line here, but it gives bad results.  Find another way to get line-level.
        _, x, y, w, h = ocr['text'][i], ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i]
        bbox = (x, y, x + w, y + h)
        bboxes.append(bbox)

    return bboxes