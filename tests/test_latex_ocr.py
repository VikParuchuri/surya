from typing import List

from PIL import Image, ImageDraw

from surya.common.surya.schema import TaskNames
from surya.recognition import OCRResult


def test_latex_ocr(recognition_predictor):
    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "E = mc2", fill="black", font_size=48)

    results: List[OCRResult] = recognition_predictor(
        [img], [TaskNames.block_without_boxes], bboxes=[[[0, 0, 200, 100]]]
    )
    text = results[0].text_lines[0].text
    assert len(results) == 1

    assert text.startswith("<math")
    assert text.endswith("</math>")
