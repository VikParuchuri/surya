from PIL import Image, ImageDraw

from surya.recognition import TaskNames


def test_latex_ocr(recognition_predictor):
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "E = mc2", fill='black', font_size=48)

    results = recognition_predictor([img], [TaskNames.block_without_boxes], bboxes= [[[0, 0, 200, 100]]])
    text = results[0].text.strip()
    assert len(results) == 1

    assert text.startswith("<math")
    assert text.endswith("</math>")