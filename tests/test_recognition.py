import time
from PIL import ImageDraw, Image


def test_recognition(recognition_predictor, detection_predictor, test_image):
    recognition_results = recognition_predictor([test_image], None, detection_predictor)

    assert len(recognition_results) == 1
    assert recognition_results[0].image_bbox == [0, 0, 1024, 1024]

    text_lines = recognition_results[0].text_lines
    assert len(text_lines) == 4
    assert "Hello World" in text_lines[0].text


def test_recognition_input_text(recognition_predictor, detection_predictor, test_image):
    start = time.time()
    recognition_predictor([test_image], None, detection_predictor)
    end = time.time() - start

    input_text = "a" * 400
    start2 = time.time()
    recognition_results = recognition_predictor(
        [test_image], None, detection_predictor, input_text=[input_text]
    )
    end2 = time.time() - start2

    assert max([end, end2]) / min([end, end2]) < 1.5, (
        "Input text should be truncated and not change inference time"
    )

    assert len(recognition_results) == 1
    assert recognition_results[0].image_bbox == [0, 0, 1024, 1024]

    text_lines = recognition_results[0].text_lines
    assert len(text_lines) == 4
    assert "Hello World" in text_lines[0].text


def test_recognition_drop_repeats(recognition_predictor, detection_predictor):
    image = Image.new("RGB", (1024, 128), "white")
    draw = ImageDraw.Draw(image)
    text = "a" * 80
    draw.text((5, 5), text, fill="black", font_size=24)

    recognition_results = recognition_predictor(
        [image], None, bboxes=[[[0, 0, 1024, 128]]], drop_repeated_text=True
    )
    assert len(recognition_results) == 1
    result = recognition_results[0].text_lines
    assert result[0].text == ""
