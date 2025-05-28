import time


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
