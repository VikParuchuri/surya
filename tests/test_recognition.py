def test_recognition(recognition_predictor, detection_predictor, test_image):
    recognition_results = recognition_predictor([test_image], [None], detection_predictor)

    assert len(recognition_results) == 1
    assert recognition_results[0].image_bbox == [0, 0, 1024, 1024]

    text_lines = recognition_results[0].text_lines
    assert len(text_lines) == 4
    assert text_lines[0].text == "Hello World"