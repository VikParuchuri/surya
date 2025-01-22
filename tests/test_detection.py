def test_detection(detection_predictor, test_image):
    detection_results = detection_predictor([test_image])

    assert len(detection_results) == 1
    assert detection_results[0].image_bbox == [0, 0, 1024, 1024]

    bboxes = detection_results[0].bboxes
    assert len(bboxes) == 4