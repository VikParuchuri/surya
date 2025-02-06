def test_detection(detection_predictor, test_image):
    detection_results = detection_predictor([test_image])

    assert len(detection_results) == 1
    assert detection_results[0].image_bbox == [0, 0, 1024, 1024]

    bboxes = detection_results[0].bboxes
    assert len(bboxes) == 4


def test_detection_chunking(detection_predictor, test_image_tall):
    detection_results = detection_predictor([test_image_tall])

    assert len(detection_results) == 1
    assert detection_results[0].image_bbox == [0, 0, 4096, 4096]

    bboxes = detection_results[0].bboxes
    assert len(bboxes) >= 3 # Sometimes merges into 3
    assert abs(4000 - bboxes[1].polygon[0][0]) < 50