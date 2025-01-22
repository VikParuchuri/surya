def test_layout_topk(layout_predictor, test_image):
    layout_results = layout_predictor([test_image])

    assert len(layout_results) == 1
    assert layout_results[0].image_bbox == [0, 0, 1024, 1024]

    bboxes = layout_results[0].bboxes
    assert len(bboxes) == 2

    assert bboxes[0].label == "SectionHeader"
    assert len(bboxes[0].top_k) == 5

    assert bboxes[1].label == "Text"
    assert len(bboxes[1].top_k) == 5
