from surya.ocr_error import batch_ocr_error_detection


def test_garbled_text(ocr_error_model):
    text = """"
    ; dh vksj ls mifLFkr vf/koDrk % Jh vfuy dqekj
    2. vfHk;qDr dh vksj ls mifLFkr vf/koDrk % Jh iznhi d
    """.strip()
    results = batch_ocr_error_detection([text], ocr_error_model, ocr_error_model.processor)
    assert results.labels[0] == "bad"


def test_good_text(ocr_error_model):
    text = """"
    There are professions more harmful than industrial design, but only a very few of them.
    """.strip()
    results = batch_ocr_error_detection([text], ocr_error_model, ocr_error_model.processor)
    assert results.labels[0] == "good"