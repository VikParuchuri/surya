def test_garbled_text(ocr_error_predictor):
    text = """"
    ; dh vksj ls mifLFkr vf/koDrk % Jh vfuy dqekj
    2. vfHk;qDr dh vksj ls mifLFkr vf/koDrk % Jh iznhi d
    """.strip()
    results = ocr_error_predictor([text])
    assert results.labels[0] == "bad"


def test_good_text(ocr_error_predictor):
    text = """"
    There are professions more harmful than industrial design, but only a very few of them.
    """.strip()
    results = ocr_error_predictor([text])
    assert results.labels[0] == "good"