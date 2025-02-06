from PIL import Image, ImageDraw


def test_inline_detection(inline_detection_predictor, detection_predictor):
    img = Image.new('RGB', (1024, 1024), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 200), "where B(x, ϵ) is the norm ball with radius xadv = x + ϵ · sgn (∇xL(f (x, w), y))", fill='black', font_size=48)

    detection_results = detection_predictor([img])
    bboxes = [[bb.bbox for bb in detection_results[0].bboxes]]

    inline_detection_results = inline_detection_predictor([img], bboxes)

    assert len(inline_detection_results) == 1
    assert inline_detection_results[0].image_bbox == [0, 0, 1024, 1024]

    bboxes = inline_detection_results[0].bboxes
    assert len(bboxes) == 2