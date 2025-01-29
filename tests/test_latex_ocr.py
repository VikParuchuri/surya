from PIL import Image, ImageDraw


def test_latex_ocr(texify_predictor):
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "E = mc2", fill='black', font_size=48)

    results = texify_predictor([img])
    text = results[0].text.strip()
    assert len(results) == 1

    assert text.startswith("<math")
    assert text.endswith("</math>")