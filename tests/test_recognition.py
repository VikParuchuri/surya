import time
from PIL import ImageDraw, Image
from surya.recognition.util import clean_math_tags


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


def test_recognition_clean_math():
    math = """<math display="block">na_n^{1+2r} \\text{cov}(\\hat{f}_n^{(r)}(x), \\hat{f}_n^{(r)}(y)) = \\frac{1}{n} \\sum_{j=1}^n \\frac{a_n^{1+2r}}{a_j^{1+2r}} \\text{cov}\\left(K^{(r)}\\left(\\frac{x-X_j}{a_j}\\right), K^{(r)}\\left(\\frac{y-X_j}{a_j}\\right)\\right) <br>+ \\frac{a_n^{1+2r}}{n} \\sum_{\\substack{j \\neq k \\\\ 1 \\le j, k \\le n}} \\frac{1}{(a_j a_k)^{1+r}} \\text{cov}\\left(K^{(r)}\\left(\\frac{x-X_j}{a_j}\\right), K^{(r)}\\left(\\frac{y-X_k}{a_k}\\right)\\right) <br>=: I_1 + I_2.</math> (1.7)</math>'"""
    clean_math = clean_math_tags(math)

    assert clean_math.count("</math>") == 1, "Should have one closing math tag"
    assert "<br>" not in clean_math, "Should not have <br> tags in cleaned math"


def test_recognition_clean_math_preserve_text():
    text = """Hello, this is a sentence with <math display="inline">x^2 + y^2 = z^2</math> and some text after it, with a weird tag <hello> and <goodbye>."""
    clean_text = clean_math_tags(text)

    assert clean_text == text
