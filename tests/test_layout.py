import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from surya.layout import batch_layout_detection
from PIL import Image, ImageDraw

def test_layout_topk(layout_model):
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello World", fill="black", font_size=72)
    draw.text((10, 200), "This is a sentence of text.\nNow it is a paragraph.\nA three-line one.", fill="black",
              font_size=24)

    layout_results = batch_layout_detection([image], layout_model, layout_model.processor)

    assert len(layout_results) == 1
    assert layout_results[0].image_bbox == [0, 0, 1024, 1024]

    bboxes = layout_results[0].bboxes
    assert len(bboxes) == 2

    assert bboxes[0].label == "SectionHeader"
    assert len(bboxes[0].top_k) == 5

    assert bboxes[1].label == "Text"
    assert len(bboxes[1].top_k) == 5
