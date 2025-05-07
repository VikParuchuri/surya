import re
from io import BytesIO
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from surya.debug.fonts import get_font_path
from surya.debug.render_html import render_text_as_html

try:
    from playwright.sync_api import sync_playwright

    has_playwright = True
except ImportError:
    has_playwright = False


def strip_html_tags(html_text):
    pattern = re.compile(r"<[\w/][^>]*>")
    text_only = pattern.sub("", html_text)

    return text_only


def get_text_size(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size):
    font = ImageFont.truetype(font_path, box_font_size)
    text_width, text_height = get_text_size(text, font)
    while (text_width > bbox_width or text_height > bbox_height) and box_font_size > 6:
        box_font_size = box_font_size - 1
        font = ImageFont.truetype(font_path, box_font_size)
        text_width, text_height = get_text_size(text, font)

    # Calculate text position (centered in bbox)
    text_width, text_height = get_text_size(text, font)
    x = s_bbox[0]
    y = s_bbox[1] + (bbox_height - text_height) / 2

    draw.text((x, y), text, fill="black", font=font)


def draw_text_with_playwright(
    bboxes, texts: List[str], image_size: Tuple[int, int]
) -> Image.Image:
    html_content, image_size = render_text_as_html(bboxes, texts, image_size)
    if not has_playwright:
        raise ImportError(
            "Playwright is not installed. Please install it using `pip install playwright`"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": image_size[0], "height": image_size[1]}
        )
        page.set_content(html_content)
        page.wait_for_timeout(1000)
        body = page.query_selector("body")
        image = body.screenshot()
        browser.close()

    pil_img = Image.open(BytesIO(image))
    return pil_img


def draw_text_on_image(
    bboxes,
    texts,
    image_size: Tuple[int, int],
    font_path=None,
    max_font_size=60,
    res_upscale=2,
) -> Image.Image:
    if has_playwright:
        return draw_text_with_playwright(bboxes, texts, image_size)

    texts = [strip_html_tags(text) for text in texts]
    if font_path is None:
        font_path = get_font_path()
    new_image_size = (image_size[0] * res_upscale, image_size[1] * res_upscale)
    image = Image.new("RGB", new_image_size, color="white")
    draw = ImageDraw.Draw(image)

    for bbox, text in zip(bboxes, texts):
        s_bbox = [int(coord * res_upscale) for coord in bbox]
        bbox_width = s_bbox[2] - s_bbox[0]
        bbox_height = s_bbox[3] - s_bbox[1]

        # Shrink the text to fit in the bbox if needed
        box_font_size = max(6, min(int(0.75 * bbox_height), max_font_size))
        render_text(
            draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size
        )

    return image
