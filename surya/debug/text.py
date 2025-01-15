from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from surya.debug.fonts import get_font_path


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


def draw_text_on_image(bboxes, texts, image_size: Tuple[int, int], langs: List[str], font_path=None, max_font_size=60, res_upscale=2):
    if font_path is None:
        font_path = get_font_path(langs)
    new_image_size = (image_size[0] * res_upscale, image_size[1] * res_upscale)
    image = Image.new('RGB', new_image_size, color='white')
    draw = ImageDraw.Draw(image)

    for bbox, text in zip(bboxes, texts):
        s_bbox = [int(coord * res_upscale) for coord in bbox]
        bbox_width = s_bbox[2] - s_bbox[0]
        bbox_height = s_bbox[3] - s_bbox[1]

        # Shrink the text to fit in the bbox if needed
        box_font_size = max(6, min(int(.75 * bbox_height), max_font_size))
        render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size)

    return image
