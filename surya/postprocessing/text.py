import os
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from surya.postprocessing.fonts import get_font_path
from surya.schema import TextLine
from surya.settings import settings
from surya.postprocessing.math.latex import is_latex


def sort_text_lines(lines: List[TextLine], tolerance=1.25):
    # Sorts in reading order.  Not 100% accurate, this should only
    # be used as a starting point for more advanced sorting.
    vertical_groups = {}
    for line in lines:
        group_key = round(line.bbox[1] / tolerance) * tolerance
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(line)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_lines = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(group, key=lambda x: x.bbox[0])
        sorted_lines.extend(sorted_group)

    return sorted_lines


def truncate_repetitions(text: str, min_len=15):
    # From nougat, with some cleanup
    if len(text) < 2 * min_len:
        return text

    # try to find a length at which the tail is repeating
    max_rep_len = None
    for rep_len in range(min_len, int(len(text) / 2)):
        # check if there is a repetition at the end
        same = True
        for i in range(0, rep_len):
            if text[len(text) - rep_len - i - 1] != text[len(text) - i - 1]:
                same = False
                break

        if same:
            max_rep_len = rep_len

    if max_rep_len is None:
        return text

    lcs = text[-max_rep_len:]

    # remove all but the last repetition
    text_to_truncate = text
    while text_to_truncate.endswith(lcs):
        text_to_truncate = text_to_truncate[:-max_rep_len]

    return text[:len(text_to_truncate)]


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


def render_math(image, draw, text, s_bbox, bbox_width, bbox_height, font_path):
        try:
            from surya.postprocessing.math.render import latex_to_pil
            box_font_size = max(10, min(int(.2 * bbox_height), 24))
            img = latex_to_pil(text, bbox_width, bbox_height, fontsize=box_font_size)
            img.thumbnail((bbox_width, bbox_height))
            image.paste(img, (s_bbox[0], s_bbox[1]))
        except Exception as e:
            print(f"Failed to render math: {e}")
            box_font_size = max(10, min(int(.75 * bbox_height), 24))
            render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size)


def draw_text_on_image(bboxes, texts, image_size: Tuple[int, int], langs: List[str], font_path=None, max_font_size=60, res_upscale=2, has_math=False):
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
        if has_math and is_latex(text):
            render_math(image, draw, text, s_bbox, bbox_width, bbox_height, font_path)
        else:
            box_font_size = max(6, min(int(.75 * bbox_height), max_font_size))
            render_text(draw, text, s_bbox, bbox_width, bbox_height, font_path, box_font_size)

    return image
