import re
from typing import List, Tuple

import numpy
import torch

from surya.common.polygon import PolygonBox
from surya.recognition.schema import TextLine, TextWord, TextChar

MATH_SYMBOLS = ["+", "-", "*", "=", "^", "_", "\\", "{", "}"]


def unwrap_math(text: str) -> str:
    if len(text) > 50:
        return text

    # Detected as math, but does not contain LaTeX commands
    if (
        re.match(r'^\s*<math(?:\s+display="inline")?.*?</math>\s*$', text, re.DOTALL)
        and text.count("<math") == 1
        and not any([symb in text for symb in MATH_SYMBOLS])
    ):
        # Remove math tags
        text = re.sub(r"<math.*?>", "", text)
        text = re.sub(r"</math>", "", text)

    return text


def detect_repeat_token(predicted_tokens: List[int], max_repeats: int = 40):
    if len(predicted_tokens) < max_repeats:
        return False

    # Detect repeats containing 1 or 2 tokens
    last_n = predicted_tokens[-max_repeats:]
    unique_tokens = len(set(last_n))
    if unique_tokens > 5:
        return False

    return last_n[-unique_tokens:] == last_n[-unique_tokens * 2 : -unique_tokens]


def sort_text_lines(lines: List[TextLine] | List[dict], tolerance=1.25):
    # Sorts in reading order.  Not 100% accurate, this should only
    # be used as a starting point for more advanced sorting.
    vertical_groups = {}
    for line in lines:
        group_key = (
            round(
                line.bbox[1]
                if isinstance(line, TextLine)
                else line["bbox"][1] / tolerance
            )
            * tolerance
        )
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(line)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_lines = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(
            group, key=lambda x: x.bbox[0] if isinstance(x, TextLine) else x["bbox"][0]
        )
        sorted_lines.extend(sorted_group)

    return sorted_lines


def clean_close_polygons(bboxes: List[List[List[int]]], thresh: float = 0.1):
    if len(bboxes) < 2:
        return bboxes

    new_bboxes = [bboxes[0]]
    for i in range(1, len(bboxes)):
        close = True
        prev_bbox = bboxes[i - 1]
        bbox = bboxes[i]
        for j in range(4):
            if (
                abs(bbox[j][0] - prev_bbox[j][0]) > thresh
                or abs(bbox[j][1] - prev_bbox[j][1]) > thresh
            ):
                close = False
                break

        if not close:
            new_bboxes.append(bboxes[i])

    return new_bboxes


def words_from_chars(chars: List[TextChar], line_box: PolygonBox):
    words = []
    word = None
    for i, char in enumerate(chars):
        if not char.bbox_valid:
            if word:
                words.append(word)
                word = None
            continue

        if not word:
            word = TextWord(**char.model_dump())

            # Fit bounds to line if first word
            if i == 0:
                word.merge_left(line_box)

        elif not char.text.strip():
            if word:
                words.append(word)
            word = None
        else:
            # Merge bboxes
            word.merge(char)
            word.text = word.text + char.text

            if i == len(chars) - 1:
                word.merge_right(line_box)
    if word:
        words.append(word)

    return words


def prediction_to_polygon_batch(
    pred: torch.Tensor,
    img_sizes: List[Tuple[int, int]],
    bbox_scaler,
    skew_scaler,
    skew_min=0.001,
):
    img_sizes = torch.from_numpy(numpy.array(img_sizes, dtype=numpy.float32)).to(
        pred.device
    )
    w_scale = (img_sizes[:, 1] / bbox_scaler)[:, None, None]
    h_scale = (img_sizes[:, 0] / bbox_scaler)[:, None, None]

    cx = pred[:, :, 0]
    cy = pred[:, :, 1]
    width = pred[:, :, 2]
    height = pred[:, :, 3]

    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = cx + width / 2
    y2 = cy + height / 2

    skew_x = torch.floor((pred[:, :, 4] - skew_scaler) / 2)
    skew_y = torch.floor((pred[:, :, 5] - skew_scaler) / 2)

    skew_x[torch.abs(skew_x) < skew_min] = 0
    skew_y[torch.abs(skew_y) < skew_min] = 0

    polygons_flat = torch.stack(
        [
            x1 - skew_x,
            y1 - skew_y,
            x2 - skew_x,
            y1 + skew_y,
            x2 + skew_x,
            y2 + skew_y,
            x1 + skew_x,
            y2 - skew_y,
        ],
        dim=2,
    )

    batch_size, seq_len, _ = pred.shape
    polygons = polygons_flat.view(batch_size, seq_len, 4, 2)

    polygons[:, :, :, 0] *= w_scale
    polygons[:, :, :, 1] *= h_scale

    return polygons
