from typing import List

from surya.recognition.schema import TextLine, TextWord, TextChar


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


def words_from_chars(chars: List[TextChar]):
    words = []
    word = None
    for char in chars:
        if not char.bbox_valid:
            if word:
                words.append(word)
                word = None
            continue

        if not word:
            word = TextWord(**char.model_dump())
        elif not char.text.strip():
            if word:
                words.append(word)
            word = None
        else:
            # Merge bboxes
            word.merge(char)
            word.text = word.text + char.text
    if word:
        words.append(word)

    return words
