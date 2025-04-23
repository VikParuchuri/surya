import re
from typing import List, Dict

from surya.recognition.schema import TextChar


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

    return text[: len(text_to_truncate)]


def extract_tags(proposed_tags: List[str]) -> List[str]:
    tags = []
    for tag in proposed_tags:
        tag_match = re.match(tag_pattern, tag)
        if not tag_match:
            continue

        if not tag_match.group(1) == "/":
            continue

        tags.append(tag_match.group(2))
    return tags


tag_pattern = re.compile(r"<(/?)([a-z]+)([^>]*)>?", re.IGNORECASE)


def cleanup_math(line: str):
    matches = re.finditer(r"(<math[^>]*>)(.*?)</math>", line, re.DOTALL)
    result = line

    for match in matches:
        opening_tag = match.group(1)  # The opening <math> tag with attributes
        full_match = match.group(0)  # The entire <math>content</math> tag
        block_content = match.group(2)  # Just the content inside the tags

        clean_block = re.sub(r"<[^>]+>", "", block_content)

        if not re.search(r"[\\\_]", clean_block):
            result = result.replace(full_match, clean_block)
        else:
            result = result.replace(full_match, f"{opening_tag}{clean_block}</math>")

    return result


def fix_unbalanced_tags(
    text_chars: List[TextChar], special_tokens: Dict[str, list]
) -> List[TextChar]:
    self_closing_tags = ["br"]

    open_tags = []

    format_tags = extract_tags(special_tokens["formatting"]) + extract_tags(
        special_tokens["math_external"]
    )

    for char in text_chars:
        if len(char.text) <= 1:
            continue

        tag_match = re.match(tag_pattern, char.text)
        if not tag_match:
            continue

        is_closing = tag_match.group(1) == "/"
        tag_name = tag_match.group(2).lower()

        if tag_name not in format_tags:
            continue

        if tag_name in self_closing_tags:
            continue

        # Self-closing tags
        if tag_match.group(3) and tag_match.group(3).strip().endswith("/"):
            continue

        if is_closing:
            if open_tags and open_tags[-1] == tag_name:
                open_tags.pop()
        else:
            open_tags.append(tag_name)

    for tag in open_tags:
        text_chars.append(
            TextChar(
                text=f"</{tag}>",
                confidence=0,
                polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
                bbox_valid=False,
            )
        )
    return text_chars
