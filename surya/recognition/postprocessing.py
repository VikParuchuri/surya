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

    return text[:len(text_to_truncate)]

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
def ensure_matching_tags(text_chars: List[TextChar], special_tokens: Dict[str, list]) -> bool:
    self_closing_tags = ["br"]

    open_tags = []

    format_tags = extract_tags(special_tokens["formatting"]) + extract_tags(special_tokens["math_external"])
    math_tags = extract_tags(special_tokens["math_internal"])

    for char in text_chars:
        if len(char.text) <= 1:
            continue

        tag_match = re.match(tag_pattern, char.text)
        if not tag_match:
            continue

        is_closing = tag_match.group(1) == "/"
        tag_name = tag_match.group(2).lower()

        if tag_name not in format_tags + math_tags:
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
                return False
        else:
            open_tags.append(tag_name)

    if not open_tags:
        return True

    return False

def replace_invalid_tags(text_chars: List[TextChar], special_tokens: Dict[str, list]):
    bad_tags = not ensure_matching_tags(text_chars, special_tokens)
    if bad_tags:
        for char in text_chars:
            if re.match(tag_pattern, char.text):
                char.text = ""
                char.confidence = 0
    return text_chars




