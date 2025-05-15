from typing import List

def detect_repeat_token(predicted_tokens: List[int], max_repeats: int = 40):
    if len(predicted_tokens) < max_repeats:
        return False

    # Detect repeats containing 1 or 2 tokens
    last_n = predicted_tokens[-max_repeats:]
    unique_tokens = len(set(last_n))
    if unique_tokens > 5:
        return False

    return last_n[-unique_tokens:] == last_n[-unique_tokens * 2 : -unique_tokens]