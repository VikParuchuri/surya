import math
from typing import List

from rapidfuzz import fuzz


def overlap_score(pred_lines: List[str], reference_lines: List[str]):
    line_scores = []
    line_weights = []
    line_match = {}
    for i, pred_line in enumerate(pred_lines):
        max_score = 0
        line_weight = 1
        match = None
        for j, ref_line in enumerate(reference_lines):
            score = fuzz.ratio(pred_line, ref_line, score_cutoff=20) / 100
            if score > max_score:
                max_score = score
                line_weight = math.sqrt(len(ref_line))
                match = j
        line_scores.append(max_score)
        line_weights.append(line_weight)
        line_match[i] = match
    line_scores = [line_scores[i] * line_weights[i] for i in range(len(line_scores))]

    return line_scores, line_weights, line_match


def overlap_score_exact(pred_lines: List[str], reference_lines: List[str]):
    line_scores = []
    line_weights = []
    assert len(pred_lines) == len(reference_lines)

    for i, (pred_line, ref_line) in enumerate(zip(pred_lines, reference_lines)):
        score = fuzz.ratio(pred_line, ref_line, score_cutoff=20) / 100
        weight = math.sqrt(len(ref_line))
        line_scores.append(score * weight)
        line_weights.append(weight)

    return line_scores, line_weights
