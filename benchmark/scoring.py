import math
from typing import List

from rapidfuzz import fuzz


def overlap_score(pred_lines: List[str], reference_lines: List[str]):
    line_scores = []
    line_weights = []
    for i, pred_line in enumerate(pred_lines):
        max_score = 0
        line_weight = 1
        for j, ref_line in enumerate(reference_lines):
            score = fuzz.ratio(pred_line, ref_line, score_cutoff=20) / 100
            if score > max_score:
                max_score = score
                line_weight = math.sqrt(len(ref_line))
        line_scores.append(max_score)
        line_weights.append(line_weight)
    line_scores = [line_scores[i] * line_weights[i] for i in range(len(line_scores))]

    return sum(line_scores) / sum(line_weights)