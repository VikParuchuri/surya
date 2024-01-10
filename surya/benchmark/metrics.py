def calculate_intersection(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area


def calculate_coverage(box, other_boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if box_area == 0:
        return 0

    total_coverage = 0

    for other_box in other_boxes:
        intersection_area = calculate_intersection(box, other_box)
        coverage = intersection_area / box_area
        total_coverage += coverage

    total_coverage = min(total_coverage, 1.0)

    return total_coverage


def mean_coverage(preds, references):
    coverages = []
    for box1 in references:
        coverage = calculate_coverage(box1, preds)
        coverages.append(coverage)

    for box2 in preds:
        coverage = calculate_coverage(box2, references)
        coverages.append(coverage)

    # Calculate the average coverage over all comparisons
    if len(coverages) == 0:
        return 0
    return sum(coverages) / len(coverages) * 100