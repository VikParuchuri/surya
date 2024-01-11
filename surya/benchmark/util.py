def merge_boxes(box1, box2):
    return (min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3]))


def join_lines(bboxes, max_gap=5):
    to_merge = {}
    for i, box1 in bboxes:
        for z, box2 in bboxes[i + 1:]:
            j = i + z + 1
            if box1 == box2:
                continue

            if box1[0] <= box2[0] and box1[2] >= box2[2]:
                if abs(box1[1] - box2[3]) <= max_gap:
                    if i not in to_merge:
                        to_merge[i] = []
                    to_merge[i].append(j)

    merged_boxes = set()
    merged = []
    for i, box in bboxes:
        if i in merged_boxes:
            continue

        if i in to_merge:
            for j in to_merge[i]:
                box = merge_boxes(box, bboxes[j][1])
                merged_boxes.add(j)

        merged.append(box)
    return merged
