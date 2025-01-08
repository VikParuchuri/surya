import fitz as pymupdf
from surya.common.util import rescale_bbox


def get_pdf_lines(pdf_path, img_sizes):
    doc = pymupdf.open(pdf_path)
    page_lines = []
    for idx, img_size in enumerate(img_sizes):
        page = doc[idx]
        blocks = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]

        line_boxes = []
        for block_idx, block in enumerate(blocks):
            for l in block["lines"]:
                line_boxes.append(list(l["bbox"]))

        page_box = page.bound()
        pwidth, pheight = page_box[2] - page_box[0], page_box[3] - page_box[1]
        line_boxes = [rescale_bbox(bbox, (pwidth, pheight), img_size) for bbox in line_boxes]
        page_lines.append(line_boxes)

    return page_lines

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
