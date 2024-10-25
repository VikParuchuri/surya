from pdftext.extraction import dictionary_output

from surya.postprocessing.text import sort_text_lines
from surya.schema import PolygonBox
import numpy as np

from surya.settings import settings


def get_page_text_lines(filepath: str, page_idxs: list, out_sizes: list, flatten_pdf: bool = settings.FLATTEN_PDF) -> list:
    assert len(page_idxs) == len(out_sizes)
    pages_text = dictionary_output(filepath, sort=False, page_range=page_idxs, keep_chars=True, flatten_pdf=flatten_pdf)
    for full_text, out_size in zip(pages_text, out_sizes):
        width = full_text["width"]
        height = full_text["height"]
        text_w_scale = out_size[0] / width
        text_h_scale = out_size[1] / height
        for block in full_text["blocks"]:
            for line in block["lines"]:
                line["bbox"] = [line["bbox"][0] * text_w_scale, line["bbox"][1] * text_h_scale,
                                line["bbox"][2] * text_w_scale, line["bbox"][3] * text_h_scale]
                for span in line["spans"]:
                    for char in span["chars"]:
                        char["bbox"] = [char["bbox"][0] * text_w_scale, char["bbox"][1] * text_h_scale,
                                        char["bbox"][2] * text_w_scale, char["bbox"][3] * text_h_scale]
    return pages_text


def get_dynamic_gap_thresh(full_text: dict, img_size: list, default_thresh=.01, min_chars=100):
    space_dists = []
    for block in full_text["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                for i in range(1, len(span["chars"])):
                    char1 = span["chars"][i - 1]
                    char2 = span["chars"][i]
                    if full_text["rotation"] == 90:
                        space_dists.append((char2["bbox"][0] - char1["bbox"][2]) / img_size[0])
                    elif full_text["rotation"] == 180:
                        space_dists.append((char2["bbox"][1] - char1["bbox"][3]) / img_size[1])
                    elif full_text["rotation"] == 270:
                        space_dists.append((char1["bbox"][0] - char2["bbox"][2]) / img_size[0])
                    else:
                        space_dists.append((char1["bbox"][1] - char2["bbox"][3]) / img_size[1])
    cell_gap_thresh = np.percentile(space_dists, 80) if len(space_dists) > min_chars else default_thresh
    return cell_gap_thresh


def is_same_span(char, curr_box, img_size, space_thresh, rotation):
    def normalized_diff(a, b, dimension, mult=1, use_abs=True):
        func = abs if use_abs else lambda x: x
        return func(a - b) / img_size[dimension] < space_thresh * mult

    bbox = char["bbox"]
    if rotation == 90:
        return all([
            normalized_diff(bbox[0], curr_box[0], 0, use_abs=False),
            normalized_diff(bbox[1], curr_box[3], 1),
            normalized_diff(bbox[0], curr_box[0], 0, mult=5)
        ])
    elif rotation == 180:
        return all([
            normalized_diff(bbox[2], curr_box[0], 0, use_abs=False),
            normalized_diff(bbox[1], curr_box[1], 1),
            normalized_diff(bbox[2], curr_box[0], 1, mult=5)
        ])
    elif rotation == 270:
        return all([
            normalized_diff(bbox[0], curr_box[0], 0, use_abs=False),
            normalized_diff(bbox[3], curr_box[1], 1),
            normalized_diff(bbox[0], curr_box[0], 1, mult=5)
        ])
    else:  # 0 or default case
        return all([
            normalized_diff(bbox[0], curr_box[2], 0, use_abs=False),
            normalized_diff(bbox[1], curr_box[1], 1),
            normalized_diff(bbox[0], curr_box[2], 1, mult=5)
        ])


def get_table_blocks(tables: list, full_text: dict, img_size: list, table_thresh=.8, space_thresh=.01):
    # Returns coordinates relative to input table, not full image
    table_texts = []
    space_thresh = max(space_thresh, get_dynamic_gap_thresh(full_text, img_size, default_thresh=space_thresh))
    for table in tables:
        table_poly = PolygonBox(polygon=[
            [table[0], table[1]],
            [table[2], table[1]],
            [table[2], table[3]],
            [table[0], table[3]]
        ])
        table_text = []
        rotation = full_text["rotation"]
        for block in full_text["blocks"]:
            for line in block["lines"]:
                line_poly = PolygonBox(polygon=[
                    [line["bbox"][0], line["bbox"][1]],
                    [line["bbox"][2], line["bbox"][1]],
                    [line["bbox"][2], line["bbox"][3]],
                    [line["bbox"][0], line["bbox"][3]]
                ])
                if line_poly.intersection_pct(table_poly) < table_thresh:
                    continue
                curr_span = None
                curr_box = None
                for span in line["spans"]:
                    for char in span["chars"]:
                        same_span = False
                        if curr_span:
                            same_span = is_same_span(char, curr_box, img_size, space_thresh, rotation)

                        if curr_span is None:
                            curr_span = char["char"]
                            curr_box = char["bbox"]
                        elif same_span:
                            curr_span += char["char"]
                            curr_box = [min(curr_box[0], char["bbox"][0]), min(curr_box[1], char["bbox"][1]),
                                        max(curr_box[2], char["bbox"][2]), max(curr_box[3], char["bbox"][3])]
                        else:
                            if curr_span.strip():
                                table_text.append({"text": curr_span, "bbox": curr_box})
                            curr_span = char["char"]
                            curr_box = char["bbox"]
                if curr_span is not None and curr_span.strip():
                    table_text.append({"text": curr_span, "bbox": curr_box})
        # Adjust to be relative to input table
        for item in table_text:
            item["bbox"] = [
                item["bbox"][0] - table[0],
                item["bbox"][1] - table[1],
                item["bbox"][2] - table[0],
                item["bbox"][3] - table[1]
            ]
        table_text = sort_text_lines(table_text)
        table_texts.append(table_text)
    return table_texts