from pdftext.extraction import dictionary_output

from surya.postprocessing.text import sort_text_lines
from surya.schema import PolygonBox


def get_page_text_lines(filepath: str, page_idxs: list, out_sizes: list) -> list:
    assert len(page_idxs) == len(out_sizes)
    pages_text = dictionary_output(filepath, sort=False, page_range=page_idxs, keep_chars=True)
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


def get_table_blocks(tables: list, full_text: dict, img_size: list, table_thresh=.8):
    # Returns coordinates relative to input table, not full image
    table_texts = []
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
                            if rotation == 90:
                                same_span = (char["bbox"][0] - curr_box[0]) / img_size[0] < 0.01 and abs(char["bbox"][1] - curr_box[3]) / img_size[1] < 0.01
                            elif rotation == 180:
                                same_span = (char["bbox"][2] - curr_box[0]) / img_size[0] < 0.01 and (char["bbox"][1] - curr_box[1]) / img_size[1] < 0.01
                            elif rotation == 270:
                                same_span = (char["bbox"][0] - curr_box[0]) / img_size[0] < 0.01 and abs(char["bbox"][3] - curr_box[1]) / img_size[1] < 0.01
                            else:
                                same_span = (char["bbox"][0] - curr_box[2]) / img_size[0] < 0.01 and (char["bbox"][1] - curr_box[1]) / img_size[1] < 0.01

                        if curr_span is None:
                            curr_span = char["char"]
                            curr_box = char["bbox"]
                        elif same_span:
                            curr_span += char["char"]
                            curr_box = [min(curr_box[0], char["bbox"][0]), min(curr_box[1], char["bbox"][1]),
                                        max(curr_box[2], char["bbox"][2]), max(curr_box[3], char["bbox"][3])]
                        else:
                            table_text.append({"text": curr_span, "bbox": curr_box})
                            curr_span = char["char"]
                            curr_box = char["bbox"]
                if curr_span is not None:
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