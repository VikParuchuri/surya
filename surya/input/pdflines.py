from pdftext.extraction import dictionary_output

from surya.postprocessing.text import sort_text_lines
from surya.schema import PolygonBox


def get_page_text_lines(filepath, page_idx, out_size):
    full_text = dictionary_output(filepath, sort=False, page_range=[page_idx], keep_chars=True)[0]
    text_bbox = full_text["bbox"]
    text_w_scale = out_size[0] / text_bbox[2]
    text_h_scale = out_size[1] / text_bbox[3]
    for block in full_text["blocks"]:
        for line in block["lines"]:
            line["bbox"] = [line["bbox"][0] * text_w_scale, line["bbox"][1] * text_h_scale,
                            line["bbox"][2] * text_w_scale, line["bbox"][3] * text_h_scale]
            for span in line["spans"]:
                for char in span["chars"]:
                    char["bbox"] = [char["bbox"][0] * text_w_scale, char["bbox"][1] * text_h_scale,
                                    char["bbox"][2] * text_w_scale, char["bbox"][3] * text_h_scale]
    return full_text


def get_table_blocks(tables, full_text, img_size):
    # Returns coordinates relative to input table, not full image
    table_texts = []
    for table in tables:
        table_text = []
        for block in full_text["blocks"]:
            for line in block["lines"]:
                line_poly = PolygonBox(polygon=[
                    [line["bbox"][0], line["bbox"][1]],
                    [line["bbox"][2], line["bbox"][1]],
                    [line["bbox"][2], line["bbox"][3]],
                    [line["bbox"][0], line["bbox"][3]]
                ])
                if line_poly.intersection_pct(table) < 0.8:
                    continue
                curr_span = None
                curr_box = None
                for span in line["spans"]:
                    for char in span["chars"]:
                        if curr_span is None:
                            curr_span = char["char"]
                            curr_box = char["bbox"]
                        elif (char["bbox"][0] - curr_box[2]) / img_size[0] < 0.01:
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
                item["bbox"][0] - table.bbox[0],
                item["bbox"][1] - table.bbox[1],
                item["bbox"][2] - table.bbox[0],
                item["bbox"][3] - table.bbox[1]
            ]
        table_text = sort_text_lines(table_text)
        table_texts.append(table_text)
    return table_texts