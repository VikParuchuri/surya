import fitz as pymupdf
from surya.postprocessing.util import rescale_bbox


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