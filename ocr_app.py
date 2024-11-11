import io
from typing import List

import pypdfium2
import streamlit as st
from pypdfium2 import PdfiumError

from surya.detection import batch_text_detection
from surya.input.pdflines import get_page_text_lines, get_table_blocks
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.model.layout.model import load_model as load_layout_model, load_processor as load_layout_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.model.table_rec.model import load_model as load_table_model
from surya.model.table_rec.processor import load_processor as load_table_processor
from surya.ordering import batch_ordering
from surya.postprocessing.heatmap import draw_polys_on_image, draw_bboxes_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, TextDetectionResult, LayoutResult, OrderResult, TableResult
from surya.settings import settings
from surya.tables import batch_table_recognition
from surya.postprocessing.util import rescale_bboxes, rescale_bbox


@st.cache_resource()
def load_det_cached():
    return load_model(), load_processor()


@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()


@st.cache_resource()
def load_layout_cached():
    return load_layout_model(), load_layout_processor()

@st.cache_resource()
def load_order_cached():
    return load_order_model(), load_order_processor()


@st.cache_resource()
def load_table_cached():
    return load_table_model(), load_table_processor()


def text_detection(img) -> (Image.Image, TextDetectionResult):
    pred = batch_text_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred


def layout_detection(img) -> (Image.Image, LayoutResult):
    _, det_pred = text_detection(img)
    pred = batch_layout_detection([img], layout_model, layout_processor, [det_pred])[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [p.label for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels, label_font_size=18)
    return layout_img, pred


def order_detection(img) -> (Image.Image, OrderResult):
    _, layout_pred = layout_detection(img)
    bboxes = [l.bbox for l in layout_pred.bboxes]
    pred = batch_ordering([img], [bboxes], order_model, order_processor)[0]
    polys = [l.polygon for l in pred.bboxes]
    positions = [str(l.position) for l in pred.bboxes]
    order_img = draw_polys_on_image(polys, img.copy(), labels=positions, label_font_size=18)
    return order_img, pred


def table_recognition(img, highres_img, filepath, page_idx: int, use_pdf_boxes: bool, skip_table_detection: bool) -> (Image.Image, List[TableResult]):
    if skip_table_detection:
        layout_tables = [(0, 0, highres_img.size[0], highres_img.size[1])]
        table_imgs = [highres_img]
    else:
        _, layout_pred = layout_detection(img)
        layout_tables_lowres = [l.bbox for l in layout_pred.bboxes if l.label == "Table"]
        table_imgs = []
        layout_tables = []
        for tb in layout_tables_lowres:
            highres_bbox = rescale_bbox(tb, img.size, highres_img.size)
            table_imgs.append(
                highres_img.crop(highres_bbox)
            )
            layout_tables.append(highres_bbox)

    try:
        page_text = get_page_text_lines(filepath, [page_idx], [highres_img.size])[0]
        table_bboxes = get_table_blocks(layout_tables, page_text, highres_img.size)
    except PdfiumError:
        # This happens when we try to get text from an image
        table_bboxes = [[] for _ in layout_tables]

    if not use_pdf_boxes or any(len(tb) == 0 for tb in table_bboxes):
        det_results = batch_text_detection(table_imgs, det_model, det_processor)
        table_bboxes = [[{"bbox": tb.bbox, "text": None} for tb in det_result.bboxes] for det_result in det_results]

    table_preds = batch_table_recognition(table_imgs, table_bboxes, table_model, table_processor)
    table_img = img.copy()

    for results, table_bbox in zip(table_preds, layout_tables):
        adjusted_bboxes = []
        labels = []
        colors = []

        for item in results.rows + results.cols:
            adjusted_bboxes.append([
                (item.bbox[0] + table_bbox[0]),
                (item.bbox[1] + table_bbox[1]),
                (item.bbox[2] + table_bbox[0]),
                (item.bbox[3] + table_bbox[1])
            ])
            labels.append(item.label)
            if hasattr(item, "row_id"):
                colors.append("blue")
            else:
                colors.append("red")
        table_img = draw_bboxes_on_image(adjusted_bboxes, highres_img, labels=labels, label_font_size=18, color=colors)
    return table_img, table_preds


# Function for OCR
def ocr(img, highres_img, langs: List[str]) -> (Image.Image, OCRResult):
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor, highres_images=[highres_img])[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs, has_math="_math" in langs)
    return rec_img, img_pred


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()
layout_model, layout_processor = load_layout_cached()
order_model, order_processor = load_order_cached()
table_model, table_processor = load_table_cached()


st.markdown("""
# Surya OCR Demo

This app will let you try surya, a multilingual OCR model. It supports text detection + layout analysis in any language, and text recognition in 90+ languages.

Notes:
- This works best on documents with printed text.
- Preprocessing the image (e.g. increasing contrast) can improve results.
- If OCR doesn't work, try changing the resolution of your image (increase if below 2048px width, otherwise decrease).
- This supports 90+ languages, see [here](https://github.com/VikParuchuri/surya/tree/master/surya/languages.py) for a full list.

Find the project [here](https://github.com/VikParuchuri/surya).
""")

in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
languages = st.sidebar.multiselect("Languages", sorted(list(CODE_TO_LANGUAGE.values())), default=[], max_selections=4, help="Select the languages in the image (if known) to improve OCR accuracy.  Optional.")

if in_file is None:
    st.stop()

filetype = in_file.type
whole_image = False
if "pdf" in filetype:
    page_count = page_count(in_file)
    page_number = st.sidebar.number_input(f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count)

    pil_image = get_page_image(in_file, page_number, settings.IMAGE_DPI)
    pil_image_highres = get_page_image(in_file, page_number, dpi=settings.IMAGE_DPI_HIGHRES)
else:
    pil_image = Image.open(in_file).convert("RGB")
    pil_image_highres = pil_image
    page_number = None

text_det = st.sidebar.button("Run Text Detection")
text_rec = st.sidebar.button("Run OCR")
layout_det = st.sidebar.button("Run Layout Analysis")
order_det = st.sidebar.button("Run Reading Order")
table_rec = st.sidebar.button("Run Table Rec")
use_pdf_boxes = st.sidebar.checkbox("PDF table boxes", value=True, help="Table recognition only: Use the bounding boxes from the PDF file vs text detection model.")
skip_table_detection = st.sidebar.checkbox("Skip table detection", value=False, help="Table recognition only: Skip table detection and treat the whole image/page as a table.")

if pil_image is None:
    st.stop()

# Run Text Detection
if text_det:
    det_img, pred = text_detection(pil_image)
    with col1:
        st.image(det_img, caption="Detected Text", use_column_width=True)
        st.json(pred.model_dump(exclude=["heatmap", "affinity_map"]), expanded=True)


# Run layout
if layout_det:
    layout_img, pred = layout_detection(pil_image)
    with col1:
        st.image(layout_img, caption="Detected Layout", use_column_width=True)
        st.json(pred.model_dump(exclude=["segmentation_map"]), expanded=True)

# Run OCR
if text_rec:
    rec_img, pred = ocr(pil_image, pil_image_highres, languages)
    with col1:
        st.image(rec_img, caption="OCR Result", use_column_width=True)
        json_tab, text_tab = st.tabs(["JSON", "Text Lines (for debugging)"])
        with json_tab:
            st.json(pred.model_dump(), expanded=True)
        with text_tab:
            st.text("\n".join([p.text for p in pred.text_lines]))

if order_det:
    order_img, pred = order_detection(pil_image)
    with col1:
        st.image(order_img, caption="Reading Order", use_column_width=True)
        st.json(pred.model_dump(), expanded=True)


if table_rec:
    table_img, pred = table_recognition(pil_image, pil_image_highres, in_file, page_number - 1 if page_number else None, use_pdf_boxes, skip_table_detection)
    with col1:
        st.image(table_img, caption="Table Recognition", use_column_width=True)
        st.json([p.model_dump() for p in pred], expanded=True)

with col2:
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)