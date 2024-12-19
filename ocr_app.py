import io
import tempfile
from typing import List

import pypdfium2
import streamlit as st
from pypdfium2 import PdfiumError

from surya.detection import batch_text_detection
from surya.input.pdflines import get_page_text_lines, get_table_blocks
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.table_rec.model import load_model as load_table_model
from surya.model.table_rec.processor import load_processor as load_table_processor
from surya.model.ocr_error.model import load_model as load_ocr_error_model, load_tokenizer as load_ocr_error_processor
from surya.postprocessing.heatmap import draw_polys_on_image, draw_bboxes_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, TextDetectionResult, LayoutResult, TableResult
from surya.settings import settings
from surya.tables import batch_table_recognition
from surya.postprocessing.util import rescale_bbox
from pdftext.extraction import plain_text_output
from surya.ocr_error import batch_ocr_error_detection


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
def load_table_cached():
    return load_table_model(), load_table_processor()

@st.cache_resource()
def load_ocr_error_cached():
    return load_ocr_error_model(), load_ocr_error_processor()


def run_ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        f.seek(0)

        # Sample the text from the middle of the PDF
        page_middle = page_count // 2
        page_range = range(max(page_middle - max_pages, 0), min(page_middle + max_pages, page_count))
        text = plain_text_output(f.name, page_range=page_range)

    sample_gap = len(text) // max_samples
    if len(text) == 0 or sample_gap == 0:
        return "This PDF has no text or very little text", ["no text"]

    if sample_gap < sample_len:
        sample_gap = sample_len

    # Split the text into samples for the model
    samples = []
    for i in range(0, len(text), sample_gap):
        samples.append(text[i:i + sample_len])

    results = batch_ocr_error_detection(samples, ocr_error_model, ocr_error_processor)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > .2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels


def text_detection(img) -> (Image.Image, TextDetectionResult):
    pred = batch_text_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred


def layout_detection(img) -> (Image.Image, LayoutResult):
    pred = batch_layout_detection([img], layout_model, layout_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [f"{p.label}-{p.position}" for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels, label_font_size=18)
    return layout_img, pred


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
    doc.close()
    return png_image


@st.cache_data()
def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()
layout_model, layout_processor = load_layout_cached()
table_model, table_processor = load_table_cached()
ocr_error_model, ocr_error_processor = load_ocr_error_cached()


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
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
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
table_rec = st.sidebar.button("Run Table Rec")
ocr_errors = st.sidebar.button("Run bad PDF text detection")
use_pdf_boxes = st.sidebar.checkbox("PDF table boxes", value=True, help="Table recognition only: Use the bounding boxes from the PDF file vs text detection model.")
skip_table_detection = st.sidebar.checkbox("Skip table detection", value=False, help="Table recognition only: Skip table detection and treat the whole image/page as a table.")

if pil_image is None:
    st.stop()

# Run Text Detection
if text_det:
    det_img, pred = text_detection(pil_image)
    with col1:
        st.image(det_img, caption="Detected Text", use_container_width=True)
        st.json(pred.model_dump(exclude=["heatmap", "affinity_map"]), expanded=True)


# Run layout
if layout_det:
    layout_img, pred = layout_detection(pil_image)
    with col1:
        st.image(layout_img, caption="Detected Layout", use_container_width=True)
        st.json(pred.model_dump(exclude=["segmentation_map"]), expanded=True)

# Run OCR
if text_rec:
    rec_img, pred = ocr(pil_image, pil_image_highres, languages)
    with col1:
        st.image(rec_img, caption="OCR Result", use_container_width=True)
        json_tab, text_tab = st.tabs(["JSON", "Text Lines (for debugging)"])
        with json_tab:
            st.json(pred.model_dump(), expanded=True)
        with text_tab:
            st.text("\n".join([p.text for p in pred.text_lines]))


if table_rec:
    table_img, pred = table_recognition(pil_image, pil_image_highres, in_file, page_number - 1 if page_number else None, use_pdf_boxes, skip_table_detection)
    with col1:
        st.image(table_img, caption="Table Recognition", use_container_width=True)
        st.json([p.model_dump() for p in pred], expanded=True)

if ocr_errors:
    if "pdf" not in filetype:
        st.error("This feature only works with PDFs.")
    label, results = run_ocr_errors(in_file, page_count)
    with col1:
        st.write(label)
        st.json(results)

with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)