import io
import tempfile
from typing import List

import pypdfium2
import streamlit as st

from surya.models import load_predictors

from surya.debug.draw import draw_polys_on_image, draw_bboxes_on_image

from surya.debug.text import draw_text_on_image
from PIL import Image
from surya.recognition.languages import CODE_TO_LANGUAGE, replace_lang_with_code
from surya.table_rec import TableResult
from surya.detection import TextDetectionResult
from surya.recognition import OCRResult
from surya.layout import LayoutResult
from surya.settings import settings
from surya.common.util import rescale_bbox, expand_bbox


@st.cache_resource()
def load_predictors_cached():
    return load_predictors()


def run_ocr_errors(pdf_file, page_count, sample_len=512, max_samples=10, max_pages=15):
    from pdftext.extraction import plain_text_output
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

    results = predictors["ocr_error"](samples)
    label = "This PDF has good text."
    if results.labels.count("bad") / len(results.labels) > .2:
        label = "This PDF may have garbled or bad OCR text."
    return label, results.labels


def text_detection(img) -> (Image.Image, TextDetectionResult):
    pred = predictors["detection"]([img])[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred


def layout_detection(img) -> (Image.Image, LayoutResult):
    pred = predictors["layout"]([img])[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [f"{p.label}-{p.position}" for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels, label_font_size=18)
    return layout_img, pred


def table_recognition(img, highres_img, skip_table_detection: bool) -> (Image.Image, List[TableResult]):
    if skip_table_detection:
        layout_tables = [(0, 0, highres_img.size[0], highres_img.size[1])]
        table_imgs = [highres_img]
    else:
        _, layout_pred = layout_detection(img)
        layout_tables_lowres = [l.bbox for l in layout_pred.bboxes if l.label in ["Table", "TableOfContents"]]
        table_imgs = []
        layout_tables = []
        for tb in layout_tables_lowres:
            highres_bbox = rescale_bbox(tb, img.size, highres_img.size)
            # Slightly expand the box
            highres_bbox = expand_bbox(highres_bbox)
            table_imgs.append(
                highres_img.crop(highres_bbox)
            )
            layout_tables.append(highres_bbox)

    table_preds = predictors["table_rec"](table_imgs)
    table_img = img.copy()

    for results, table_bbox in zip(table_preds, layout_tables):
        adjusted_bboxes = []
        labels = []
        colors = []

        for item in results.cells:
            adjusted_bboxes.append([
                (item.bbox[0] + table_bbox[0]),
                (item.bbox[1] + table_bbox[1]),
                (item.bbox[2] + table_bbox[0]),
                (item.bbox[3] + table_bbox[1])
            ])
            labels.append(item.label)
            if "Row" in item.label:
                colors.append("blue")
            else:
                colors.append("red")
        table_img = draw_bboxes_on_image(adjusted_bboxes, highres_img, labels=labels, label_font_size=18, color=colors)
    return table_img, table_preds


# Function for OCR
def ocr(img, highres_img, langs: List[str]) -> (Image.Image, OCRResult):
    replace_lang_with_code(langs)
    img_pred = predictors["recognition"]([img], [langs], predictors["detection"], highres_images=[highres_img])[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs)
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

predictors = load_predictors_cached()

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

run_text_det = st.sidebar.button("Run Text Detection")
run_text_rec = st.sidebar.button("Run OCR")
run_layout_det = st.sidebar.button("Run Layout Analysis")
run_table_rec = st.sidebar.button("Run Table Rec")
run_ocr_errors = st.sidebar.button("Run bad PDF text detection")
use_pdf_boxes = st.sidebar.checkbox("PDF table boxes", value=True, help="Table recognition only: Use the bounding boxes from the PDF file vs text detection model.")
skip_table_detection = st.sidebar.checkbox("Skip table detection", value=False, help="Table recognition only: Skip table detection and treat the whole image/page as a table.")

if pil_image is None:
    st.stop()

# Run Text Detection
if run_text_det:
    det_img, pred = text_detection(pil_image)
    with col1:
        st.image(det_img, caption="Detected Text", use_container_width=True)
        st.json(pred.model_dump(exclude=["heatmap", "affinity_map"]), expanded=True)


# Run layout
if run_layout_det:
    layout_img, pred = layout_detection(pil_image)
    with col1:
        st.image(layout_img, caption="Detected Layout", use_container_width=True)
        st.json(pred.model_dump(exclude=["segmentation_map"]), expanded=True)

# Run OCR
if run_text_rec:
    rec_img, pred = ocr(pil_image, pil_image_highres, languages)
    with col1:
        st.image(rec_img, caption="OCR Result", use_container_width=True)
        json_tab, text_tab = st.tabs(["JSON", "Text Lines (for debugging)"])
        with json_tab:
            st.json(pred.model_dump(), expanded=True)
        with text_tab:
            st.text("\n".join([p.text for p in pred.text_lines]))


if run_table_rec:
    table_img, pred = table_recognition(pil_image, pil_image_highres, skip_table_detection)
    with col1:
        st.image(table_img, caption="Table Recognition", use_container_width=True)
        st.json([p.model_dump() for p in pred], expanded=True)

if run_ocr_errors:
    if "pdf" not in filetype:
        st.error("This feature only works with PDFs.")
    label, results = run_ocr_errors(in_file, page_count)
    with col1:
        st.write(label)
        st.json(results)

with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)