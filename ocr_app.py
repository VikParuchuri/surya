import io
from typing import List

import pypdfium2
import streamlit as st
from pypdfium2 import PdfiumError

# 从detection中引入batch_text_detection函数
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
from surya.postprocessing.heatmap import draw_polys_on_image, draw_bboxes_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, TextDetectionResult, LayoutResult, TableResult

# 导入设置
from surya.settings import settings
from surya.tables import batch_table_recognition
from surya.postprocessing.util import rescale_bboxes, rescale_bbox

# 缓存加载检测模型和处理器


@st.cache_resource()
def load_det_cached():
    return load_model(), load_processor()

# 缓存加载识别模型和处理器


@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()

# 缓存加载布局模型和处理器


@st.cache_resource()
def load_layout_cached():
    return load_layout_model(), load_layout_processor()

# 缓存加载表格识别模型和处理器


@st.cache_resource()
def load_table_cached():
    return load_table_model(), load_table_processor()

# 文本检测函数 传入图像 返回检测结果


def text_detection(img) -> (Image.Image, TextDetectionResult):
    # 批量文本检测
    pred = batch_text_detection([img], det_model, det_processor)[0]
    # 获取多边形
    polygons = [p.polygon for p in pred.bboxes]
    # 在图像上绘制多边形
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

# 布局检测函数


def layout_detection(img) -> (Image.Image, LayoutResult):
    # 批量布局检测
    pred = batch_layout_detection([img], layout_model, layout_processor)[0]
    # 获取多边形
    polygons = [p.polygon for p in pred.bboxes]
    # 获取标签
    labels = [f"{p.label}-{p.position}" for p in pred.bboxes]
    # 在图像上绘制多边形和标签
    layout_img = draw_polys_on_image(
        polygons, img.copy(), labels=labels, label_font_size=18)
    return layout_img, pred

# 表格识别函数


def table_recognition(img, highres_img, filepath, page_idx: int, use_pdf_boxes: bool, skip_table_detection: bool) -> (Image.Image, List[TableResult]):
    if skip_table_detection:
        # 如果跳过表格检测，将整个图像视为一个表格
        layout_tables = [(0, 0, highres_img.size[0], highres_img.size[1])]
        table_imgs = [highres_img]
    else:
        # 否则进行布局检测
        _, layout_pred = layout_detection(img)
        layout_tables_lowres = [
            l.bbox for l in layout_pred.bboxes if l.label == "Table"]
        table_imgs = []
        layout_tables = []
        for tb in layout_tables_lowres:
            # 调整边界框
            highres_bbox = rescale_bbox(tb, img.size, highres_img.size)
            table_imgs.append(
                highres_img.crop(highres_bbox)
            )
            layout_tables.append(highres_bbox)

    try:
        # 获取页面文本行
        page_text = get_page_text_lines(
            filepath, [page_idx], [highres_img.size])[0]
        # 获取表格块
        table_bboxes = get_table_blocks(
            layout_tables, page_text, highres_img.size)
    except PdfiumError:
        # 如果获取文本失败，将表格块设为空
        table_bboxes = [[] for _ in layout_tables]

    if not use_pdf_boxes or any(len(tb) == 0 for tb in table_bboxes):
        # 如果不使用PDF边界框或表格块为空，进行文本检测
        det_results = batch_text_detection(
            table_imgs, det_model, det_processor)
        table_bboxes = [[{"bbox": tb.bbox, "text": None}
                         for tb in det_result.bboxes] for det_result in det_results]

    # 批量表格识别
    table_preds = batch_table_recognition(
        table_imgs, table_bboxes, table_model, table_processor)
    table_img = img.copy()

    for results, table_bbox in zip(table_preds, layout_tables):
        adjusted_bboxes = []
        labels = []
        colors = []

        for item in results.rows + results.cols:
            # 调整边界框
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
        # 在图像上绘制边界框
        table_img = draw_bboxes_on_image(
            adjusted_bboxes, highres_img, labels=labels, label_font_size=18, color=colors)
    return table_img, table_preds

# OCR函数


def ocr(img, highres_img, langs: List[str]) -> (Image.Image, OCRResult):
    # 替换语言代码
    replace_lang_with_code(langs)
    # 运行OCR
    img_pred = run_ocr([img], [langs], det_model, det_processor,
                       rec_model, rec_processor, highres_images=[highres_img])[0]

    # 获取边界框和文本
    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    # 在图像上绘制文本
    rec_img = draw_text_on_image(
        bboxes, text, img.size, langs, has_math="_math" in langs)
    return rec_img, img_pred

# 打开PDF文件


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)

# 获取页面图像 使用配置默认的DPI


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    # 打开PDF文件
    doc = open_pdf(pdf_file)
    # 渲染页面
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

# 获取PDF页数


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


# 设置页面配置
st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

# 加载模型和处理器
det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()
layout_model, layout_processor = load_layout_cached()
table_model, table_processor = load_table_cached()

# 页面标题和描述
st.markdown("""
# 计算机视觉在图片文字处理的应用

22软件(智能开发)本四 张明

""")

st.sidebar.markdown("## 图像文字处理")


# 文件上传 st.slidebar 为在布局左侧创建一个侧边栏
in_file = st.sidebar.file_uploader(
    "PDF 文件 或者 图片:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
# 选择语言
languages = st.sidebar.multiselect("语言", sorted(list(CODE_TO_LANGUAGE.values(
))), default=[], max_selections=4, help="选择图片中所述语言(可以提高OCR准确率).  可选.")

# 如果没有上传文件，停止执行
if in_file is None:
    st.stop()

# 获取文件上传后的文件类型
filetype = in_file.type
whole_image = False


# 如果上传的是pdf文件
if "pdf" in filetype:
    # 获取PDF页数
    page_count = page_count(in_file)
    # 选择页码
    page_number = st.sidebar.number_input(
        f"该pdf共 {page_count} 页:", min_value=1, value=1, max_value=page_count)

    # 获取指定页的图像 输入文件、页码、图像分辨率
    pil_image = get_page_image(in_file, page_number, settings.IMAGE_DPI)
    pil_image_highres = get_page_image(
        in_file, page_number, dpi=settings.IMAGE_DPI_HIGHRES)
else:
    # 处理其他类型的图像文件
    pil_image = Image.open(in_file).convert("RGB")
    pil_image_highres = pil_image
    page_number = None

# 侧边栏按钮
text_det = st.sidebar.button("进行图像文本识别")
text_rec = st.sidebar.button("进行光学字符识别(OCR)")
layout_det = st.sidebar.button("进行文字布局分析")
table_rec = st.sidebar.button("进行图表记录识别")
use_pdf_boxes = st.sidebar.checkbox(
    "PDF 表格内容", value=True, help="只能识别表格: 使用 PDF 文件中的边界框与文本检测模型进行对比.")
skip_table_detection = st.sidebar.checkbox(
    "Skip table detection", value=False, help="只能识别表格: 跳过表格检测，将整个图像/页面视为一个表格")

# 如果没有图像，停止执行
if pil_image is None:
    st.stop()

# 执行图像文本识别
if text_det:
    # 执行文本检测
    det_img, pred = text_detection(pil_image)
    with col1:
        st.image(det_img, caption="识别的图像为", use_container_width=True)
        st.json(pred.model_dump(
            exclude=["heatmap", "affinity_map"]), expanded=True)

# 执行文字布局分析
if layout_det:
    layout_img, pred = layout_detection(pil_image)
    with col1:
        st.image(layout_img, caption="识别的布局", use_container_width=True)
        st.json(pred.model_dump(exclude=["segmentation_map"]), expanded=True)

# 执行OCR
if text_rec:

    #
    rec_img, pred = ocr(pil_image, pil_image_highres, languages)

    print('文本识别的效果:', rec_img, "\n".join([p.text for p in pred.text_lines]))
    with col1:
        st.image(rec_img, caption="OCR 结果", use_container_width=True)
        json_tab, text_tab = st.tabs(["JSON格式", "文本格式"])

        # main_cols = st.columns([6, 6])

        # 如果是json格式
        with json_tab:
            st.json(pred.model_dump(), expanded=True)

        # 如果是文本格式 这里可以接入ollama 进行文本分析 给出建议
        with text_tab:
            # st.text("\n".join([p.text for p in pred.text_lines]))
            # for index, p in enumerate(pred.text_lines):
            #     st.text(p.text)
            #     st.button("分析", key=f"analyze_button_{index}")
            for index, p in enumerate(pred.text_lines):
                cols = st.columns([4, 1])  # 创建两列，第一列宽度为4，第二列宽度为1
                with cols[0]:
                    st.text(p.text)  # 在第一列显示文本
                with cols[1]:
                    st.button("分析", key=f"analyze_button_{index}")  # 在第二列显示按钮

# 执行表格识别
if table_rec:
    table_img, pred = table_recognition(pil_image, pil_image_highres, in_file,
                                        page_number - 1 if page_number else None, use_pdf_boxes, skip_table_detection)
    with col1:
        st.image(table_img, caption="图表识别", use_container_width=True)
        st.json([p.model_dump() for p in pred], expanded=True)

# 显示上传的图片
with col2:
    st.image(pil_image, caption="上传的图片", use_container_width=True)
