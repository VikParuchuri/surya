from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from surya.schema import PolygonBox, TextLine, TableResult, Bbox, TableCell
from surya.settings import settings


def get_batch_size():
    batch_size = settings.ORDER_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 32
    return batch_size


def rescale_boxes(pred_bboxes, image_size):
    cx, cy, w, h = pred_bboxes.unbind(-1)
    img_h, img_w = image_size
    x0 = (cx - 0.5 * w) * img_w
    x1 = (cx + 0.5 * w) * img_w
    y0 = (cy - 0.5 * h) * img_h
    y1 = (cy + 0.5 * h) * img_h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def sort_table_blocks(cells, tolerance=5) -> list:
    vertical_groups = {}
    for idx, cell in enumerate(cells):
        cell.cell_id = idx # Save id before sorting
        bbox = cell.bbox
        group_key = round((bbox[1] + bbox[3]) / 2 / tolerance)
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(cell)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_rows = []
    for idx, (_, group) in enumerate(sorted(vertical_groups.items())):
        sorted_group = sorted(group, key=lambda x: x.bbox[0]) # sort by x within each row
        for cell in sorted_group:
            cell.row_id = idx
            # TODO: if too few cells in row, merge with row above
        sorted_rows.append(sorted_group)

    return sorted_rows


def post_process(results, img_size, id2label):
    m = results.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())
    pred_scores = list(m.values.detach().cpu().numpy())
    pred_bboxes = results.pred_boxes.detach().cpu()
    batch_columns = []
    for pred_label, pred_score, pred_bbox in zip(pred_labels, pred_scores, pred_bboxes):
        pred_bbox = [elem.tolist() for elem in rescale_boxes(pred_bbox, img_size)]

        columns = []
        for label, score, bbox in zip(pred_label, pred_score, pred_bbox):
            class_label = id2label.get(int(label), "unknown")
            score = float(score)
            if class_label == "table column" and score > settings.TABLE_REC_MIN_SCORE:
                columns.append(Bbox(bbox=[float(elem) for elem in bbox]))
        columns = sorted(columns, key=lambda x: x.bbox[0])
        batch_columns.append(columns)
    return batch_columns


def batch_table_recognition(images: List, cells: List[List[PolygonBox]], model, processor, text_lines: Optional[List[List[TextLine]]] = None, batch_size: Optional[int] = None, min_text_assign_score=.2) -> List[TableResult]:
    assert all([isinstance(image, Image.Image) for image in images])
    if batch_size is None:
        batch_size = get_batch_size()

    all_results = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing tables"):
        batch_images = images[i:i + batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images
        image_bboxes = [[0, 0, img.size[0], img.size[1]] for img in batch_images]
        batch_cells = cells[i:i + batch_size]
        batch_text_lines = text_lines[i:i + batch_size] if text_lines is not None else None

        pixel_values = processor(batch_images)
        pixel_values = pixel_values.to(model.device).to(model.dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        batch_columns = post_process(outputs, img_size=(settings.RECOGNITION_IMAGE_SIZE["height"], settings.RECOGNITION_IMAGE_SIZE["width"]), id2label=model.config.id2label)

        # Assign cells to columns
        results = []
        for columns, cells, image_bbox in zip(batch_columns, batch_cells, image_bboxes):
            rows = sort_table_blocks(cells)
            result = []
            for idx, row in enumerate(rows):
                for cell in row:
                    cell.col_id = -1
                    for col_idx, column in enumerate(columns):
                        if column.bbox[0] <= cell.bbox[0]:
                            cell.col_id = col_idx
                    result.append(TableCell(
                        row_id=cell.row_id,
                        cell_id=cell.cell_id,
                        text="",
                        col_id=cell.col_id,
                        polygon=cell.polygon
                    ))
            results.append(TableResult(cells=result, image_bbox=image_bbox))

        if batch_text_lines is not None:
            # Assign text to cells
            for text_line, result in zip(batch_text_lines, results):
                for text in text_line:
                    cell_assignment = None
                    max_intersect = None
                    for cell_idx, cell in result.cells:
                        if max_intersect is None or text.intersection_pct(cell) > max_intersect:
                            max_intersect = text.intersection_pct(cell)
                            cell_assignment = cell_idx
                    if max_intersect > min_text_assign_score:
                        result.cells[cell_assignment].text += text.text + " "

            all_results.extend(results)
    return all_results



