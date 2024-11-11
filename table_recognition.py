import pypdfium2 as pdfium # Needs to be on top to avoid warning
import os
import argparse
import copy
import json
from collections import defaultdict

from surya.detection import batch_text_detection
from surya.input.load import load_from_folder, load_from_file
from surya.input.pdflines import get_table_blocks
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.layout.model import load_model as load_layout_model, load_processor as load_layout_processor
from surya.model.table_rec.model import load_model as load_model
from surya.model.table_rec.processor import load_processor
from surya.tables import batch_table_recognition
from surya.postprocessing.heatmap import draw_bboxes_on_image
from surya.settings import settings
from surya.postprocessing.util import rescale_bboxes, rescale_bbox


def main():
    parser = argparse.ArgumentParser(description="Find reading order of an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file or folder to find reading order in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with layout results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--images", action="store_true", help="Save images of detected layout bboxes.", default=False)
    parser.add_argument("--detect_boxes", action="store_true", help="Detect table boxes.", default=False)
    parser.add_argument("--skip_table_detection", action="store_true", help="Tables are already cropped, so don't re-detect tables.", default=False)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    layout_model = load_layout_model()
    layout_processor = load_layout_processor()

    det_model = load_det_model()
    det_processor = load_det_processor()

    if os.path.isdir(args.input_path):
        images, _, _ = load_from_folder(args.input_path, args.max)
        highres_images, names, text_lines = load_from_folder(args.input_path, args.max, dpi=settings.IMAGE_DPI_HIGHRES, load_text_lines=True)
        folder_name = os.path.basename(args.input_path)
    else:
        images, _, _ = load_from_file(args.input_path, args.max)
        highres_images, names, text_lines = load_from_file(args.input_path, args.max, dpi=settings.IMAGE_DPI_HIGHRES, load_text_lines=True)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    pnums = []
    prev_name = None
    for i, name in enumerate(names):
        if prev_name is None or prev_name != name:
            pnums.append(0)
        else:
            pnums.append(pnums[-1] + 1)

        prev_name = name

    line_predictions = batch_text_detection(images, det_model, det_processor)
    layout_predictions = batch_layout_detection(images, layout_model, layout_processor, line_predictions)
    table_cells = []

    table_imgs = []
    table_counts = []

    for layout_pred, text_line, img, highres_img in zip(layout_predictions, text_lines, images, highres_images):
        # The table may already be cropped
        if args.skip_table_detection:
            table_imgs.append(highres_img)
            table_counts.append(1)
            page_table_imgs = [highres_img]
            highres_bbox = [[0, 0, highres_img.size[0], highres_img.size[1]]]
        else:
            # The bbox for the entire table
            bbox = [l.bbox for l in layout_pred.bboxes if l.label == "Table"]
            # Number of tables per page
            table_counts.append(len(bbox))

            if len(bbox) == 0:
                continue

            page_table_imgs = []
            highres_bbox = []
            for bb in bbox:
                highres_bb = rescale_bbox(bb, img.size, highres_img.size)
                page_table_imgs.append(highres_img.crop(highres_bb))
                highres_bbox.append(highres_bb)

            table_imgs.extend(page_table_imgs)

        # The text cells inside each table
        table_blocks = get_table_blocks(highres_bbox, text_line, highres_img.size) if text_line is not None else None
        if text_line is None or args.detect_boxes or any(len(tb) == 0 for tb in table_blocks):
            det_results = batch_text_detection(page_table_imgs, det_model, det_processor,)
            cell_bboxes = [[{"bbox": tb.bbox, "text": None} for tb in det_result.bboxes] for det_result in det_results]
            table_cells.extend(cell_bboxes)
        else:
            table_cells.extend(table_blocks)

    table_preds = batch_table_recognition(table_imgs, table_cells, model, processor)
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    img_idx = 0
    prev_count = 0
    table_predictions = defaultdict(list)
    for i in range(sum(table_counts)):
        while i >= prev_count + table_counts[img_idx]:
            prev_count += table_counts[img_idx]
            img_idx += 1

        pred = table_preds[i]
        orig_name = names[img_idx]
        pnum = pnums[img_idx]
        table_img = table_imgs[i]

        out_pred = pred.model_dump()
        out_pred["page"] = pnum + 1
        table_idx = i - prev_count
        out_pred["table_idx"] = table_idx
        table_predictions[orig_name].append(out_pred)

        if args.images:
            rows = [l.bbox for l in pred.rows]
            cols = [l.bbox for l in pred.cols]
            row_labels = [f"Row {l.row_id}" for l in pred.rows]
            col_labels = [f"Col {l.col_id}" for l in pred.cols]
            cells = [l.bbox for l in pred.cells]

            rc_image = copy.deepcopy(table_img)
            rc_image = draw_bboxes_on_image(rows, rc_image, labels=row_labels, label_font_size=20, color="blue")
            rc_image = draw_bboxes_on_image(cols, rc_image, labels=col_labels, label_font_size=20, color="red")
            rc_image.save(os.path.join(result_path, f"{name}_page{pnum + 1}_table{table_idx}_rc.png"))

            cell_image = copy.deepcopy(table_img)
            cell_image = draw_bboxes_on_image(cells, cell_image, color="green")
            cell_image.save(os.path.join(result_path, f"{name}_page{pnum + 1}_table{table_idx}_cells.png"))

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(table_predictions, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()