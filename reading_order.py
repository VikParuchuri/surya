import argparse
import copy
import json
from collections import defaultdict

from surya.detection import batch_text_detection
from surya.input.load import load_from_folder, load_from_file
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.ordering.model import load_model
from surya.model.ordering.processor import load_processor
from surya.ordering import batch_ordering
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.settings import settings
import os


def main():
    parser = argparse.ArgumentParser(description="Find reading order of an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file or folder to find reading order in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with layout results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--images", action="store_true", help="Save images of detected layout bboxes.", default=False)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    layout_model = load_det_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    layout_processor = load_det_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

    det_model = load_det_model()
    det_processor = load_det_processor()

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    line_predictions = batch_text_detection(images, det_model, det_processor)
    layout_predictions = batch_layout_detection(images, layout_model, layout_processor, line_predictions)
    bboxes = []
    for layout_pred in layout_predictions:
        bbox = [l.bbox for l in layout_pred.bboxes]
        bboxes.append(bbox)

    order_predictions = batch_ordering(images, bboxes, model, processor)
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    if args.images:
        for idx, (image, layout_pred, order_pred, name) in enumerate(zip(images, layout_predictions, order_predictions, names)):
            polys = [l.polygon for l in order_pred.bboxes]
            labels = [str(l.position) for l in order_pred.bboxes]
            bbox_image = draw_polys_on_image(polys, copy.deepcopy(image), labels=labels, label_font_size=20)
            bbox_image.save(os.path.join(result_path, f"{name}_{idx}_order.png"))

    predictions_by_page = defaultdict(list)
    for idx, (layout_pred, pred, name, image) in enumerate(zip(layout_predictions, order_predictions, names, images)):
        out_pred = pred.model_dump()
        for bbox, layout_bbox in zip(out_pred["bboxes"], layout_pred.bboxes):
            bbox["label"] = layout_bbox.label

        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    # Sort in reading order
    for name in predictions_by_page:
        for page_preds in predictions_by_page[name]:
            page_preds["bboxes"] = sorted(page_preds["bboxes"], key=lambda x: x["position"])

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
