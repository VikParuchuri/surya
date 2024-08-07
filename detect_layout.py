import argparse
import copy
import json
import os
from collections import defaultdict

from surya.detection import batch_text_detection
from surya.input.load import load_from_file, load_from_folder
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Detect layout of an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file or folder to detect layout in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with layout results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--images", action="store_true", help="Save images of detected layout bboxes.", default=False)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    line_predictions = batch_text_detection(images, det_model, det_processor)

    layout_predictions = batch_layout_detection(images, model, processor, line_predictions)
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    if args.images:
        for idx, (image, layout_pred, name) in enumerate(zip(images, layout_predictions, names)):
            polygons = [p.polygon for p in layout_pred.bboxes]
            labels = [p.label for p in layout_pred.bboxes]
            bbox_image = draw_polys_on_image(polygons, copy.deepcopy(image), labels=labels)
            bbox_image.save(os.path.join(result_path, f"{name}_{idx}_layout.png"))

            if args.debug:
                heatmap = layout_pred.segmentation_map
                heatmap.save(os.path.join(result_path, f"{name}_{idx}_segmentation.png"))

    predictions_by_page = defaultdict(list)
    for idx, (pred, name, image) in enumerate(zip(layout_predictions, names, images)):
        out_pred = pred.model_dump(exclude=["segmentation_map"])
        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
