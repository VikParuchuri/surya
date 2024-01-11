import argparse
import copy
import json

from PIL import Image

from surya.model.segformer import load_model, load_processor
from surya.model.processing import open_pdf, get_page_images
from surya.detection import batch_inference
from surya.postprocessing.affinity import draw_lines_on_image
from surya.postprocessing.heatmap import draw_bboxes_on_image
from surya.settings import settings
import os
import filetype


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in an input file (PDF or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--images", action="store_true", help="Save images of detected bboxes.", default=False)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    input_type = filetype.guess(args.input_path)
    if input_type.extension == "pdf":
        doc = open_pdf(args.input_path)
        page_count = len(doc)
        if args.max:
            page_count = min(args.max, page_count)

        page_indices = list(range(page_count))

        images = get_page_images(doc, page_indices)
        doc.close()
    else:
        image = Image.open(args.input_path).convert("RGB")
        images = [image]

    predictions = batch_inference(images, model, processor)

    folder_name = os.path.basename(args.input_path).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    if args.images:
        for idx, (image, pred) in enumerate(zip(images, predictions)):
            bbox_image = draw_bboxes_on_image(pred["bboxes"], copy.deepcopy(image))
            bbox_image.save(os.path.join(result_path, f"{idx}_bbox.png"))

            column_image = draw_lines_on_image(pred["vertical_lines"], copy.deepcopy(image))
            column_image.save(os.path.join(result_path, f"{idx}_column.png"))

            if args.debug:
                heatmap = pred["heatmap"]
                heatmap.save(os.path.join(result_path, f"{idx}_heat.png"))

                affinity_map = pred["affinity_map"]
                affinity_map.save(os.path.join(result_path, f"{idx}_affinity.png"))

    # Remove all the images from the predictions
    for pred in predictions:
        pred.pop("heatmap", None)
        pred.pop("affinity_map", None)

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(predictions, f, indent=4)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()







