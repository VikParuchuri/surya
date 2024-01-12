import argparse
import copy
import json
from collections import defaultdict

from PIL import Image

from surya.model.segformer import load_model, load_processor
from surya.model.processing import open_pdf, get_page_images
from surya.detection import batch_inference
from surya.postprocessing.affinity import draw_lines_on_image
from surya.postprocessing.heatmap import draw_bboxes_on_image
from surya.settings import settings
import os
import filetype


def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]


def load_pdf(pdf_path, max_pages=None):
    doc = open_pdf(pdf_path)
    page_count = len(doc)
    if max_pages:
        page_count = min(max_pages, page_count)

    page_indices = list(range(page_count))

    images = get_page_images(doc, page_indices)
    doc.close()
    names = [get_name_from_path(pdf_path) for _ in page_indices]
    return images, names


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    name = get_name_from_path(image_path)
    return [image], [name]


def load_from_file(input_path, max_pages=None):
    input_type = filetype.guess(input_path)
    if input_type.extension == "pdf":
        return load_pdf(input_path, max_pages)
    else:
        return load_image(input_path)


def load_from_folder(folder_path, max_pages=None):
    image_paths = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)]
    image_paths = [ip for ip in image_paths if not os.path.isdir(ip) and not ip.startswith(".")]

    images = []
    names = []
    for path in image_paths:
        if filetype.guess(path).extension == "pdf":
            image, name = load_pdf(path, max_pages)
            images.extend(image)
            names.extend(name)
        else:
            image, name = load_image(path)
            images.extend(image)
            names.extend(name)
    return images, names


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--images", action="store_true", help="Save images of detected bboxes.", default=False)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    predictions = batch_inference(images, model, processor)
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    if args.images:
        for idx, (image, pred, name) in enumerate(zip(images, predictions, names)):
            bbox_image = draw_bboxes_on_image(pred["bboxes"], copy.deepcopy(image))
            bbox_image.save(os.path.join(result_path, f"{name}_{idx}_bbox.png"))

            column_image = draw_lines_on_image(pred["vertical_lines"], copy.deepcopy(image))
            column_image.save(os.path.join(result_path, f"{name}_{idx}_column.png"))

            if args.debug:
                heatmap = pred["heatmap"]
                heatmap.save(os.path.join(result_path, f"{name}_{idx}_heat.png"))

                affinity_map = pred["affinity_map"]
                affinity_map.save(os.path.join(result_path, f"{name}_{idx}_affinity.png"))

    # Remove all the images from the predictions
    for pred in predictions:
        pred.pop("heatmap", None)
        pred.pop("affinity_map", None)

    predictions_by_page = defaultdict(list)
    for idx, (pred, name) in enumerate(zip(predictions, names)):
        pred["page_number"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(pred)

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(predictions_by_page, f)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()







