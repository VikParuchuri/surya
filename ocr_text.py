import argparse
import json
from collections import defaultdict

from surya.input.load import load_from_folder, load_from_file
from surya.input.processing import slice_polys_from_image
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.postprocessing.text import draw_text_on_image
from surya.detection import batch_detection
from surya.recognition import batch_recognition
from surya.settings import settings
import os


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--start_page", type=int, help="Page to start processing at.", default=0)
    parser.add_argument("--images", action="store_true", help="Save images of detected bboxes.", default=False)
    parser.add_argument("--lang", type=str, help="Language to use for OCR. Comma separate for multiple.", default="en")
    args = parser.parse_args()

    langs = args.lang.split(",")
    detection_processor = load_detection_processor()
    detection_model = load_detection_model()

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    det_predictions = batch_detection(images, detection_model, detection_processor)
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    del detection_processor
    del detection_model

    _, lang_tokens = _tokenize("", langs)
    recognition_model = load_recognition_model(langs=lang_tokens) # Prune model moes to only include languages we need
    recognition_processor = load_recognition_processor()

    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, pred, name) in enumerate(zip(images, det_predictions, names)):
        slices = slice_polys_from_image(image, pred["polygons"])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([langs] * len(slices))

    rec_predictions = batch_recognition(all_slices, all_langs, recognition_model, recognition_processor)

    predictions_by_page = defaultdict(list)
    slice_start = 0
    for idx, (image, det_pred, name) in enumerate(zip(images, det_predictions, names)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        slice_start = slice_end

        assert len(image_lines) == len(det_pred["polygons"]) == len(det_pred["bboxes"])
        predictions_by_page[name].append({
            "lines": image_lines,
            "polys": det_pred["polygons"],
            "bboxes": det_pred["bboxes"],
            "name": name,
            "page_number": len(predictions_by_page[name]) + 1
        })

        if args.images:
            page_image = draw_text_on_image(det_pred["bboxes"], image_lines, image.size)
            page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(predictions_by_page, f)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()







