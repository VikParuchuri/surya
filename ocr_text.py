import argparse
import copy
import json
from collections import defaultdict

from surya.input.load import load_from_folder, load_from_file
from surya.input.processing import slice_polys_from_image
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.detection import batch_detection
from surya.recognition import batch_recognition
from surya.postprocessing.affinity import draw_lines_on_image
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.settings import settings
import os


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--start_page", type=int, help="Page to start processing at.", default=0)
    parser.add_argument("--images", action="store_true", help="Save images of detected bboxes.", default=False)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    parser.add_argument("--lang", type=str, help="Language to use for OCR. Comma separate for multiple.", default="en")
    args = parser.parse_args()

    langs = args.lang.split(",")
    detection_model = load_detection_model()
    detection_processor = load_detection_processor()

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    predictions = batch_detection(images, detection_model, detection_processor)
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    del detection_processor
    del detection_model

    recognition_model = load_recognition_model()
    recognition_processor = load_recognition_processor()

    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, pred, name) in enumerate(zip(images, predictions, names)):
        slices = slice_polys_from_image(image, pred["polygons"])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([langs] * len(slices))

    predictions = batch_recognition(all_slices, all_langs, recognition_model, recognition_processor)
    print(predictions)


if __name__ == "__main__":
    main()







