import argparse
import json
from collections import defaultdict

from surya.input.load import load_from_folder, load_from_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings
from surya.languages import LANGUAGE_TO_CODE, CODE_TO_LANGUAGE
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

    # Split and validate language codes
    langs = args.lang.split(",")
    for i in range(len(langs)):
        if langs[i] in LANGUAGE_TO_CODE:
            langs[i] = LANGUAGE_TO_CODE[langs[i]]
        if langs[i] not in CODE_TO_LANGUAGE:
            raise ValueError(f"Language code {langs[i]} not found.")

    det_processor = load_detection_processor()
    det_model = load_detection_model()

    _, lang_tokens = _tokenize("", langs)
    rec_model = load_recognition_model(langs=lang_tokens) # Prune model moes to only include languages we need
    rec_processor = load_recognition_processor()

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    image_langs = [langs] * len(images)
    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor)

    page_num = defaultdict(int)
    for i, pred in enumerate(predictions_by_image):
        pred["name"] = names[i]
        pred["page"] = page_num[names[i]]
        page_num[names[i]] += 1

    if args.images:
        for idx, (name, image, pred) in enumerate(zip(names, images, predictions_by_image)):
            page_image = draw_text_on_image(pred["bboxes"], pred["text_lines"], image.size)
            page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(predictions_by_image, f)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()







