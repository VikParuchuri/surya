import argparse
import json
from collections import defaultdict

from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.input.load import load_from_folder, load_from_file, load_lang_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings
import os


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in an input file or folder (PDFs or image).")
    parser.add_argument("input_path", type=str, help="Path to pdf or image file or folder to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "surya"))
    parser.add_argument("--max", type=int, help="Maximum number of pages to process.", default=None)
    parser.add_argument("--start_page", type=int, help="Page to start processing at.", default=0)
    parser.add_argument("--images", action="store_true", help="Save images of detected bboxes.", default=False)
    parser.add_argument("--langs", type=str, help="Language(s) to use for OCR. Comma separate for multiple. Can be a capitalized language name, or a 2-letter ISO 639 code.", default=None)
    parser.add_argument("--lang_file", type=str, help="Path to file with languages to use for OCR. Should be a JSON dict with file names as keys, and the value being a list of language codes/names.", default=None)
    args = parser.parse_args()

    assert args.langs or args.lang_file, "Must provide either --langs or --lang_file"

    if os.path.isdir(args.input_path):
        images, names = load_from_folder(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path)
    else:
        images, names = load_from_file(args.input_path, args.max, args.start_page)
        folder_name = os.path.basename(args.input_path).split(".")[0]

    if args.lang_file:
        # We got all of our language settings from a file
        langs = load_lang_file(args.lang_file, names)
        for lang in langs:
            replace_lang_with_code(lang)
        image_langs = langs
    else:
        # We got our language settings from the input
        langs = args.langs.split(",")
        replace_lang_with_code(langs)
        image_langs = [langs] * len(images)

    det_processor = load_detection_processor()
    det_model = load_detection_model()

    _, lang_tokens = _tokenize("", get_unique_langs(image_langs))
    rec_model = load_recognition_model(langs=lang_tokens) # Prune model moe layer to only include languages we need
    rec_processor = load_recognition_processor()

    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor)

    if args.images:
        for idx, (name, image, pred, langs) in enumerate(zip(names, images, predictions_by_image, image_langs)):
            bboxes = [l.bbox for l in pred.text_lines]
            pred_text = [l.text for l in pred.text_lines]
            page_image = draw_text_on_image(bboxes, pred_text, image.size, langs, has_math="_math" in langs)
            page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    out_preds = defaultdict(list)
    for name, pred, image in zip(names, predictions_by_image, images):
        out_pred = pred.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()







