import argparse
from collections import defaultdict

from benchmark.scoring import overlap_score
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.ocr import run_ocr, run_recognition
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings
import os
import datasets
import json


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in a PDF.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of pdf pages to OCR.", default=None)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    rec_model = load_recognition_model()
    rec_processor = load_recognition_processor()

    split = "train"
    if args.max:
        split = f"train[:{args.max}]"

    dataset = datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split=split)
    images = list(dataset["image"])
    images = [i.convert("RGB") for i in images]
    bboxes = list(dataset["bboxes"])
    line_text = list(dataset["text"])
    languages = list(dataset["language"])

    print(f"Loaded {len(images)} images. Running OCR...")

    lang_list = []
    for l in languages:
        if not isinstance(l, list):
            lang_list.append([l])
        else:
            lang_list.append(l)

    predictions_by_image = run_recognition(images, lang_list, rec_model, rec_processor, bboxes=bboxes)

    image_scores = defaultdict(list)
    for idx, (pred, ref_text, lang) in enumerate(zip(predictions_by_image, line_text, lang_list)):
        image_score = overlap_score(pred["text_lines"], ref_text)
        for l in lang:
            image_scores[l].append(image_score)

    image_avgs = {l: sum(scores) / len(scores) for l, scores in image_scores.items()}
    print(image_avgs)

    result_path = os.path.join(args.results_dir, "rec_bench")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(image_scores, f)

    if args.debug:
        for idx, (image, pred, ref_text, bbox, lang) in enumerate(zip(images, predictions_by_image, line_text, bboxes, lang_list)):
            pred_image_name = f"{'_'.join(lang)}_{idx}_pred.png"
            ref_image_name = f"{'_'.join(lang)}_{idx}_ref.png"
            pred_image = draw_text_on_image(bbox, pred["text_lines"], image.size)
            pred_image.save(os.path.join(result_path, pred_image_name))
            ref_image = draw_text_on_image(bbox, ref_text, image.size)
            ref_image.save(os.path.join(result_path, ref_image_name))
            image.save(os.path.join(result_path, f"{'_'.join(lang)}_{idx}_image.png"))

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
