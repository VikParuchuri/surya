import argparse
from collections import defaultdict

from benchmark.scoring import overlap_score
from surya.detection import batch_detection
from surya.input.processing import slice_polys_from_image
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.recognition import batch_recognition
from surya.settings import settings
import os
import time
from tabulate import tabulate
import datasets


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in a PDF.")
    parser.add_argument("--pdf_path", type=str, help="Path to PDF to detect bboxes in.", default=None)
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of pdf pages to OCR.", default=100)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    det_model = load_detection_model()
    det_processor = load_detection_processor()
    rec_model = load_recognition_model() # Prune model moes to only include languages we need
    rec_processor = load_recognition_processor()

    dataset = datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split=f"train[:{args.max}]")
    images = list(dataset["image"])
    images = [i.convert("RGB") for i in images]
    bboxes = list(dataset["bboxes"])
    line_text = list(dataset["text"])
    languages = list(dataset["language"])

    lang_list = []
    for l in languages:
        if not isinstance(l, list):
            lang_list.append([l])
        else:
            lang_list.append(l)

    predictions_by_image = run_ocr(images, lang_list, det_model, det_processor, rec_model, rec_processor)

    image_scores = defaultdict(list)
    for idx, (pred, ref_text, lang) in enumerate(zip(predictions_by_image, line_text, languages)):
        image_score = overlap_score(pred["text_lines"], ref_text)
        for l in lang:
            image_scores[l].append(image_score)

    print(image_scores)


if __name__ == "__main__":
    main()
