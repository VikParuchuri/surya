import argparse
import collections
import json

from surya.benchmark.bbox import get_pdf_lines
from surya.benchmark.metrics import mean_coverage
from surya.benchmark.tesseract import tesseract_bboxes
from surya.model.segformer import load_model, load_processor
from surya.model.processing import open_pdf, get_page_images
from surya.detection import batch_inference
from surya.settings import settings
import os
import time
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(description="Detect bboxes in a PDF.")
    parser.add_argument("pdf_path", type=str, help="Path to PDF to detect bboxes in.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of pdf pages to OCR.", default=10)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    doc = open_pdf(args.pdf_path)
    page_count = len(doc)
    page_indices = list(range(page_count))

    if args.max:
        page_indices = page_indices[:args.max]

    images = get_page_images(doc, page_indices)
    doc.close()

    start = time.time()
    predictions = batch_inference(images, model, processor)
    surya_time = time.time() - start

    start = time.time()
    tess_predictions = [tesseract_bboxes(img) for img in images]
    tess_time = time.time() - start

    image_sizes = [img.size for img in images]

    correct_boxes = get_pdf_lines(args.pdf_path, image_sizes)

    folder_name = os.path.basename(args.pdf_path).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    coverages = collections.OrderedDict()
    for idx, (tb, sb, cb) in enumerate(zip(tess_predictions, predictions, correct_boxes)):
        surya_boxes = sb["bboxes"]

        surya_coverage = mean_coverage(surya_boxes, cb)
        tess_coverage = mean_coverage(tb, cb)

        coverages[idx] = {
            "surya": surya_coverage,
            "tesseract": tess_coverage
        }

    avg_surya_coverage = sum([c["surya"] for c in coverages.values()]) / len(coverages)
    avg_tess_coverage = sum([c["tesseract"] for c in coverages.values()]) / len(coverages)

    out_data = {
        "times": {
            "surya": surya_time,
            "tesseract": tess_time
        },
        "coverage": {
            "surya": avg_surya_coverage,
            "tesseract": avg_tess_coverage
        },
        "page_coverage": coverages
    }

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Model", "Time (s)", "Time per page (s)", "Coverage (%)"]
    table_data = [
        ["surya", surya_time, surya_time / len(images), avg_surya_coverage],
        ["tesseract", tess_time, tess_time / len(images), avg_tess_coverage]
    ]

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print("Coverage is the average % of the correct bboxes that are covered by a predicted bbox, and vice versa.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()







