import collections
import copy
import json
from pathlib import Path

import click

from benchmark.utils.metrics import precision_recall
from surya.debug.draw import draw_polys_on_image
from surya.input.processing import convert_if_not_rgb
from surya.common.util import rescale_bbox
from surya.settings import settings
from surya.detection import DetectionPredictor, InlineDetectionPredictor

import os
import time
from tabulate import tabulate
import datasets


@click.command(help="Benchmark inline math detection model.")
@click.option("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
@click.option("--max_rows", type=int, help="Maximum number of pdf pages to OCR.", default=100)
@click.option("--debug", is_flag=True, help="Enable debug mode.", default=False)
def main(results_dir: str, max_rows: int, debug: bool):
    det_predictor = DetectionPredictor()
    inline_det_predictor = InlineDetectionPredictor()

    dataset = datasets.load_dataset(settings.INLINE_MATH_BENCH_DATASET_NAME, split=f"train[:{max_rows}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    correct_boxes = []
    for i, boxes in enumerate(dataset["bboxes"]):
        img_size = images[i].size
        # Rescale from normalized 0-1 vals to image size
        correct_boxes.append([rescale_bbox(b, (1, 1), img_size) for b in boxes])

    if settings.DETECTOR_STATIC_CACHE:
        # Run through one batch to compile the model
        det_predictor(images[:1])
        inline_det_predictor(images[:1], [[]])

    start = time.time()
    det_results = det_predictor(images)

    # Reformat text boxes to inline math input format
    text_boxes = []
    for result in det_results:
        text_boxes.append([b.bbox for b in result.bboxes])

    inline_results = inline_det_predictor(images, text_boxes)
    surya_time = time.time() - start

    result_path = Path(results_dir) / "inline_math_bench"
    result_path.mkdir(parents=True, exist_ok=True)

    page_metrics = collections.OrderedDict()
    for idx, (sb, cb) in enumerate(zip(inline_results, correct_boxes)):
        surya_boxes = [s.bbox for s in sb.bboxes]
        surya_polys = [s.polygon for s in sb.bboxes]

        surya_metrics = precision_recall(surya_boxes, cb)

        page_metrics[idx] = {
            "surya": surya_metrics,
        }

        if debug:
            bbox_image = draw_polys_on_image(surya_polys, copy.deepcopy(images[idx]))
            bbox_image.save(result_path / f"{idx}_bbox.png")

    mean_metrics = {}
    metric_types = sorted(page_metrics[0]["surya"].keys())
    models = ["surya"]

    for k in models:
        for m in metric_types:
            metric = []
            for page in page_metrics:
                metric.append(page_metrics[page][k][m])
            if k not in mean_metrics:
                mean_metrics[k] = {}
            mean_metrics[k][m] = sum(metric) / len(metric)

    out_data = {
        "times": {
            "surya": surya_time,
        },
        "metrics": mean_metrics,
        "page_metrics": page_metrics
    }

    with open(result_path / "results.json", "w+", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Model", "Time (s)", "Time per page (s)"] + metric_types
    table_data = [
        ["surya", surya_time, surya_time / len(images)] + [mean_metrics["surya"][m] for m in metric_types],
    ]

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print("Precision and recall are over the mutual coverage of the detected boxes and the ground truth boxes at a .5 threshold.  There is a precision penalty for multiple boxes overlapping reference lines.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
