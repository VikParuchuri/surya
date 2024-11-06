import argparse
import collections
import copy
import json

from surya.benchmark.metrics import precision_recall
from surya.detection import batch_text_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.layout.model import load_model, load_processor
from surya.input.processing import convert_if_not_rgb
from surya.layout import batch_layout_detection
from surya.postprocessing.heatmap import draw_bboxes_on_image
from surya.settings import settings
import os
import time
from tabulate import tabulate
import datasets


def main():
    parser = argparse.ArgumentParser(description="Benchmark surya layout model.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=100)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.", default=False)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()
    det_model = load_det_model()
    det_processor = load_det_processor()

    pathname = "layout_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    dataset = datasets.load_dataset(settings.LAYOUT_BENCH_DATASET_NAME, split=f"train[:{args.max}]")
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)

    if settings.LAYOUT_STATIC_CACHE:
        line_prediction = batch_text_detection(images[:1], det_model, det_processor)
        batch_layout_detection(images[:1], model, processor, line_prediction)

    start = time.time()
    line_predictions = batch_text_detection(images, det_model, det_processor)
    layout_predictions = batch_layout_detection(images, model, processor, line_predictions)
    surya_time = time.time() - start

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    label_alignment = { # First is publaynet, second is surya
        "Image": [["Figure"], ["Picture", "Figure"]],
        "Table": [["Table"], ["Table"]],
        "Text": [["Text", "List"], ["Text", "Formula", "Footnote", "Caption", "List-item"]],
        "Title": [["Title"], ["Section-header", "Title"]]
    }

    page_metrics = collections.OrderedDict()
    for idx, pred in enumerate(layout_predictions):
        row = dataset[idx]
        all_correct_bboxes = []
        page_results = {}
        for label_name in label_alignment:
            correct_cats, surya_cats = label_alignment[label_name]
            correct_bboxes = [b for b, l in zip(row["bboxes"], row["labels"]) if l in correct_cats]
            all_correct_bboxes.extend(correct_bboxes)
            pred_bboxes = [b.bbox for b in pred.bboxes if b.label in surya_cats]

            metrics = precision_recall(pred_bboxes, correct_bboxes, penalize_double=False)
            weight = len(correct_bboxes)
            metrics["weight"] = weight
            page_results[label_name] = metrics

        page_metrics[idx] = page_results

        if args.debug:
            bbox_image = draw_bboxes_on_image(all_correct_bboxes, copy.deepcopy(images[idx]))
            bbox_image.save(os.path.join(result_path, f"{idx}_layout.png"))

    mean_metrics = collections.defaultdict(dict)
    layout_types = sorted(page_metrics[0].keys())
    metric_types = sorted(page_metrics[0][layout_types[0]].keys())
    metric_types.remove("weight")
    for l in layout_types:
        for m in metric_types:
            metric = []
            total = 0
            for page in page_metrics:
                metric.append(page_metrics[page][l][m] * page_metrics[page][l]["weight"])
                total += page_metrics[page][l]["weight"]

            value = sum(metric)
            if value > 0:
                value /= total
            mean_metrics[l][m] = value

    out_data = {
        "time": surya_time,
        "metrics": mean_metrics,
        "page_metrics": page_metrics
    }

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    table_headers = ["Layout Type", ] + metric_types
    table_data = []
    for layout_type in layout_types:
        table_data.append([layout_type, ] + [f"{mean_metrics[layout_type][m]:.5f}" for m in metric_types])

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print(f"Took {surya_time / len(images):.5f} seconds per image, and {surya_time:.5f} seconds total.")
    print("Precision and recall are over the mutual coverage of the detected boxes and the ground truth boxes at a .5 threshold.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
