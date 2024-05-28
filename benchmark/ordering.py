import argparse
import collections
import copy
import json

from surya.input.processing import convert_if_not_rgb
from surya.model.ordering.model import load_model
from surya.model.ordering.processor import load_processor
from surya.ordering import batch_ordering
from surya.settings import settings
from surya.benchmark.metrics import rank_accuracy
import os
import time
import datasets


def main():
    parser = argparse.ArgumentParser(description="Benchmark surya reading order model.")
    parser.add_argument("--results_dir", type=str, help="Path to JSON file with benchmark results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
    parser.add_argument("--max", type=int, help="Maximum number of images to run benchmark on.", default=None)
    args = parser.parse_args()

    model = load_model()
    processor = load_processor()

    pathname = "order_bench"
    # These have already been shuffled randomly, so sampling from the start is fine
    split = "train"
    if args.max is not None:
        split = f"train[:{args.max}]"
    dataset = datasets.load_dataset(settings.ORDER_BENCH_DATASET_NAME, split=split)
    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    bboxes = list(dataset["bboxes"])

    start = time.time()
    order_predictions = batch_ordering(images, bboxes, model, processor)
    surya_time = time.time() - start

    folder_name = os.path.basename(pathname).split(".")[0]
    result_path = os.path.join(args.results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    page_metrics = collections.OrderedDict()
    mean_accuracy = 0
    for idx, order_pred in enumerate(order_predictions):
        row = dataset[idx]
        pred_labels = [str(l.position) for l in order_pred.bboxes]
        labels = row["labels"]
        accuracy = rank_accuracy(pred_labels, labels)
        mean_accuracy += accuracy
        page_results = {
            "accuracy": accuracy,
            "box_count": len(labels)
        }

        page_metrics[idx] = page_results

    mean_accuracy /= len(order_predictions)

    out_data = {
        "time": surya_time,
        "mean_accuracy": mean_accuracy,
        "page_metrics": page_metrics
    }

    with open(os.path.join(result_path, "results.json"), "w+") as f:
        json.dump(out_data, f, indent=4)

    print(f"Mean accuracy is {mean_accuracy:.2f}.")
    print(f"Took {surya_time / len(images):.2f} seconds per image, and {surya_time:.1f} seconds total.")
    print("Mean accuracy is the % of correct ranking pairs.")
    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
