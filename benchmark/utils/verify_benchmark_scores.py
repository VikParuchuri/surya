import json
import argparse


def verify_layout(data):
    scores = data["metrics"]
    for layout_type, metrics in scores.items():
        if layout_type == "List": # Skip lists since none appear early on
            continue

        if metrics["precision"] <= 0.6 or metrics["recall"] <= 0.6:
            raise ValueError("Scores do not meet the required threshold")


def verify_det(data):
    scores = data["metrics"]["surya"]
    if scores["precision"] <= 0.9 or scores["recall"] <= 0.9:
        raise ValueError("Scores do not meet the required threshold")


def verify_rec(data):
    scores = data["surya"]
    if scores["avg_score"] <= 0.9:
        raise ValueError("Scores do not meet the required threshold")


def verify_order(data):
    score = data["mean_accuracy"]
    if score < 0.75:
        raise ValueError("Scores do not meet the required threshold")


def verify_table_rec(data):
    row_score = data["surya"]["mean_row_iou"]
    col_score = data["surya"]["mean_col_iou"]

    if row_score < 0.75 or col_score < 0.75:
        raise ValueError("Scores do not meet the required threshold")


def verify_scores(file_path, bench_type):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if bench_type == "detection":
        verify_det(data)
    elif bench_type == "recognition":
        verify_rec(data)
    elif bench_type == "layout":
        verify_layout(data)
    elif bench_type == "ordering":
        verify_order(data)
    elif bench_type == "table_recognition":
        verify_table_rec(data)
    else:
        raise ValueError("Invalid benchmark type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark scores")
    parser.add_argument("file_path", type=str, help="Path to the json file")
    parser.add_argument("--bench_type", type=str, help="Type of benchmark to verify", default="detection")
    args = parser.parse_args()
    verify_scores(args.file_path, args.bench_type)
