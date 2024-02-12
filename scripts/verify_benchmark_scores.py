import json
import argparse


def verify_det(data):
    scores = data["metrics"]["surya"]
    if scores["precision"] <= 0.9 or scores["recall"] <= 0.9:
        raise ValueError("Scores do not meet the required threshold")


def verify_rec(data):
    scores = data["surya"]
    if scores["avg_score"] <= 0.9:
        raise ValueError("Scores do not meet the required threshold")


def verify_scores(file_path, bench_type):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if bench_type == "detection":
        verify_det(data)
    elif bench_type == "recognition":
        verify_rec(data)
    else:
        raise ValueError("Invalid benchmark type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark scores")
    parser.add_argument("file_path", type=str, help="Path to the json file")
    parser.add_argument("--bench_type", type=str, help="Type of benchmark to verify", default="detection")
    args = parser.parse_args()
    verify_scores(args.file_path, args.bench_type)
