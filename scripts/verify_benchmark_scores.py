import json
import argparse


def verify_scores(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    scores = data["metrics"]["surya"]

    if scores["precision"] <= 0.9 or scores["recall"] <= 0.9:
        print(scores)
        raise ValueError("Scores do not meet the required threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark scores")
    parser.add_argument("file_path", type=str, help="Path to the json file")
    args = parser.parse_args()
    verify_scores(args.file_path)
