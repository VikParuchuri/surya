import os

import click
import json
import time
from collections import defaultdict

from surya.scripts.config import CLILoader
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


@click.command(help="OCR LaTeX equations.")
@CLILoader.common_options
def ocr_latex_cli(input_path: str, **kwargs):
    loader = CLILoader(input_path, kwargs, highres=True)

    texify_predictor = RecognitionPredictor()
    tasks = [TaskNames.block_without_boxes] * len(loader.images)
    bboxes = [[[0, 0, image.width, image.height]] for image in loader.images]

    start = time.time()
    predictions_by_image = texify_predictor(
        loader.images,
        tasks,
        bboxes=bboxes,
    )

    latex_predictions = [p.text_lines[0].text for p in predictions_by_image]

    if loader.debug:
        print(f"OCR took {time.time() - start:.2f} seconds")
        max_chars = max([len(latex) for latex in latex_predictions])
        print(f"Max chars: {max_chars}")

    out_preds = defaultdict(list)
    for name, pred, image in zip(loader.names, latex_predictions, loader.images):
        out_pred = {
            "equation": pred,
            "page": len(out_preds[name]) + 1,
        }
        out_preds[name].append(out_pred)

    with open(
        os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(out_preds, f, ensure_ascii=False)

    print(f"Wrote results to {loader.result_path}")
