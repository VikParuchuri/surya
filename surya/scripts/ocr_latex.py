import os
import click
import json
import time
from collections import defaultdict

from surya.scripts.config import CLILoader
from surya.texify import TexifyPredictor


@click.command(help="OCR LaTeX equations.")
@CLILoader.common_options
def ocr_latex_cli(input_path: str, **kwargs):
    loader = CLILoader(input_path, kwargs, highres=True)

    texify_predictor = TexifyPredictor()

    start = time.time()
    predictions_by_image = texify_predictor(
        loader.images,
    )

    if loader.debug:
        print(f"OCR took {time.time() - start:.2f} seconds")
        max_chars = max([len(l.text) for p in predictions_by_image for l in p.text_lines])
        print(f"Max chars: {max_chars}")

    out_preds = defaultdict(list)
    for name, pred, image in zip(loader.names, predictions_by_image, loader.images):
        out_pred = pred.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    print(f"Wrote results to {loader.result_path}")