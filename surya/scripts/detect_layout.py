import time
import click
import copy
import json
from collections import defaultdict

from surya.layout import LayoutPredictor
from surya.debug.draw import draw_polys_on_image
from surya.scripts.config import CLILoader
import os

@click.command(help="Detect layout of an input file or folder (PDFs or image).")
@CLILoader.common_options
def detect_layout_cli(input_path: str, **kwargs):
    loader = CLILoader(input_path, kwargs)

    layout_predictor = LayoutPredictor()

    start = time.time()
    layout_predictions = layout_predictor(loader.images)

    if loader.debug:
        print(f"Layout took {time.time() - start} seconds")

    if loader.images:
        for idx, (image, layout_pred, name) in enumerate(zip(loader.images, layout_predictions, loader.names)):
            polygons = [p.polygon for p in layout_pred.bboxes]
            labels = [f"{p.label}-{p.position}" for p in layout_pred.bboxes]
            bbox_image = draw_polys_on_image(polygons, copy.deepcopy(image), labels=labels)
            bbox_image.save(os.path.join(loader.result_path, f"{name}_{idx}_layout.png"))

    predictions_by_page = defaultdict(list)
    for idx, (pred, name, image) in enumerate(zip(layout_predictions, loader.names, loader.images)):
        out_pred = pred.model_dump()
        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    with open(os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    print(f"Wrote results to {loader.result_path}")