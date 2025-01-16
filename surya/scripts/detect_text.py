import click
import copy
import json
import time
from collections import defaultdict

from surya.detection import DetectionPredictor
from surya.debug.draw import draw_polys_on_image
from surya.scripts.config import CLILoader
import os

@click.command(help="Detect bboxes in an input file or folder (PDFs or image).")
@CLILoader.common_options
def detect_text_cli(input_path: str, **kwargs):
    loader = CLILoader(input_path, kwargs)

    det_predictor = DetectionPredictor()

    start = time.time()
    predictions = det_predictor(loader.images, include_maps=loader.debug)
    end = time.time()
    if loader.debug:
        print(f"Detection took {end - start} seconds")

    if loader.images:
        for idx, (image, pred, name) in enumerate(zip(loader.images, predictions, loader.names)):
            polygons = [p.polygon for p in pred.bboxes]
            bbox_image = draw_polys_on_image(polygons, copy.deepcopy(image))
            bbox_image.save(os.path.join(loader.result_path, f"{name}_{idx}_bbox.png"))

            if loader.debug:
                heatmap = pred.heatmap
                heatmap.save(os.path.join(loader.result_path, f"{name}_{idx}_heat.png"))

    predictions_by_page = defaultdict(list)
    for idx, (pred, name, image) in enumerate(zip(predictions, loader.names, loader.images)):
        out_pred = pred.model_dump(exclude=["heatmap", "affinity_map"])
        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    with open(os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    print(f"Wrote results to {loader.result_path}")