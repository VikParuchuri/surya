import os
import click
import json
import time
from collections import defaultdict

from surya.detection import DetectionPredictor
from surya.recognition.languages import replace_lang_with_code
from surya.input.load import load_lang_file
from surya.debug.text import draw_text_on_image
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader


@click.command(help="OCR text.")
@CLILoader.common_options
@click.option("--langs", type=str, help="Optional language(s) to use for OCR. Comma separate for multiple. Can be a capitalized language name, or a 2-letter ISO 639 code.", default=None)
@click.option("--lang_file", type=str, help="Optional path to file with languages to use for OCR. Should be a JSON dict with file names as keys, and the value being a list of language codes/names.", default=None)
def ocr_text_cli(input_path: str, langs: str, lang_file: str, **kwargs):
    loader = CLILoader(input_path, kwargs, highres=True)

    if lang_file:
        # We got all of our language settings from a file
        langs = load_lang_file(lang_file, loader.names)
        for lang in langs:
            replace_lang_with_code(lang)
        image_langs = langs
    elif langs:
        # We got our language settings from the input
        langs = langs.split(",")
        replace_lang_with_code(langs)
        image_langs = [langs] * len(loader.images)
    else:
        image_langs = [None] * len(loader.images)

    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    start = time.time()
    predictions_by_image = rec_predictor(
        loader.images,
        image_langs,
        det_predictor=det_predictor,
        highres_images=loader.highres_images
    )

    if loader.debug:
        print(f"OCR took {time.time() - start:.2f} seconds")
        max_chars = max([len(l.text) for p in predictions_by_image for l in p.text_lines])
        print(f"Max chars: {max_chars}")

    if loader.images:
        for idx, (name, image, pred, langs) in enumerate(zip(loader.names, loader.images, predictions_by_image, image_langs)):
            bboxes = [l.bbox for l in pred.text_lines]
            pred_text = [l.text for l in pred.text_lines]
            page_image = draw_text_on_image(bboxes, pred_text, image.size, langs)
            page_image.save(os.path.join(loader.result_path, f"{name}_{idx}_text.png"))

    out_preds = defaultdict(list)
    for name, pred, image in zip(loader.names, predictions_by_image, loader.images):
        out_pred = pred.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    print(f"Wrote results to {loader.result_path}")