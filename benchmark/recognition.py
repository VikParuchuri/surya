import argparse
from collections import defaultdict

import click

from benchmark.utils.scoring import overlap_score
from surya.input.processing import convert_if_not_rgb
from surya.debug.text import draw_text_on_image
from surya.recognition import RecognitionPredictor
from surya.settings import settings
from surya.recognition.languages import CODE_TO_LANGUAGE
from benchmark.utils.tesseract import tesseract_ocr_parallel, surya_lang_to_tesseract, TESS_CODE_TO_LANGUAGE
from benchmark.utils.textract import textract_ocr_parallel
import os
import datasets
import json
import time
from tabulate import tabulate

KEY_LANGUAGES = ["Chinese", "Spanish", "English", "Arabic", "Hindi", "Bengali", "Russian", "Japanese"]

@click.command(help="Benchmark recognition model.")
@click.option("--results_dir", type=str, help="Path to JSON file with OCR results.", default=os.path.join(settings.RESULT_DIR, "benchmark"))
@click.option("--max_rows", type=int, help="Maximum number of pdf pages to OCR.", default=None)
@click.option("--debug", is_flag=True, help="Enable debug mode.", default=False)
@click.option("--tesseract", is_flag=True, help="Run benchmarks on tesseract.", default=False)
@click.option("--textract", is_flag=True, help="Run benchmarks on textract.", default=False)
@click.option("--langs", type=str, help="Specify certain languages to benchmark.", default=None)
@click.option("--tess_cpus", type=int, help="Number of CPUs to use for tesseract.", default=28)
@click.option("--textract_cpus", type=int, help="Number of CPUs to use for textract.", default=28)
@click.option("--specify_language", is_flag=True, help="Pass language codes into the model.", default=False)
def main(results_dir: str, max_rows: int, debug: bool, tesseract: bool, textract: bool, langs: str, tess_cpus: int, textract_cpus:int, specify_language: bool):
    rec_predictor = RecognitionPredictor()

    split = "train"
    dataset = datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split=split)

    if langs:
        langs = langs.split(",")
        dataset = dataset.filter(lambda x: x["language"] in langs, num_proc=4)
    
    if max_rows and max_rows<len(dataset):
        dataset = dataset.shuffle().select(range(max_rows))

    images = list(dataset["image"])
    images = convert_if_not_rgb(images)
    bboxes = list(dataset["bboxes"])
    line_text = list(dataset["text"])
    languages = list(dataset["language"])

    print(f"Loaded {len(images)} images. Running OCR...")

    lang_list = []
    for l in languages:
        if not isinstance(l, list):
            lang_list.append([l])
        else:
            lang_list.append(l)
    n_list = [None] * len(images)

    if settings.RECOGNITION_STATIC_CACHE:
        # Run through one batch to compile the model
        rec_predictor(images[:1], lang_list[:1], bboxes=bboxes[:1])

    start = time.time()
    predictions_by_image = rec_predictor(images, lang_list if specify_language else n_list, bboxes=bboxes)
    surya_time = time.time() - start

    surya_scores = defaultdict(list)
    img_surya_scores = []
    for idx, (pred, ref_text, lang) in enumerate(zip(predictions_by_image, line_text, lang_list)):
        pred_text = [l.text for l in pred.text_lines]
        image_score = overlap_score(pred_text, ref_text)
        img_surya_scores.append(image_score)
        for l in lang:
            surya_scores[CODE_TO_LANGUAGE[l]].append(image_score)

    flat_surya_scores = [s for l in surya_scores for s in surya_scores[l]]
    benchmark_stats = {
        "surya": {
            "avg_score": sum(flat_surya_scores) / max(1, len(flat_surya_scores)),
            "lang_scores": {l: sum(scores) / max(1, len(scores)) for l, scores in surya_scores.items()},
            "time_per_img": surya_time / max(1, len(images))
        }
    }

    result_path = os.path.join(results_dir, "rec_bench")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "surya_scores.json"), "w+") as f:
        json.dump(surya_scores, f)

    if tesseract:
        tess_valid = []
        tess_langs = []
        for idx, lang in enumerate(lang_list):
            # Tesseract does not support all languages
            tess_lang = surya_lang_to_tesseract(lang[0])
            if tess_lang is None:
                continue

            tess_valid.append(idx)
            tess_langs.append(tess_lang)

        tess_imgs = [images[i] for i in tess_valid]
        tess_bboxes = [bboxes[i] for i in tess_valid]
        tess_reference = [line_text[i] for i in tess_valid]
        start = time.time()
        tess_predictions = tesseract_ocr_parallel(tess_imgs, tess_bboxes, tess_langs, cpus=tess_cpus)
        tesseract_time = time.time() - start

        tess_scores = defaultdict(list)
        for idx, (pred, ref_text, lang) in enumerate(zip(tess_predictions, tess_reference, tess_langs)):
            image_score = overlap_score(pred, ref_text)
            tess_scores[TESS_CODE_TO_LANGUAGE[lang]].append(image_score)

        flat_tess_scores = [s for l in tess_scores for s in tess_scores[l]]
        benchmark_stats["tesseract"] = {
            "avg_score": sum(flat_tess_scores) / len(flat_tess_scores),
            "lang_scores": {l: sum(scores) / len(scores) for l, scores in tess_scores.items()},
            "time_per_img": tesseract_time / len(tess_imgs)
        }

        with open(os.path.join(result_path, "tesseract_scores.json"), "w+") as f:
            json.dump(tess_scores, f)

    if textract:
        start = time.time()
        textract_predictions = textract_ocr_parallel(images, cpus=textract_cpus)
        textract_time = time.time()-start

        textract_scores = defaultdict(list)
        for idx, (pred, ref_text, lang) in enumerate(zip(textract_predictions, line_text, lang_list)):
            image_score = overlap_score(pred, ref_text)
            for l in lang:
                textract_scores[CODE_TO_LANGUAGE[l]].append(image_score)

        flat_textract_scores = [s for l in textract_scores for s in textract_scores[l]]
        benchmark_stats["textract"] = {
            "avg_score": sum(flat_textract_scores) / len(flat_textract_scores),
            "lang_scores": {l: sum(scores) / len(scores) for l, scores in textract_scores.items()},
            "time_per_img": textract_time / len(images)
        }
        print(len(flat_textract_scores))

        with open(os.path.join(result_path, "textract_scores.json"), "w+") as f:
            json.dump(textract_scores, f)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(benchmark_stats, f)

    key_languages = [k for k in KEY_LANGUAGES if k in surya_scores]
    table_headers = ["Model", "Time per page (s)", "Avg Score"] + key_languages
    table_data = [
        ["surya", benchmark_stats["surya"]["time_per_img"], benchmark_stats["surya"]["avg_score"]] + [benchmark_stats["surya"]["lang_scores"][l] for l in key_languages],
    ]
    if tesseract:
        table_data.append(
            ["tesseract", benchmark_stats["tesseract"]["time_per_img"], benchmark_stats["tesseract"]["avg_score"]] + [benchmark_stats["tesseract"]["lang_scores"].get(l, 0) for l in key_languages]
        )
    if textract:
        table_data.append(
            ["textract", benchmark_stats["textract"]["time_per_img"], benchmark_stats["textract"]["avg_score"]] + [benchmark_stats["textract"]["lang_scores"][l] for l in key_languages],
        )

    print(tabulate(table_data, headers=table_headers, tablefmt="github"))
    print("Only a few major languages are displayed. See the result path for additional languages.")

    if debug >= 1:
        bad_detections = []
        for idx, (score, lang) in enumerate(zip(flat_surya_scores, lang_list)):
            if score < .8:
                bad_detections.append((idx, lang, score))
        print(f"Found {len(bad_detections)} bad detections. Writing to file...")
        with open(os.path.join(result_path, "bad_detections.json"), "w+") as f:
            json.dump(bad_detections, f)

    if debug == 2:
        for idx, (image, pred, ref_text, bbox, lang) in enumerate(zip(images, predictions_by_image, line_text, bboxes, lang_list)):
            pred_image_name = f"{'_'.join(lang)}_{idx}_pred.png"
            ref_image_name = f"{'_'.join(lang)}_{idx}_ref.png"
            pred_text = [l.text for l in pred.text_lines]
            pred_image = draw_text_on_image(bbox, pred_text, image.size, lang)
            pred_image.save(os.path.join(result_path, pred_image_name))
            ref_image = draw_text_on_image(bbox, ref_text, image.size, lang)
            ref_image.save(os.path.join(result_path, ref_image_name))
            image.save(os.path.join(result_path, f"{'_'.join(lang)}_{idx}_image.png"))

    print(f"Wrote results to {result_path}")


if __name__ == "__main__":
    main()
