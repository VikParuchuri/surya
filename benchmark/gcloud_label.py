import argparse
import json
from collections import defaultdict

import datasets
from surya.settings import settings
from google.cloud import vision
import hashlib
import os
from tqdm import tqdm
import io

DATA_DIR = os.path.join(settings.BASE_DIR, settings.DATA_DIR)
RESULT_DIR = os.path.join(settings.BASE_DIR, settings.RESULT_DIR)

rtl_langs = ["ar", "fa", "he", "ur", "ps", "sd", "yi", "ug"]

def polygon_to_bbox(polygon):
    x = [vertex["x"] for vertex in polygon["vertices"]]
    y = [vertex["y"] for vertex in polygon["vertices"]]
    return (min(x), min(y), max(x), max(y))


def text_with_break(text, property, is_rtl=False):
    break_type = None
    prefix = False
    if property:
        if "detectedBreak" in property:
            if "type" in property["detectedBreak"]:
                break_type = property["detectedBreak"]["type"]
            if "isPrefix" in property["detectedBreak"]:
                prefix = property["detectedBreak"]["isPrefix"]
    break_char = ""
    if break_type == 1:
        break_char = " "
    if break_type == 5:
        break_char = "\n"

    if is_rtl:
        prefix = not prefix

    if prefix:
        text = break_char + text
    else:
        text = text + break_char
    return text


def bbox_overlap_pct(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    dx = min(x2, x4) - max(x1, x3)
    dy = min(y2, y4) - max(y1, y3)
    if (dx >= 0) and (dy >= 0):
        return dx * dy / ((x2 - x1) * (y2 - y1))
    return 0


def annotate_image(img, client, language, cache_dir):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format)
    img_byte_arr = img_byte_arr.getvalue()

    img_hash = hashlib.sha256(img_byte_arr).hexdigest()
    cache_path = os.path.join(cache_dir, f"{img_hash}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            response = json.load(f)
        return response

    gc_image = vision.Image(content=img_byte_arr)
    context = vision.ImageContext(language_hints=[language])
    response = client.document_text_detection(image=gc_image, image_context=context)
    response_json = vision.AnnotateImageResponse.to_json(response)
    loaded_response = json.loads(response_json)
    with open(cache_path, "w+") as f:
        json.dump(loaded_response, f)
    return loaded_response


def get_line_text(response, lines, is_rtl=False):
    document = response["fullTextAnnotation"]

    bounds = []
    for page in document["pages"]:
        for block in page["blocks"]:
            for paragraph in block["paragraphs"]:
                for word in paragraph["words"]:
                    for symbol in word["symbols"]:
                        bounds.append((symbol["boundingBox"], symbol["text"], symbol.get("property")))

    bboxes = [(polygon_to_bbox(b[0]), text_with_break(b[1], b[2], is_rtl)) for b in bounds]
    line_boxes = defaultdict(list)
    for i, bbox in enumerate(bboxes):
        max_overlap_pct = 0
        max_overlap_idx = None
        for j, line in enumerate(lines):
            overlap = bbox_overlap_pct(bbox[0], line)
            if overlap > max_overlap_pct:
                max_overlap_pct = overlap
                max_overlap_idx = j
        if max_overlap_idx is not None:
            line_boxes[max_overlap_idx].append(bbox)

    ocr_lines = []
    for j, line in enumerate(lines):
        ocr_bboxes = sorted(line_boxes[j], key=lambda x: x[0][0])
        if is_rtl:
            ocr_bboxes = list(reversed(ocr_bboxes))
        ocr_text = "".join([b[1] for b in ocr_bboxes])
        ocr_lines.append(ocr_text)

    assert len(ocr_lines) == len(lines)
    return ocr_lines


def main():
    parser = argparse.ArgumentParser(description="Label text in dataset with google cloud vision.")
    parser.add_argument("--project_id", type=str, help="Google cloud project id.", required=True)
    parser.add_argument("--service_account", type=str, help="Path to service account json.", required=True)
    parser.add_argument("--max", type=int, help="Maximum number of pages to label.", default=None)
    args = parser.parse_args()

    cache_dir = os.path.join(DATA_DIR, "gcloud_cache")
    os.makedirs(cache_dir, exist_ok=True)

    dataset = datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split="train")
    client = vision.ImageAnnotatorClient.from_service_account_json(args.service_account)

    all_gc_lines = []
    for i in tqdm(range(len(dataset))):
        img = dataset[i]["image"]
        lines = dataset[i]["bboxes"]
        language = dataset[i]["language"]

        response = annotate_image(img, client, language, cache_dir)
        ocr_lines = get_line_text(response, lines, is_rtl=language in rtl_langs)

        all_gc_lines.append(ocr_lines)

        if args.max is not None and i >= args.max:
            break

    with open(os.path.join(RESULT_DIR, "gcloud_ocr.json"), "w+") as f:
        json.dump(all_gc_lines, f)


if __name__ == "__main__":
    main()