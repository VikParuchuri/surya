from typing import List, Optional

import numpy as np
import pytesseract
from pytesseract import Output
from tqdm import tqdm

from surya.input.processing import slice_bboxes_from_image
from surya.settings import settings
import os
from concurrent.futures import ProcessPoolExecutor
from surya.detection import get_batch_size as get_det_batch_size
from surya.recognition import get_batch_size as get_rec_batch_size
from surya.languages import CODE_TO_LANGUAGE


def surya_lang_to_tesseract(code: str) -> Optional[str]:
    lang_str = CODE_TO_LANGUAGE[code]
    try:
        tess_lang = TESS_LANGUAGE_TO_CODE[lang_str]
    except KeyError:
        return None
    return tess_lang


def tesseract_ocr(img, bboxes, lang: str):
    line_imgs = slice_bboxes_from_image(img, bboxes)
    config = f'--tessdata-dir "{settings.TESSDATA_PREFIX}"'
    lines = []
    for line_img in line_imgs:
        line = pytesseract.image_to_string(line_img, lang=lang, config=config)
        lines.append(line)
    return lines


def tesseract_ocr_parallel(imgs, bboxes, langs: List[str], cpus=None):
    tess_parallel_cores = min(len(imgs), get_rec_batch_size())
    if not cpus:
        cpus = os.cpu_count()
    tess_parallel_cores = min(tess_parallel_cores, cpus)

    # Tesseract uses up to 4 processes per instance
    # Divide by 2 because tesseract doesn't seem to saturate all 4 cores with these small images
    tess_parallel = max(tess_parallel_cores // 2, 1)

    with ProcessPoolExecutor(max_workers=tess_parallel) as executor:
        tess_text = tqdm(executor.map(tesseract_ocr, imgs, bboxes, langs), total=len(imgs), desc="Running tesseract OCR")
        tess_text = list(tess_text)
    return tess_text


def tesseract_bboxes(img):
    arr_img = np.asarray(img, dtype=np.uint8)
    ocr = pytesseract.image_to_data(arr_img, output_type=Output.DICT)

    bboxes = []
    n_boxes = len(ocr['level'])
    for i in range(n_boxes):
        # It is possible to merge by line here with line number, but it gives bad results.
        _, x, y, w, h = ocr['text'][i], ocr['left'][i], ocr['top'][i], ocr['width'][i], ocr['height'][i]
        bbox = (x, y, x + w, y + h)
        bboxes.append(bbox)

    return bboxes


def tesseract_parallel(imgs):
    # Tesseract uses 4 threads per instance
    tess_parallel_cores = min(len(imgs), get_det_batch_size())
    cpus = os.cpu_count()
    tess_parallel_cores = min(tess_parallel_cores, cpus)

    # Tesseract uses 4 threads per instance
    tess_parallel = max(tess_parallel_cores // 4, 1)

    with ProcessPoolExecutor(max_workers=tess_parallel) as executor:
        tess_bboxes = tqdm(executor.map(tesseract_bboxes, imgs), total=len(imgs), desc="Running tesseract bbox detection")
        tess_bboxes = list(tess_bboxes)
    return tess_bboxes


TESS_CODE_TO_LANGUAGE = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "ara": "Arabic",
    "asm": "Assamese",
    "aze": "Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bod": "Tibetan",
    "bos": "Bosnian",
    "bre": "Breton",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "chi_sim": "Chinese",
    "chr": "Cherokee",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "dzo": "Dzongkha",
    "ell": "Greek",
    "eng": "English",
    "epo": "Esperanto",
    "est": "Estonian",
    "eus": "Basque",
    "fas": "Persian",
    "fin": "Finnish",
    "fra": "French",
    "fry": "Western Frisian",
    "guj": "Gujarati",
    "gla": "Scottish Gaelic",
    "gle": "Irish",
    "glg": "Galician",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "iku": "Inuktitut",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lat": "Latin",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mon": "Mongolian",
    "msa": "Malay",
    "mya": "Burmese",
    "nep": "Nepali",
    "nld": "Dutch",
    "nor": "Norwegian",
    "ori": "Oriya",
    "pan": "Punjabi",
    "pol": "Polish",
    "por": "Portuguese",
    "pus": "Pashto",
    "ron": "Romanian",
    "rus": "Russian",
    "san": "Sanskrit",
    "sin": "Sinhala",
    "slk": "Slovak",
    "slv": "Slovenian",
    "snd": "Sindhi",
    "spa": "Spanish",
    "sqi": "Albanian",
    "srp": "Serbian",
    "swa": "Swahili",
    "swe": "Swedish",
    "syr": "Syriac",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tha": "Thai",
    "tir": "Tigrinya",
    "tur": "Turkish",
    "uig": "Uyghur",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzb": "Uzbek",
    "vie": "Vietnamese",
    "yid": "Yiddish"
}

TESS_LANGUAGE_TO_CODE = {v:k for k,v in TESS_CODE_TO_LANGUAGE.items()}
