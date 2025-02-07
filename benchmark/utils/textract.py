import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback

from surya.input.processing import slice_bboxes_from_image
from surya.recognition import RecognitionPredictor

def textract_ocr(extractor, img):
    try:
        document = extractor.detect_document_text(file_source=img)
        return [line.text for line in document.lines]
    except:
        traceback.print_exc()
        return [None]

def textract_ocr_parallel(imgs, cpus=None):
    from textractor import Textractor # Optional dependency

    extractor = Textractor(profile_name='default')
    parallel_cores = min(len(imgs), RecognitionPredictor().get_batch_size())
    if not cpus:
        cpus = os.cpu_count()
    parallel_cores = min(parallel_cores, cpus)

    with ThreadPoolExecutor(max_workers=parallel_cores) as executor:
        textract_text = tqdm(executor.map(textract_ocr, [extractor]*len(imgs), imgs), total=len(imgs), desc="Running textract OCR")
        textract_text = list(textract_text)
    return textract_text