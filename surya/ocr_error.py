from typing import List, Optional
from math import ceil
from tqdm import tqdm
import torch
import numpy as np

from surya.model.ocr_error.model import DistilBertTokenizer
from surya.model.ocr_error.encoder import DistilBertForSequenceClassification
from surya.model.ocr_error.config import ID2LABEL
from surya.settings import settings
from surya.schema import OCRErrorDetectionResult

def get_batch_size():
    batch_size = settings.OCR_ERROR_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 64
    return batch_size

def batch_ocr_error_detection(
  texts: List[str],
  model: DistilBertForSequenceClassification,
  tokenizer: DistilBertTokenizer,
  batch_size: Optional[int] = None
):
    if batch_size is None:
        batch_size = get_batch_size()

    num_batches = ceil(len(texts)/batch_size)
    texts_processed = tokenizer(texts, padding='longest', truncation=True, return_tensors='pt')
    predictions = []
    for batch_idx in tqdm(range(num_batches)):
        start_idx, end_idx = batch_idx*batch_size, (batch_idx+1)*batch_size
        batch_input_ids = texts_processed.input_ids[start_idx:end_idx].to(model.device)
        batch_attention_mask = texts_processed.attention_mask[start_idx:end_idx].to(model.device)

        with torch.inference_mode():
            pred = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = pred.logits.detach().cpu().numpy().astype(np.float32)
            predictions.extend(np.argmax(logits, axis=1).tolist())

    return OCRErrorDetectionResult(
        texts=texts,
        labels=[ID2LABEL[p] for p in predictions]
    )
