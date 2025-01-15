import math
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from surya.common.predictor import BasePredictor
from surya.ocr_error.loader import OCRErrorModelLoader
from surya.ocr_error.model.config import ID2LABEL
from surya.ocr_error.schema import OCRErrorDetectionResult
from surya.settings import settings


class OCRErrorPredictor(BasePredictor):
    model_loader_cls = OCRErrorModelLoader
    batch_size = settings.OCR_ERROR_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 8,
        "mps": 8,
        "cuda": 64
    }

    def __call__(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ):
        return self.batch_ocr_error_detection(texts, batch_size)

    def batch_ocr_error_detection(
            self,
            texts: List[str],
            batch_size: Optional[int] = None
    ):
        if batch_size is None:
            batch_size = self.get_batch_size()

        num_batches = math.ceil(len(texts) / batch_size)
        texts_processed = self.processor(texts, padding='longest', truncation=True, return_tensors='pt')
        predictions = []
        for batch_idx in tqdm(range(num_batches)):
            start_idx, end_idx = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_input_ids = texts_processed.input_ids[start_idx:end_idx].to(self.model.device)
            batch_attention_mask = texts_processed.attention_mask[start_idx:end_idx].to(self.model.device)

            with torch.inference_mode():
                pred = self.model(batch_input_ids, attention_mask=batch_attention_mask)
                logits = pred.logits.detach().cpu().numpy().astype(np.float32)
                predictions.extend(np.argmax(logits, axis=1).tolist())

        return OCRErrorDetectionResult(
            texts=texts,
            labels=[ID2LABEL[p] for p in predictions]
        )