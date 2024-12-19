from __future__ import annotations
import torch

from surya.model.ocr_error.encoder import DistilBertForSequenceClassification
from surya.model.ocr_error.config import DistilBertConfig
from surya.model.ocr_error.tokenizer import DistilBertTokenizer
from surya.settings import settings

def load_model(checkpoint=settings.OCR_ERROR_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE) -> DistilBertForSequenceClassification:
    config = DistilBertConfig.from_pretrained(checkpoint)
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint, torch_dtype=dtype, config=config).to(device).eval()

    if settings.OCR_ERROR_STATIC_CACHE:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 1
        torch._dynamo.config.suppress_errors = False

        print(f"Compiling detection model {checkpoint} on device {device} with dtype {dtype}")
        model = torch.compile(model)

    return model

def load_tokenizer(checkpoint=settings.OCR_ERROR_MODEL_CHECKPOINT):
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
    return tokenizer


