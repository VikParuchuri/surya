from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.ocr_error.model.config import DistilBertConfig
from surya.ocr_error.model.encoder import DistilBertForSequenceClassification
from surya.ocr_error.tokenizer import DistilBertTokenizer
from surya.settings import settings


class OCRErrorModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.OCR_ERROR_MODEL_CHECKPOINT

        self.checkpoint, self.revision = self.split_checkpoint_revision(self.checkpoint)

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=settings.MODEL_DTYPE
    ) -> DistilBertForSequenceClassification:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        config = DistilBertConfig.from_pretrained(self.checkpoint, revision=self.revision)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            config=config,
            revision=self.revision
        ).to(device).eval()

        if settings.OCR_ERROR_STATIC_CACHE:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 1
            torch._dynamo.config.suppress_errors = False

            print(f"Compiling detection model {self.checkpoint} on device {device} with dtype {dtype}")
            model = torch.compile(model)

        return model

    def processor(
            self
    ) -> DistilBertTokenizer:
        return DistilBertTokenizer.from_pretrained(self.checkpoint, revision=self.revision)