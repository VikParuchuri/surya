from typing import Optional

import torch
from transformers.utils import is_flash_attn_2_available

from surya.common.load import ModelLoader
from surya.common.surya.config import SuryaModelConfig
from surya.common.surya import SuryaModel
from surya.common.surya.processor import SuryaOCRProcessor
from surya.common.surya.processor.tokenizer import SuryaOCRTokenizer
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


class FoundationModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):

        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.RECOGNITION_MODEL_CHECKPOINT

    def model(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE_BFLOAT
    ) -> SuryaModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE_BFLOAT

        torch.set_float32_matmul_precision("high")
        config = SuryaModelConfig.from_pretrained(self.checkpoint)

        if is_flash_attn_2_available():
            config.decoder._attn_implementation = "flash_attention_2"
            config.vision_encoder._attn_implementation = "flash_attention_2"
        else:
            config.decoder._attn_implementation = "sdpa"
            config.vision_encoder._attn_implementation = "sdpa"

        model = SuryaModel.from_pretrained(
            self.checkpoint, torch_dtype=dtype, config=config
        ).to(device)
        model = model.eval()

        logger.debug(
            f"Loaded recognition model {self.checkpoint} from {SuryaModel.get_local_path(self.checkpoint)} onto device {model.device} with dtype {dtype}, using decoder attention mechanism {model.config.decoder._attn_implementation}, encoder attention mechanism {model.config.vision_encoder._attn_implementation}."
        )
        return model

    def processor(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE_BFLOAT
    ) -> SuryaOCRProcessor:
        config: SuryaModelConfig = SuryaModelConfig.from_pretrained(self.checkpoint)

        ocr_tokenizer = SuryaOCRTokenizer(
            special_tokens=config.special_ocr_tokens, model_checkpoint=self.checkpoint
        )

        processor = SuryaOCRProcessor(
            ocr_tokenizer=ocr_tokenizer,
            blank_bbox_token_id=config.blank_bbox_token_id,
            num_register_tokens=config.num_register_tokens,
            sequence_length=None,
            patch_size=config.vision_encoder.patch_size,
            merge_size=config.vision_encoder.spatial_merge_size,
            model_device=device,
        )
        config.eos_token_id = processor.eos_token_id
        config.pad_token_id = processor.pad_token_id
        config.bos_token_id = processor.bos_token_id

        return processor
