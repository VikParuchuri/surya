import warnings

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from typing import List, Optional, Tuple
from surya.model.recognition.encoderdecoder import OCREncoderDecoderModel
from surya.model.recognition.config import EfficientViTConfig, SuryaOCRConfig, SuryaOCRDecoderConfig
from surya.model.recognition.encoder import EfficientViTEncoder
from surya.model.recognition.decoder import SuryaOCRDecoder
from surya.settings import settings


def load_model(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE, langs: Optional[List[int]] = None):

    config = SuryaOCRConfig.from_pretrained(checkpoint)
    decoder_config = config.decoder
    decoder = SuryaOCRDecoderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = config.encoder
    encoder = EfficientViTConfig(**encoder_config)
    config.encoder = encoder

    model = OCREncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype, ignore_mismatched_sizes=True)

    assert isinstance(model.decoder, SuryaOCRDecoder)
    assert isinstance(model.encoder, EfficientViTEncoder)

    model = model.to(device)
    model = model.eval()
    print(f"Loaded recognition model {checkpoint} on device {device} with dtype {dtype}")
    return model