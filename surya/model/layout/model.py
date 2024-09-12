import warnings

import torch

from surya.model.layout.config import SuryaLayoutConfig, SuryaLayoutDecoderConfig, DonutSwinLayoutConfig, \
    SuryaLayoutTextEncoderConfig
from surya.model.layout.decoder import SuryaLayoutDecoder
from surya.model.layout.encoderdecoder import LayoutEncoderDecoderModel
from surya.model.recognition.decoder import SuryaOCRTextEncoder
from surya.model.recognition.encoder import DonutSwinModel

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from typing import List, Optional, Tuple
from surya.settings import settings

if not settings.ENABLE_EFFICIENT_ATTENTION:
    print("Efficient attention is disabled. This will use significantly more VRAM.")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
    config = SuryaLayoutConfig.from_pretrained(checkpoint)
    decoder_config = config.decoder
    decoder = SuryaLayoutDecoderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = config.encoder
    encoder = DonutSwinLayoutConfig(**encoder_config)
    config.encoder = encoder

    text_encoder_config = config.text_encoder
    text_encoder = SuryaLayoutTextEncoderConfig(**text_encoder_config)
    config.text_encoder = text_encoder

    model = LayoutEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)

    assert isinstance(model.decoder, SuryaLayoutDecoder)
    assert isinstance(model.encoder, DonutSwinModel)
    assert isinstance(model.text_encoder, SuryaOCRTextEncoder)

    model = model.to(device)
    model = model.eval()

    print(f"Loaded recognition model {checkpoint} on device {device} with dtype {dtype}")
    return model