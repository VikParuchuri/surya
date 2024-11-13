import torch

from surya.model.layout.encoderdecoder import SuryaLayoutModel
from surya.model.layout.config import SuryaLayoutConfig, SuryaLayoutDecoderConfig, DonutSwinLayoutConfig
from surya.settings import settings


def load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE) -> SuryaLayoutModel:
    config = SuryaLayoutConfig.from_pretrained(checkpoint)
    decoder_config = config.decoder
    decoder = SuryaLayoutDecoderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = config.encoder
    encoder = DonutSwinLayoutConfig(**encoder_config)
    config.encoder = encoder

    model = SuryaLayoutModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)
    model = model.to(device)
    model = model.eval()

    if settings.LAYOUT_STATIC_CACHE:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 16
        torch._dynamo.config.suppress_errors = False

        print(f"Compiling layout model {checkpoint} on device {device} with dtype {dtype}")
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)

    print(f"Loaded layout model {checkpoint} on device {device} with dtype {dtype}")
    return model
