from surya.model.table_rec.encoder import DonutSwinModel
from surya.model.table_rec.config import SuryaTableRecConfig, SuryaTableRecDecoderConfig, DonutSwinTableRecConfig, \
    SuryaTableRecTextEncoderConfig
from surya.model.table_rec.decoder import SuryaTableRecDecoder, SuryaTableRecTextEncoder
from surya.model.table_rec.encoderdecoder import TableRecEncoderDecoderModel
from surya.settings import settings

import torch


def load_model(checkpoint=settings.TABLE_REC_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE) -> TableRecEncoderDecoderModel:

    config = SuryaTableRecConfig.from_pretrained(checkpoint)
    decoder_config = config.decoder
    decoder = SuryaTableRecDecoderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = config.encoder
    encoder = DonutSwinTableRecConfig(**encoder_config)
    config.encoder = encoder

    model = TableRecEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)

    assert isinstance(model.decoder, SuryaTableRecDecoder)
    assert isinstance(model.encoder, DonutSwinModel)

    model = model.to(device)
    model = model.eval()
    
    if settings.TABLE_REC_STATIC_CACHE:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 16
        torch._dynamo.config.suppress_errors = False
        
        print(f"Compiling table recognition model {checkpoint} on device {device} with dtype {dtype}")
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)

    print(f"Loaded table recognition model {checkpoint} on device {device} with dtype {dtype}")
    return model