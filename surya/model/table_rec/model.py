from transformers import VisionEncoderDecoderConfig, AutoModel, AutoModelForCausalLM

from surya.model.ordering.config import VariableDonutSwinConfig
from surya.model.ordering.encoder import VariableDonutSwinModel
from surya.model.table_rec.config import TableRecDecoderConfig
from surya.model.table_rec.decoder import TableRecDecoder
from surya.model.table_rec.encoderdecoder import TableRecVisionEncoderDecoderModel
from surya.settings import settings


def load_model(checkpoint=settings.TABLE_REC_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint)

    decoder_config = vars(config.decoder)
    decoder = TableRecDecoderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder

    # Get transformers to load custom model
    AutoModel.register(TableRecDecoderConfig, TableRecDecoder)
    AutoModelForCausalLM.register(TableRecDecoderConfig, TableRecDecoder)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    model = TableRecVisionEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)
    assert isinstance(model.decoder, TableRecDecoder)
    assert isinstance(model.encoder, VariableDonutSwinModel)

    model = model.to(device)
    model = model.eval()
    print(f"Loaded reading order model {checkpoint} on device {device} with dtype {dtype}")
    return model