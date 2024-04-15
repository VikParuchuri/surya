from transformers import DetrConfig, BeitConfig, DetrImageProcessor, VisionEncoderDecoderConfig, AutoModelForCausalLM, \
    AutoModel
from surya.model.ordering.config import MBartOrderConfig, VariableDonutSwinConfig
from surya.model.ordering.decoder import MBartOrder
from surya.model.ordering.encoder import VariableDonutSwinModel
from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.model.ordering.processor import OrderImageProcessor
from surya.settings import settings


def load_model(checkpoint=settings.ORDER_MODEL_CHECKPOINT):
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint)

    decoder_config = vars(config.decoder)
    decoder = MBartOrderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder

    # Get transformers to load custom model
    AutoModel.register(MBartOrderConfig, MBartOrder)
    AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    model = OrderVisionEncoderDecoderModel.from_pretrained(checkpoint, config=config)
    assert isinstance(model.decoder, MBartOrder)
    assert isinstance(model.encoder, VariableDonutSwinModel)

    return model