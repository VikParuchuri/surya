import warnings

import torch

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from typing import List, Optional, Tuple
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, AutoModel, AutoModelForCausalLM
from surya.model.recognition.config import MBartMoEConfig, VariableDonutSwinConfig
from surya.model.recognition.encoder import VariableDonutSwinModel
from surya.model.recognition.decoder import MBartMoE
from surya.settings import settings


def load_model(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE, langs: Optional[List[int]] = None):
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint)

    # Prune moe experts that are not needed before loading the model
    if langs is not None:
        config.decoder.langs = {lang_iso : lang_int for lang_iso, lang_int in config.decoder.langs.items() if lang_int in langs}

    decoder_config = vars(config.decoder)
    decoder = MBartMoEConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder

    # Get transformers to load custom encoder/decoder
    AutoModel.register(MBartMoEConfig, MBartMoE)
    AutoModelForCausalLM.register(MBartMoEConfig, MBartMoE)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    model = LangVisionEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)
    assert isinstance(model.decoder, MBartMoE)
    assert isinstance(model.encoder, VariableDonutSwinModel)

    model = model.to(device)
    model = model.eval()
    print(f"Loaded recognition model {checkpoint} on device {device} with dtype {dtype}")
    return model


class LangVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def prepare_inputs_for_generation(
            self, input_ids, decoder_langs=None, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, langs=decoder_langs, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
            "decoder_langs": decoder_inputs["langs"],
        }
        return input_dict

