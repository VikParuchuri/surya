from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.utils import ModelOutput

from surya.recognition.model.config import SuryaOCRTextEncoderConfig
from transformers.modeling_outputs import CausalLMOutput
from surya.common.adetr.decoder import SuryaADETRDecoderModel, SuryaADETRDecoderPreTrainedModel, WrappedEmbedding
from surya.settings import settings

_MAX_SQRT_GRADIENT = 1000.0


@dataclass
class OCRModelOutput(ModelOutput):
    logits: torch.Tensor
    aux_logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class SuryaOCRDecoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = WrappedEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.RECOGNITION_STATIC_CACHE,
            max_boxes=settings.RECOGNITION_MAX_TOKENS
        )
        self.vocab_size = config.vocab_size
        aux_heads = config.aux_heads if config.aux_heads is not None else 0
        lm_heads = aux_heads + 1
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size * lm_heads, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        prefill: bool = False,
        **kwargs
    ) -> Union[Tuple, OCRModelOutput]:
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            prefill=prefill,
        )

        hidden_states = outputs[0]
        all_logits = self.lm_head(hidden_states)
        all_logits = torch.split(all_logits, self.vocab_size, dim=-1)
        logits = all_logits[0]
        aux_logits = all_logits[1:] if len(all_logits) > 1 else None

        return OCRModelOutput(
            logits=logits,
            aux_logits=aux_logits,
            hidden_states=outputs.hidden_states,
        )

@dataclass
class TextEncoderOutput(CausalLMOutput):
    hidden_states: torch.FloatTensor = None


class SuryaOCRTextEncoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None
    config_class = SuryaOCRTextEncoderConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = WrappedEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.RECOGNITION_STATIC_CACHE,
            max_boxes=settings.RECOGNITION_MAX_TOKENS
        )
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        return TextEncoderOutput(
            hidden_states=outputs.last_hidden_state,
        )