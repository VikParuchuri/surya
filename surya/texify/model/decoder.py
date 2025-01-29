from typing import Optional, Union, Tuple

import torch

from surya.common.adetr.decoder import SuryaADETRDecoderPreTrainedModel, SuryaADETRDecoderModel, WrappedEmbedding
from torch import nn

from surya.settings import settings
from surya.texify.model.config import TexifyModelOutput


class TexifyDecoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = WrappedEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.TEXIFY_STATIC_CACHE,
            max_boxes=settings.TEXIFY_MAX_TOKENS
        )
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pre_output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        input_ids: torch.LongTensor = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, TexifyModelOutput]:
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

        hidden_states = self.pre_output_norm(outputs[0])
        logits = self.lm_head(hidden_states)

        return TexifyModelOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
        )