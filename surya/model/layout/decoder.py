import collections
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from surya.model.common.adetr.decoder import SuryaADETRDecoderModel, SuryaADETRDecoderPreTrainedModel, WrappedEmbedding
from surya.model.layout.config import LayoutModelOutput, SuryaLayoutTextEncoderConfig, \
    SuryaLayoutTextEncoderOutput
from transformers.modeling_outputs import CausalLMOutput
from surya.settings import settings


class BboxEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.h_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.cx_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.cy_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.xskew_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.yskew_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.label_embed = nn.Embedding(config.label_count, config.hidden_size)
        self.box_size = config.bbox_size

    def forward(self, boxes: torch.LongTensor, input_boxes_counts: torch.LongTensor):
        cx, cy, w, h, xskew, yskew, label = boxes.to(torch.long).unbind(dim=-1)

        label_embeds = self.label_embed(label)
        size_embeds = self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy)
        skew_embeds = self.xskew_embed(xskew) + self.yskew_embed(yskew)
        embedded = label_embeds + size_embeds + skew_embeds

        return embedded


class SuryaLayoutTextEncoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None
    config_class = SuryaLayoutTextEncoderConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = WrappedEmbedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.LAYOUT_STATIC_CACHE,
            max_boxes=settings.LAYOUT_MAX_BOXES
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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        return SuryaLayoutTextEncoderOutput(
            hidden_states=outputs.last_hidden_state,
        )


class SuryaLayoutDecoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = BboxEmbedding(config)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.LAYOUT_STATIC_CACHE,
            max_boxes=settings.LAYOUT_MAX_BOXES
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.label_count, bias=False)
        self.bbox_head = nn.Linear(config.hidden_size, 6, bias=True)

        self.bbox_size = config.bbox_size
        self.label_count = config.label_count
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
        input_boxes: torch.LongTensor = None,
        input_boxes_counts: torch.LongTensor = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        outputs = self.model(
            input_boxes=input_boxes,
            input_boxes_counts=input_boxes_counts,
            cache_position=cache_position,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs[0]
        class_logits = self.lm_head(hidden_states)
        bbox_logits = F.sigmoid(self.bbox_head(hidden_states))

        return LayoutModelOutput(
            bbox_logits=bbox_logits,
            class_logits=class_logits,
            hidden_states=outputs.hidden_states,
        )