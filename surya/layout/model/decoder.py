from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from surya.common.adetr.decoder import SuryaADETRDecoderModel, SuryaADETRDecoderPreTrainedModel
from surya.layout.model.config import LayoutModelOutput
from transformers.modeling_outputs import CausalLMOutput
from surya.settings import settings


class BboxEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cx_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cy_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.xskew_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.yskew_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.label_embed = nn.Embedding(config.label_count, config.hidden_size)

        self.x1_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.y1_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.x2_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.y2_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.x3_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.y3_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.x4_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.y4_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        self.config = config

    def forward(self, boxes: torch.LongTensor, input_box_counts: torch.LongTensor):
        cx, cy, w, h, xskew, yskew, label = boxes.to(torch.long).unbind(dim=-1)

        xskew_actual = ((xskew - self.config.bbox_size // 2) / 2).to(torch.long)
        yskew_actual = ((yskew - self.config.bbox_size // 2) / 2).to(torch.long)

        x1 = (cx - w // 2 - xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y1 = (cy - h // 2 - yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        x2 = (cx + w // 2 - xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y2 = (cy + h // 2 + yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        x3 = (cx + w // 2 + xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y3 = (cy + h // 2 + yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        x4 = (cx - w // 2 + xskew_actual).clamp(0, self.config.bbox_size).to(torch.long)
        y4 = (cy - h // 2 - yskew_actual).clamp(0, self.config.bbox_size).to(torch.long)

        label_embeds = self.label_embed(label)
        size_embeds = self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy)
        skew_embeds = self.xskew_embed(xskew) + self.yskew_embed(yskew)
        corner_embeds = self.x1_embed(x1) + self.y1_embed(y1) + self.x2_embed(x2) + self.y2_embed(y2) + self.x3_embed(x3) + self.y3_embed(y3) + self.x4_embed(x4) + self.y4_embed(y4)
        embedded = label_embeds + size_embeds + skew_embeds + corner_embeds

        return embedded


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
        self.pre_output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
            input_ids=input_boxes,
            input_boxes_counts=input_boxes_counts,
            cache_position=cache_position,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = self.pre_output_norm(outputs[0])
        class_logits = self.lm_head(hidden_states)
        bbox_logits = F.sigmoid(self.bbox_head(hidden_states))

        return LayoutModelOutput(
            bbox_logits=bbox_logits,
            class_logits=class_logits,
            hidden_states=outputs.hidden_states,
        )