from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from surya.model.common.adetr.decoder import SuryaADETRDecoderModel, SuryaADETRDecoderPreTrainedModel
from surya.model.table_rec.config import TableRecModelOutput, SuryaTableRecTextEncoderConfig
from surya.settings import settings


class LabelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.x1_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.y1_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.x2_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.y2_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.w_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.h_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.cx_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.cy_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.class_embed = nn.Embedding(config.max_classes, config.hidden_size)
        self.max_width = config.max_width
        self.max_height = config.max_height
        self.max_classes = config.max_classes

    def forward(self, labels: torch.LongTensor, input_box_counts: torch.LongTensor):
        cx, cy, w, h, class_ = labels.to(torch.long).unbind(dim=-1)
        # Shape is (batch_size, num_boxes/seq len, d_model)
        x1 = (cx - w // 2).long()
        y1 = (cy - h // 2).long()
        x2 = (cx + w // 2).long()
        y2 = (cy + h // 2).long()
        x1 = torch.clamp(x1, 0, self.max_width - 1)
        y1 = torch.clamp(y1, 0, self.max_height - 1)
        x2 = torch.clamp(x2, 0, self.max_width - 1)
        y2 = torch.clamp(y2, 0, self.max_height - 1)

        class_ = torch.clamp(class_, 0, self.max_classes - 1).long()

        w = torch.clamp(w, 0, self.max_width - 1).long()
        h = torch.clamp(h, 0, self.max_height - 1).long()
        cx = torch.clamp(cx, 0, self.max_width - 1).long()
        cy = torch.clamp(cy, 0, self.max_height - 1).long()

        coord_embeds = self.x1_embed(x1) + self.y1_embed(y1) + self.x2_embed(x2) + self.y2_embed(y2)
        class_embeds = self.class_embed(class_)
        embedded = coord_embeds + self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy) + class_embeds

        return embedded


class BboxEmbedding(nn.Module):
    def __init__(self, config, embed_positions=False):
        super().__init__()
        self.x1_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.y1_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.x2_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.y2_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.w_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.h_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.cx_embed = nn.Embedding(config.max_width, config.hidden_size)
        self.cy_embed = nn.Embedding(config.max_height, config.hidden_size)
        self.box_pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.max_width = config.max_width
        self.max_height = config.max_height
        self.embed_positions = embed_positions

    def forward(self, boxes: torch.LongTensor, input_box_counts: torch.LongTensor):
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        x1 = torch.clamp(x1, 0, self.max_width - 1).long()
        y1 = torch.clamp(y1, 0, self.max_height - 1).long()
        x2 = torch.clamp(x2, 0, self.max_width - 1).long()
        y2 = torch.clamp(y2, 0, self.max_height - 1).long()

        # Shape is (batch_size, num_boxes/seq len, d_model)
        w = x2 - x1
        h = y2 - y1
        # Center x and y in torch long tensors
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cx = cx.long()
        cy = cy.long()

        w = torch.clamp(w, 0, self.max_width - 1).long()
        h = torch.clamp(h, 0, self.max_height - 1).long()
        cx = torch.clamp(cx, 0, self.max_width - 1).long()
        cy = torch.clamp(cy, 0, self.max_height - 1).long()

        coord_embeds = self.x1_embed(x1) + self.y1_embed(y1) + self.x2_embed(x2) + self.y2_embed(y2)
        embedded = coord_embeds + self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy)

        # Add in positional embeddings for the boxes and labels
        if self.embed_positions:
            for j in range(embedded.shape[0]):
                box_start = input_box_counts[j, 0]
                box_end = input_box_counts[j, 1] - 1 # Skip the sep token
                box_count = box_end - box_start
                embedded[j, box_start:box_end] = embedded[j, box_start:box_end] + self.box_pos_embed.weight[:box_count]

        return embedded


class SuryaTableRecDecoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = LabelEmbedding(config)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.TABLE_REC_STATIC_CACHE,
            max_boxes=settings.TABLE_REC_MAX_BOXES
        )
        self.vocab_size = config.vocab_size

        self.bbox_head = nn.Linear(config.hidden_size, config.max_width * 4, bias=False)
        self.class_head = nn.Linear(config.hidden_size, config.max_classes, bias=False)
        self.max_width = config.max_width

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
    ) -> Union[Tuple, TableRecModelOutput]:
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
        bbox_logits = self.bbox_head(hidden_states)
        class_logits = self.class_head(hidden_states)
        bsz, seq_len = class_logits.shape[:2]
        bbox_logits = bbox_logits.view(bsz, seq_len, 4, self.max_width)

        return TableRecModelOutput(
            bbox_logits=bbox_logits,
            class_logits=class_logits,
            hidden_states=hidden_states,
        )
@dataclass
class TextEncoderOutput(CausalLMOutput):
    hidden_states: torch.FloatTensor = None


class SuryaTableRecTextEncoder(SuryaADETRDecoderPreTrainedModel):
    _tied_weights_keys = None
    config_class = SuryaTableRecTextEncoderConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        embed_tokens = BboxEmbedding(config, embed_positions=True)
        self.model = SuryaADETRDecoderModel(
            config,
            embedder=embed_tokens,
            static_cache=settings.TABLE_REC_STATIC_CACHE,
            max_boxes=settings.TABLE_REC_MAX_BOXES
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
        input_boxes: Optional[torch.LongTensor] = None,
        input_boxes_counts: Optional[torch.LongTensor] = None,
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

        return TextEncoderOutput(
            hidden_states=outputs.last_hidden_state,
        )