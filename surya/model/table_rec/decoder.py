import copy
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
from torch import nn
from transformers import MBartForCausalLM, MBartPreTrainedModel
from transformers.utils import ModelOutput

from surya.model.ordering.decoder import MBartOrderDecoderWrapper
from surya.model.table_rec.config import TableRecDecoderConfig


@dataclass
class TableRecDecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    row_logits: torch.FloatTensor = None
    col_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class TableRecDecoder(MBartForCausalLM):
    config_class = TableRecDecoderConfig
    _tied_weights_keys = []

    def __init__(self, config, **kwargs):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        MBartPreTrainedModel.__init__(self, config)
        self.model = MBartOrderDecoderWrapper(config)

        self.row_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.col_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_boxes: torch.LongTensor = None,
        input_boxes_mask: Optional[torch.Tensor] = None,
        input_boxes_counts: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, TableRecDecoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_boxes=input_boxes,
            input_boxes_mask=input_boxes_mask,
            input_boxes_counts=input_boxes_counts,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        row_logits = self.row_head(outputs[0])
        col_logits = self.col_head(outputs[0])

        return TableRecDecoderOutput(
            loss=None,
            col_logits=col_logits,
            row_logits=row_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )