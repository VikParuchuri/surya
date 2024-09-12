from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithNoAttention
from transformers.utils import ModelOutput

from surya.model.layout.config import SuryaLayoutConfig
from surya.model.recognition.decoder import SuryaOCRDecoderPreTrainedModel, SuryaOCRDecoderLayer, SuryaOCRDecoderRMSNorm
from surya.settings import settings


class BboxEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.x1_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.y1_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.x2_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.y2_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.w_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.h_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.cx_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.cy_embed = nn.Embedding(config.bbox_size, config.hidden_size)
        self.label_embed = nn.Embedding(config.label_count, config.hidden_size)
        self.box_size = config.bbox_size

    def forward(self, boxes: torch.LongTensor):
        cx, cy, w, h, label = boxes.unbind(dim=-1)
        # Shape is (batch_size, num_boxes/seq len, d_model)
        x1 = (cx - w // 2).long()
        y1 = (cy - h // 2).long()
        x2 = (cx + w // 2).long()
        y2 = (cy + h // 2).long()
        x1 = torch.clamp(x1, 0, self.box_size - 1)
        y1 = torch.clamp(y1, 0, self.box_size - 1)
        x2 = torch.clamp(x2, 0, self.box_size - 1)
        y2 = torch.clamp(y2, 0, self.box_size - 1)

        coord_embeds = self.x1_embed(x1) + self.y1_embed(y1) + self.x2_embed(x2) + self.y2_embed(y2)
        label_embeds = self.label_embed(label)
        size_embeds = self.w_embed(w) + self.h_embed(h) + self.cx_embed(cx) + self.cy_embed(cy)
        embedded = coord_embeds + label_embeds + size_embeds

        return embedded


class SuryaLayoutModel(SuryaOCRDecoderPreTrainedModel):
    def __init__(self, config: SuryaLayoutConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = BboxEmbedding(config)
        self.layers = nn.ModuleList(
            [SuryaOCRDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = SuryaOCRDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.causal = config.causal

        self.register_buffer(
            "normalizer", torch.tensor(self.config.hidden_size**0.5, dtype=torch.float32), persistent=False
        )
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_boxes: torch.LongTensor = None,
        input_boxes_counts: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # Embed the input ids in this case
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_boxes)

        hidden_states = inputs_embeds

        if use_cache and inputs_embeds.shape[1] != 1:
            self._setup_cache(self.config, hidden_states.shape[0], hidden_states.device, hidden_states.dtype)

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        all_hidden_states = () if output_hidden_states else None
        for i, residual_block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    residual_block.__call__, hidden_states, position_ids, causal_mask, encoder_hidden_states, encoder_attention_mask, cache_position, use_cache
                )
            else:
                hidden_states = residual_block(hidden_states, position_ids, causal_mask, encoder_hidden_states, encoder_attention_mask, cache_position, use_cache)

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    # Ignore copy
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if not self.causal:
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = max(settings.LAYOUT_MAX_TOKENS, sequence_length)

        diagonal = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = diagonal
        if sequence_length != 1:
            # Select the upper triangular part of the matrix, but unmask current token (the diagonal)
            # triu will be the min_dtype, everything else is 0 (attended to)
            causal_mask = torch.triu(diagonal, diagonal=1)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                # Mask positions in the causal mask that are masked in the attention mask
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if attention_mask is not None and attention_mask.device.type == "cuda":
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


@dataclass
class LayoutModelOutput(ModelOutput):
    bbox_logits: torch.Tensor
    class_logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class SuryaLayoutDecoder(SuryaOCRDecoderPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SuryaLayoutModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.label_count, bias=False)
        self.bbox_head = nn.Linear(config.hidden_size, 4, bias=False)

        self.bbox_size = config.bbox_size
        self.label_count = config.label_count
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
        bbox_logits = self.bbox_head(hidden_states)

        return LayoutModelOutput(
            bbox_logits=bbox_logits,
            class_logits=class_logits,
            hidden_states=outputs.hidden_states,
        )

