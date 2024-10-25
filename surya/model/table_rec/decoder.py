from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers.utils import ModelOutput

from surya.model.table_rec.config import SuryaTableRecDecoderConfig, SuryaTableRecTextEncoderConfig
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from surya.settings import settings

_MAX_SQRT_GRADIENT = 1000.0

@dataclass
class TableRecModelOutput(ModelOutput):
    bbox_logits: torch.Tensor
    class_logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class SuryaTableRecDecoderRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst SuryaTableRecDecoder is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


ALL_LAYERNORM_LAYERS.append(SuryaTableRecDecoderRMSNorm)


class SuryaTableRecDecoderRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    # Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding.forward with Gemma->SuryaTableRecDecoder
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SuryaTableRecDecoderSdpaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    Modified for GQA
    """

    def __init__(self, config: SuryaTableRecDecoderConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.config.encoder_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.config.encoder_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=True)
        self.rotary_emb = SuryaTableRecDecoderRotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Encoder attention mask currently ignored

        bsz, q_len, _ = hidden_states.size()
        _, v_len, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        if self.key_states is None:
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            key_states = key_states.view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if use_cache:
                self._update_cache(key_states, value_states)
        else:
            key_states = self.key_states
            value_states = self.value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.head_dim**-0.5,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _setup_cache(self, batch_size, device, dtype=None):
        # Setup initial caches
        self.value_states = None
        self.key_states = None

    @torch.no_grad()
    def _update_cache(self, key_states, value_states, **cache_kwargs):
        self.value_states = value_states
        self.key_states = key_states


class SuryaTableRecDecoderSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SuryaTableRecDecoderConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=True)
        self.rotary_emb = SuryaTableRecDecoderRotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        window_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Final is bsz, num_attention_heads, seq_len, head_dim
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if use_cache and hasattr(self, "key_states"):
            cache_kwargs = {"cache_position": cache_position, "window_attn": window_attn}
            key_states, value_states = self._update_cache(key_states, value_states, **cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            # Mask is batch, head, seq_len, kv_len
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
            current_cache_position = cache_position[-1].item() if cache_position is not None else None
            if current_cache_position and settings.RECOGNITION_STATIC_CACHE:
                # Mask out future cache positions
                position_mask = torch.ones_like(causal_mask, dtype=torch.bool, device=causal_mask.device)
                position_mask[:, :, :, :current_cache_position + 1] = False
                causal_mask = torch.where(position_mask, torch.finfo(causal_mask.dtype).min, causal_mask)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.head_dim**-0.5,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _setup_cache(self, batch_size, device, dtype=None):
        if dtype is None and self.config.torch_dtype is not None:
            dtype = self.config.torch_dtype
        dtype = dtype if dtype is not None else torch.float32

        # Setup initial caches
        self.value_states = None
        self.key_states = None

        if settings.RECOGNITION_STATIC_CACHE:
            cache_shape = (batch_size, self.num_key_value_heads, settings.RECOGNITION_MAX_TOKENS, self.head_dim)
            self.value_states = torch.zeros(cache_shape, dtype=dtype, device=device)
            self.key_states = torch.zeros(cache_shape, dtype=dtype, device=device)

    def _update_static_cache(self, key_states, value_states, **cache_kwargs):
        cache_position = cache_kwargs.get("cache_position")
        k_out, v_out = self.key_states.to(key_states.device), self.value_states.to(value_states.device)

        k_out[:, :, cache_position] = key_states.to(k_out.dtype)
        v_out[:, :, cache_position] = value_states.to(v_out.dtype)

        self.key_states, self.value_states = k_out, v_out
        return k_out, v_out

    def _update_dynamic_cache(self, key_states, value_states, **cache_kwargs):
        k_out = key_states
        if self.key_states is not None:
            k_out = torch.cat([self.key_states, key_states], dim=2)

        v_out = value_states
        if self.value_states is not None:
            v_out = torch.cat([self.value_states, value_states], dim=2)

        self.key_states, self.value_states = k_out, v_out
        return k_out, v_out

    @torch.no_grad()
    def _update_cache(self, key_states, value_states, **cache_kwargs):
        if settings.RECOGNITION_STATIC_CACHE:
            return self._update_static_cache(key_states, value_states, **cache_kwargs)

        return self._update_dynamic_cache(key_states, value_states, **cache_kwargs)


class SuryaTableRecDecoderMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_activation is None:
            config.hidden_activation = "gelu_pytorch_tanh"
        hidden_activation = config.hidden_activation
        self.act_fn = ACT2FN[hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SuryaTableRecDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        super().__init__()
        self.cross_pre_norm = SuryaTableRecDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.temporal_pre_norm = SuryaTableRecDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.temporal_block = None
        if layer_idx in config.self_attn_layers:
            self.temporal_block = SuryaTableRecDecoderSdpaAttention(config)

        self.cross_attn_block = None
        if layer_idx in config.cross_attn_layers:
            self.cross_attn_block = SuryaTableRecDecoderSdpaCrossAttention(config)

        self.window_attn = layer_idx not in config.global_attn_layers
        self.channel_pre_norm = SuryaTableRecDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_block = SuryaTableRecDecoderMlp(config)

    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        use_cache: bool = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_activations = activations

        if self.cross_attn_block is not None:
            # Do cross-attention on encoder outputs
            cross_attn_inputs = self.cross_pre_norm(activations)
            cross_attn_path = self.cross_attn_block(
                cross_attn_inputs, encoder_hidden_states, attention_mask, encoder_attention_mask, use_cache=use_cache
            )
            cross_attn_output = cross_attn_path + raw_activations
        else:
            cross_attn_output = raw_activations

        if self.temporal_block is not None:
            inputs_normalized = self.temporal_pre_norm(cross_attn_output)  # RMSNorm introduces slight slight differences
            hidden_states = self.temporal_block(
                inputs_normalized, position_ids, attention_mask, cache_position=cache_position, use_cache=use_cache, window_attn=self.window_attn
            )

            residual = hidden_states + raw_activations
        else:
            residual = cross_attn_output

        hidden_states = self.channel_pre_norm(residual)
        hidden_states = self.mlp_block(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states


class SuryaTableRecDecoderPreTrainedModel(PreTrainedModel):
    config_class = SuryaTableRecDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SuryaTableRecDecoderLayer"]
    _skip_keys_device_placement = ["cache"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False  # we can't compare with eager for now
    _supports_cache_class = True
    _supports_quantized_cache = True

    def _init_weights(self, module):
        if isinstance(module, SuryaTableRecDecoderSdpaAttention):
            torch.nn.init.normal_(module.q_proj.weight, mean=0.0, std=self.config.init_std)
            torch.nn.init.normal_(module.k_proj.weight, mean=0.0, std=self.config.init_std)
            torch.nn.init.normal_(module.v_proj.weight, mean=0.0, std=self.config.init_std)

            torch.nn.init.normal_(module.o_proj.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, config, batch, device, dtype):
        layers = getattr(self, "model", self).layers
        for layer in layers:
            if layer.temporal_block:
                layer.temporal_block._setup_cache(batch, device, dtype)
            if layer.cross_attn_block:
                layer.cross_attn_block._setup_cache(batch, device, dtype)

    def reset_cache(self, batch, device, dtype):
        pass

    def _tie_weights(self):
        pass

    def tie_weights(self):
        pass


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


class SuryaTableRecDecoderModel(SuryaTableRecDecoderPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SuryaTableRecDecoderDecoderLayer`]

    Args:
        config: SuryaTableRecDecoderConfig
    """

    def __init__(self, config: SuryaTableRecDecoderConfig, embed_labels=False, embed_positions=True):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.causal = config.causal

        if embed_labels:
            self.embed_tokens = LabelEmbedding(config)
        else:
            self.embed_tokens = BboxEmbedding(config, embed_positions=embed_positions)

        self.layers = nn.ModuleList(
            [SuryaTableRecDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = SuryaTableRecDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

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
        input_ids: torch.LongTensor = None,
        input_boxes_counts: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prefill: bool = False
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        inputs_embeds = self.embed_tokens(input_ids, input_boxes_counts)
        hidden_states = inputs_embeds

        if use_cache and prefill:
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
        target_length = max(settings.TABLE_REC_MAX_BOXES, sequence_length)

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


class SuryaTableRecDecoder(SuryaTableRecDecoderPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SuryaTableRecDecoderModel(config, embed_labels=True, embed_positions=False)
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


class SuryaTableRecTextEncoder(SuryaTableRecDecoderPreTrainedModel):
    _tied_weights_keys = None
    config_class = SuryaTableRecTextEncoderConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SuryaTableRecDecoderModel(config, embed_labels=False, embed_positions=True)
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