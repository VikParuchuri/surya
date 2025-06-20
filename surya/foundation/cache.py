from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import Cache, StaticCache
from transformers import PretrainedConfig
from transformers.utils import is_torchdynamo_compiling

"""
Special cache class for the surya foundation model that supports - 
1) Static shape
2) A custom sliding window, where image tokens stay in cache, and text tokens are popped
3) Continuous batching - merging etc
4) Attention mask management - To match with what's currently in the cache

Heavily inspired from https://github.com/huggingface/transformers/blob/0725cd6953803b8aacfc85288cbfb83dea30c469/src/transformers/cache_utils.py#L1079
"""
class ContinuousBatchingCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int,
        max_cache_len: int,
        text_sliding_window: int,
        device: int,
        dtype: int
    ):
        # batch_size is deprecated in newer versions
        super().__init__(config, batch_size=None, max_cache_len=max_cache_len, device=device, dtype=dtype, max_batch_size=batch_size)
        self.text_sliding_window = text_sliding_window
        self.num_layers = config.num_hidden_layers

        # TODO Setup these as buffers since its a nn.Module
        self.attention_mask = torch.zeros((self.batch_size, self.max_cache_len), device=device, dtype=torch.int)
        self.text_token_counts = [[0 for _ in range(batch_size)] for _ in range(self.num_layers)]

    def _shift_attention_mask_left(self, batch_idx: int, shift_amount: int):
        self.attention_mask[batch_idx, :-shift_amount] = self.attention_mask[batch_idx, shift_amount:]
        self.attention_mask[batch_idx, -shift_amount:] = 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = cache_kwargs.get('prefill', False)
        if prefill:
            return self._prefill_update(
                key_states,
                value_states,
                layer_idx,
                cache_kwargs
            )
        else:
            return self._decode_update(
                key_states,
                value_states,
                layer_idx,
                cache_kwargs
            )

    def _prefill_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cache_idxs: List[int] = cache_kwargs.get("cache_idx", None)
        assert cache_idxs is not None, "cache_idxs must be specified during prefill"

        # key and value states will already be left padded during prefill
        valid_batch_size = len(cache_idxs)
        self.key_cache[layer_idx][cache_idxs] = key_states[:valid_batch_size]
        self.value_cache[layer_idx][cache_idxs] = value_states[:valid_batch_size]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _decode_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Static cache update
        - adjust attention mask with new left padding
        - respects per-batch text token limits
        - per-batch valid token lengths (right-padded inputs)

        kv states are expected to have shape [batch_size, kv_heads, T_pad, head_dim]
        They may have different `true` lengths, to account for multi token preds, or beacon tokens
        Expects `valid_tokens` in cache_kwargs: a tensor of shape (B,) indicating the number
        of actual (non-padded) tokens to add per batch element.
        """

        valid_tokens: torch.Tensor = cache_kwargs.get("valid_tokens")  # shape: (B,)
        assert valid_tokens is not None, "`valid_tokens` must be provided in `cache_kwargs`"

        # Update only selected batch indices, useful for prefill in continuous batching
        cache_idxs: List[int] = cache_kwargs.get("cache_idx", None)
        if cache_idxs is None:
            cache_idxs = list(range(self.batch_size))

        k_cache = self.key_cache[layer_idx]   # (B, H, L, D)
        v_cache = self.value_cache[layer_idx] # (B, H, L, D)

        for batch_idx, cache_idx in enumerate(cache_idxs):
            new_text_len = valid_tokens[batch_idx].item()
            if new_text_len == 0:
                continue  # skip padded batch entry

            curr_text_cache_len = self.text_token_counts[layer_idx][cache_idx]

            k_new = key_states[batch_idx, :, :new_text_len, :]  # (H, new_text_len, D)
            v_new = value_states[batch_idx, :, :new_text_len, :]

            if curr_text_cache_len + new_text_len <= self.text_sliding_window:
                # If we are under the sliding window length, shift the entire cache left
                # Since we setup the max cache length with enough buffer, this will ONLY drop 
                # left padding tokens out
                shift = new_text_len
                if curr_text_cache_len > 0:
                    k_cache[cache_idx, :, :-shift, :] = k_cache[cache_idx, :, shift:, :]
                    v_cache[cache_idx, :, :-shift, :] = v_cache[cache_idx, :, shift:, :]
                k_cache[cache_idx, :, -shift:, :] = k_new
                v_cache[cache_idx, :, -shift:, :] = v_new

                self.text_token_counts[layer_idx][cache_idx] += new_text_len
                if layer_idx == (self.num_layers - 1):
                    self._shift_attention_mask_left(cache_idx, shift)
            else:
                # Expand text region to exactly text_sliding_window tokens
                # Shift entire cache left to make room for the full sliding window
                
                # Calculate how much to shift left to accommodate full sliding window
                desired_text_start = self.max_cache_len - self.text_sliding_window
                
                # We need to figure out how many text tokens to keep and where to place them
                keep = self.text_sliding_window - new_text_len
                assert keep > 0, "Cannot add more new text tokens than the sliding window"
                
                # Shift entire cache left to make room for full text sliding window
                shift_amount = self.text_sliding_window - curr_text_cache_len
                if shift_amount > 0:        # Cannot be negative, may be exactly 0
                    k_cache[cache_idx :, :-shift_amount, :] = k_cache[cache_idx :, shift_amount:, :]
                    v_cache[cache_idx :, :-shift_amount, :] = v_cache[cache_idx :, shift_amount:, :]

                    if layer_idx == (self.num_layers - 1):
                        self._shift_attention_mask_left(cache_idx, shift_amount)
                
                # Now place the most recent 'keep' text tokens at the start of text region
                old_text_start = self.max_cache_len - curr_text_cache_len - shift_amount
                k_cache[cache_idx :, desired_text_start:desired_text_start + keep, :] = k_cache[cache_idx :, old_text_start + (curr_text_cache_len - keep):old_text_start + curr_text_cache_len, :]
                v_cache[cache_idx :, desired_text_start:desired_text_start + keep, :] = v_cache[cache_idx :, old_text_start + (curr_text_cache_len - keep):old_text_start + curr_text_cache_len, :]
                
                # Add new tokens at the end
                k_cache[cache_idx :, desired_text_start + keep:self.max_cache_len, :] = k_new
                v_cache[cache_idx :, desired_text_start + keep:self.max_cache_len, :] = v_new
                
                self.text_token_counts[layer_idx][cache_idx] = self.text_sliding_window

        self.key_cache[layer_idx] = k_cache
        self.value_cache[layer_idx] = v_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
