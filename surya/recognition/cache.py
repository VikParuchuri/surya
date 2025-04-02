from __future__ import annotations
import torch
from transformers import DynamicCache, StaticCache
from typing import Any, Dict, List, Optional, Tuple

class ContinuousBatchingDynamicCache(DynamicCache):
    def pad_left(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        padding_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_size <= 0:
            return key_states, value_states

        # Size is assumed to be (batch_size, num_kv_heads, seq_length, head_dim) - To match huggingface
        key_padding = torch.zeros((key_states.shape[0], key_states.shape[1], padding_size, key_states.shape[3]), device=key_states.device, dtype=key_states.dtype)
        key_states_padded = torch.cat([key_padding, key_states], dim=-2)  # Pad along the sequence length dimension (dim=-2)

        value_padding = torch.zeros((value_states.shape[0], value_states.shape[1], padding_size, value_states.shape[3]), device=value_states.device, dtype=value_states.dtype)
        value_states_padded = torch.cat([value_padding, value_states], dim=-2)  # Pad along the sequence length dimension (dim=-2)

        return key_states_padded, value_states_padded

    def merge(
        self, 
        new_cache: DynamicCache,
        merge_idxs: List[int]
    ):
        assert len(new_cache) == len(self), "The two caches should have the same number of layers"
        
        current_seq_length = self.get_seq_length()
        new_cache_seq_length = new_cache.get_seq_length()
        offset = current_seq_length - new_cache_seq_length

        merge_idxs_tensor = torch.tensor(merge_idxs, device=self.key_cache[0].device)

        with torch.inference_mode():
            # Since we set the attention mask and position ids correctly, padding value can be anything
            for layer_idx in range(len(self)):
                new_k, new_v = new_cache[layer_idx]
                new_k, new_v = self.pad_left(new_k, new_v, max(0, offset))


                adjusted_key_cache, adjusted_value_cache = self.pad_left(
                    self.key_cache[layer_idx], self.value_cache[layer_idx], max(0, -offset)
                )

                adjusted_key_cache[merge_idxs_tensor] = new_k
                adjusted_value_cache[merge_idxs_tensor] = new_v

                self.key_cache[layer_idx] = adjusted_key_cache
                self.value_cache[layer_idx] = adjusted_value_cache

        return offset


class ContinuousBatchingStaticCache(StaticCache):
    # Modified version of StaticCache.update for continuous batching
    # HF implementation assumes that cache_position is the same across all batches when updating cache which doesn't apply in this case
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        cache_position = cache_kwargs.get("cache_position")  # Shape: (batch_size, seq_len)

        k_out = self.key_cache[layer_idx]  # Shape: (batch, num_heads, max_seq_len, head_dim)
        v_out = self.value_cache[layer_idx]

        key_states = key_states.to(k_out.dtype)  # Ensure dtype consistency
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            batch_size = key_states.shape[0]
            cache_position = cache_position.to(dtype=torch.long)  # Ensure integer indices

            # Loop over each batch element only
            for b in range(batch_size):
                pos = cache_position[b]  # (seq_len,) -> index positions for this batch
                k_out[b, :, pos, :] = key_states[b]
                v_out[b, :, pos, :] = value_states[b]

        return k_out, v_out

    def __len__(self) -> int:
        return len(self.key_cache)

    # We return the max seq length among all slots
    @torch._dynamo.disable
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        all_seq_lengths = self.key_cache[layer_idx][:, 0, :, :].sum(-1)
        return all_seq_lengths.max()

    def merge(
        self,
        new_cache: ContinuousBatchingStaticCache,
        merge_idxs: List[int]
    ):

        assert len(new_cache) == len(self), "The two caches should have the same number of layers"
        assert new_cache.key_cache[0].shape == self.key_cache[0].shape, "The two caches should have the same shape"
        num_new_elements = len(merge_idxs)
        merge_idxs_tensor = torch.tensor(merge_idxs, device=self.key_cache[0].device)

        with torch.inference_mode():
            for layer_idx in range(len(self)):
                # Merge idxs can be of less length than the prefill cache, since prefill cache may have padded batch elements
                new_k, new_v = new_cache.key_cache[layer_idx], new_cache.value_cache[layer_idx]
                new_k, new_v = new_k[:num_new_elements], new_v[:num_new_elements]

                # In the static batching case, both prefill and current cache have the exact same shape, so no padding
                self.key_cache[layer_idx][merge_idxs_tensor] = new_k
                self.value_cache[layer_idx][merge_idxs_tensor] = new_v