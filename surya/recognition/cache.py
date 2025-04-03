import torch
from transformers import DynamicCache
from typing import List, Tuple

class ContinuousBatchingCache(DynamicCache):
    def pad_left(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        padding_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Size is assumed to be (batch_size, num_kv_heads, seq_length, head_dim) - To match huggingface
        key_padding = torch.zeros((key_states.shape[0], key_states.shape[1], padding_size, key_states.shape[3]), device=key_states.device, dtype=key_states.dtype)
        key_states_padded = torch.cat([key_padding, key_states], dim=-2)  # Pad along the sequence length dimension (dim=-2)

        # Pad value_states to the left by `padding_size`
        value_padding = torch.zeros((value_states.shape[0], value_states.shape[1], padding_size, value_states.shape[3]), device=value_states.device, dtype=value_states.dtype)
        value_states_padded = torch.cat([value_padding, value_states], dim=-2)  # Pad along the sequence length dimension (dim=-2)

        return key_states_padded, value_states_padded

    # Trim the cache from the left - Useful when longer sequences are evicted and we have long padding on the left
    def trim_left(
        self,
        trim_length: int
    ):
        for layer_idx in range(len(self)):
            # cache sape is (batch_size, num_kv_heads, seq_length, head_dim); Trimming from head dim
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, trim_length:, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, trim_length:, :]

    def merge(
        self, 
        new_cache: DynamicCache,
        merge_idxs: List[int]
    ):
        assert len(new_cache) == len(self), "The two caches should have the same number of layers"
        
        # We should TECHNICALLY be able to pad these values to 0s now, since they will be attention masked
        current_seq_length = self.get_seq_length()
        new_cache_seq_length = new_cache.get_seq_length()
        offset = current_seq_length - new_cache_seq_length      # Generally positive, but negative case is handled too
        with torch.inference_mode():
            # As long as we set the attention mask and position ids correctly, padding value can be anything
            for layer_idx in range(len(self)):
                new_k, new_v = new_cache[layer_idx]
                if offset > 0:
                    new_k, new_v = self.pad_left(new_k, new_v, offset)

                if offset < 0:
                    adjusted_key_cache, adjusted_value_cache = self.pad_left(self.key_cache[layer_idx], self.value_cache[layer_idx], abs(offset))
                else:
                    adjusted_key_cache, adjusted_value_cache = self.key_cache[layer_idx], self.value_cache[layer_idx]

                # TODO Make this assignment batched? 
                for i, merge_idx in enumerate(merge_idxs):
                    adjusted_key_cache[merge_idx] = new_k[i]
                    adjusted_value_cache[merge_idx] = new_v[i]

                    self.key_cache[layer_idx] = adjusted_key_cache
                    self.value_cache[layer_idx] = adjusted_value_cache

        return offset