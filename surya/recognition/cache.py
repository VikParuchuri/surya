import torch
from transformers import DynamicCache
from typing import List, Tuple

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

        with torch.inference_mode():
            # Since we set the attention mask and position ids correctly, padding value can be anything
            for layer_idx in range(len(self)):
                new_k, new_v = new_cache[layer_idx]
                new_k, new_v = self.pad_left(new_k, new_v, max(0, offset))


                adjusted_key_cache, adjusted_value_cache = self.pad_left(
                    self.key_cache[layer_idx], self.value_cache[layer_idx], max(0, -offset)
                )

                merge_idxs_tensor = torch.tensor(merge_idxs)
                adjusted_key_cache.index_copy_(0, merge_idxs_tensor, new_k)
                adjusted_value_cache.index_copy_(0, merge_idxs_tensor, new_v)

                self.key_cache[layer_idx] = adjusted_key_cache
                self.value_cache[layer_idx] = adjusted_value_cache

        return offset