from __future__ import annotations
import torch
from transformers import DynamicCache, StaticCache, HQQQuantizedCache
from typing import Any, Dict, List, Optional, Tuple

class ContinuousBatchingMixin:
    def pad_left(
        self, key_states: torch.Tensor, value_states: torch.Tensor, padding_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_size <= 0:
            return key_states, value_states

        # Size is assumed to be (batch_size, num_kv_heads, seq_length, head_dim) - To match huggingface
        key_padding = torch.zeros(
            (
                key_states.shape[0],
                key_states.shape[1],
                padding_size,
                key_states.shape[3],
            ),
            device=key_states.device,
            dtype=key_states.dtype,
        )
        key_states_padded = torch.cat(
            [key_padding, key_states], dim=-2
        )  # Pad along the sequence length dimension (dim=-2)

        # Pad value_states to the left by `padding_size`
        value_padding = torch.zeros(
            (
                value_states.shape[0],
                value_states.shape[1],
                padding_size,
                value_states.shape[3],
            ),
            device=value_states.device,
            dtype=value_states.dtype,
        )
        value_states_padded = torch.cat(
            [value_padding, value_states], dim=-2
        )  # Pad along the sequence length dimension (dim=-2)

        return key_states_padded, value_states_padded

    # Trim the cache from the left - Useful when longer sequences are evicted and we have long padding on the left
    def trim_left(self, trim_length: int):
        self._seen_tokens -= trim_length
        for layer_idx in range(len(self)):
            # cache sape is (batch_size, num_kv_heads, seq_length, head_dim); Trimming from head dim
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, trim_length:, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][
                :, :, trim_length:, :
            ]

    def get_full_cache(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def set_full_cache(
        self, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor
    ):
        self.key_cache[layer_idx] = key_cache
        self.value_cache[layer_idx] = value_cache

    def merge(self, new_cache: "ContinuousBatchingCache", merge_idxs: List[int]):
        assert len(new_cache) == len(self), (
            "The two caches should have the same number of layers"
        )

        # We should TECHNICALLY be able to pad these values to 0s now, since they will be attention masked
        current_seq_length = self.get_seq_length()
        new_cache_seq_length = new_cache.get_seq_length()
        offset = (
            current_seq_length - new_cache_seq_length
        )  # Generally positive, but negative case is handled too
        if offset > 0:
            new_cache._seen_tokens += offset
        elif offset < 0:
            self._seen_tokens += abs(offset)

        with torch.inference_mode():
            # Since we set the attention mask and position ids correctly, padding value can be anything
            for layer_idx in range(len(self)):
                new_k, new_v = new_cache.get_full_cache(layer_idx)
                if offset > 0:
                    new_k, new_v = self.pad_left(new_k, new_v, offset)

                old_k, old_v = self.get_full_cache(layer_idx)
                if offset < 0:
                    adjusted_key_cache, adjusted_value_cache = self.pad_left(
                        old_k,
                        old_v,
                        abs(offset),
                    )
                else:
                    adjusted_key_cache, adjusted_value_cache = (
                        old_k,
                        old_v,
                    )

                adjusted_key_cache[merge_idxs] = new_k[list(range(len(merge_idxs)))]
                adjusted_value_cache[merge_idxs] = new_v[list(range(len(merge_idxs)))]

                self.set_full_cache(layer_idx, adjusted_key_cache, adjusted_value_cache)

        return offset


class ContinuousBatchingCache(ContinuousBatchingMixin, DynamicCache):
    pass


class ContinuousBatchingQuantizedCache(ContinuousBatchingMixin, HQQQuantizedCache):
    def get_full_cache(self, layer_idx: int):
        unquant_key_cache = self.key_cache[layer_idx]
        unquant_value_cache = self.value_cache[layer_idx]
        quant_key_cache = self._dequantize(self._quantized_key_cache[layer_idx])
        quant_value_cache = self._dequantize(self._quantized_value_cache[layer_idx])

        # Concatenate the unquantized and quantized caches
        full_key_cache = torch.cat([quant_key_cache, unquant_key_cache], dim=-2)
        full_value_cache = torch.cat([quant_value_cache, unquant_value_cache], dim=-2)

        return full_key_cache, full_value_cache

    def set_full_cache(
        self, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor
    ):
        if key_cache.shape[-2] < self.residual_length:
            # HF quantized cache is setup so prefill is always quantized
            # So we treat this as a prefill case
            self.key_cache[layer_idx] = torch.zeros(
                0, dtype=key_cache.dtype, device=key_cache.device
            )
            self.value_cache[layer_idx] = torch.zeros(
                0, dtype=value_cache.dtype, device=value_cache.device
            )
            self._quantized_key_cache[layer_idx] = self._quantize(
                key_cache.contiguous(), axis=self.axis_key
            )
            self._quantized_value_cache[layer_idx] = self._quantize(
                value_cache.contiguous(), axis=self.axis_value
            )
        else:
            self.key_cache[layer_idx] = key_cache[:, :, self.residual_length :, :]
            self.value_cache[layer_idx] = value_cache[:, :, self.residual_length :, :]

            # Quantize the new cache
            quant_key_cache = key_cache[:, :, : self.residual_length, :]
            quant_value_cache = value_cache[:, :, : self.residual_length, :]
            quant_key_cache = self._quantize(quant_key_cache, axis=self.axis_key)
            quant_value_cache = self._quantize(quant_value_cache, axis=self.axis_value)

            # Set the quantized cache
            self._quantized_key_cache[layer_idx] = quant_key_cache
            self._quantized_value_cache[layer_idx] = quant_value_cache

    def trim_left(self, trim_length: int):
        if trim_length == 0:
            return

        self._seen_tokens -= trim_length
        to_keep = self._seen_tokens - trim_length
        quantized_to_keep = to_keep - self.residual_length

        for layer_idx in range(len(self.key_cache)):
            if quantized_to_keep > 0:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])[
                    :, :, trim_length:, :
                ]
                dequant_value = self._dequantize(
                    self._quantized_value_cache[layer_idx]
                )[:, :, trim_length:, :]
                self._quantized_key_cache[layer_idx] = self._quantize(
                    dequant_key, axis=self.axis_key
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    dequant_value, axis=self.axis_value
                )
            else:
                main_to_keep = self._seen_tokens - trim_length
                main_start_idx = self.residual_length - main_to_keep

                full_key_cache = self.key_cache[layer_idx][:, :, main_start_idx:, :]
                full_value_cache = self.value_cache[layer_idx][:, :, main_start_idx:, :]

                self.set_full_cache(layer_idx, full_key_cache, full_value_cache)

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