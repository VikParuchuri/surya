import torch
from transformers import DynamicCache, HQQQuantizedCache, StaticCache
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F


class ContinuousBatchingMixin:
    def pad_left(
        self, key_states: torch.Tensor, value_states: torch.Tensor, padding_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Size is assumed to be (batch_size, num_kv_heads, seq_length, head_dim) - To match huggingface
        key_states_padded = F.pad(
            key_states,
            pad=(
                0,
                0,
                padding_size,
                0,
            ),  # (left, right, top, bottom) for last two dimensions
            mode="constant",
            value=0,
        )

        value_states_padded = F.pad(
            value_states,
            pad=(
                0,
                0,
                padding_size,
                0,
            ),  # (left, right, top, bottom) for last two dimensions
            mode="constant",
            value=0,
        )

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

    def merge(
        self,
        new_cache: "ContinuousBatchingCache",
        merge_idxs: List[int],
        device: torch.device,
    ):
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

        merge_idxs = torch.tensor(merge_idxs, dtype=torch.long, device=device)

        with torch.inference_mode():
            # As long as we set the attention mask and position ids correctly, padding value can be anything
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

                adjusted_key_cache.index_put_((merge_idxs,), new_k)
                adjusted_value_cache.index_put_((merge_idxs,), new_v)

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
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Updates the static cache at specific cache positions for each batch element.

        Args:
            key_states: (batch_size, num_kv_heads, seq_len, head_dim)
            value_states: same shape as key_states
            cache_position: (batch_size, seq_len)
        """
        cache_position = cache_kwargs.get('cache_position', None)
        assert cache_position is not None, "`cache_position` cannot be None for static cache"
        bsz, num_heads, seq_len, head_dim = key_states.shape

        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]

        new_k = key_states
        new_v = value_states

        for b in range(bsz):
            pos = cache_position[b]  # (seq_len,)
            k_cache[b, :, pos, :] = new_k[b]
            v_cache[b, :, pos, :] = new_v[b]

        self.key_cache[layer_idx] = k_cache
        self.value_cache[layer_idx] = v_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def merge(self, *args, **kwargs):
            raise NotImplementedError("No merging support with static cache yet!")