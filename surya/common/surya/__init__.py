from typing import Optional, TypedDict
import warnings
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.config import SuryaModelConfig
from surya.common.surya.decoder import SuryaDecoderModel
from surya.common.surya.embedder import SimpleTokenEmbedder
from surya.common.surya.encoder import SuryaEncoderModel

from transformers.utils import is_flash_attn_2_available

from surya.logging import get_logger

if is_flash_attn_2_available():
    from surya.common.surya.flash_attn_utils import _get_unpad_data

logger = get_logger()


@dataclass
class SuryaModelOutput(CausalLMOutputWithPast):
    bbox_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumlative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


class KwargsForCausalLM(FlashAttentionKwargs): ...


class SuryaModel(S3DownloaderMixin, PreTrainedModel):
    config_class = SuryaModelConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    main_input_name = "input_ids"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: SuryaModelConfig,
        embedder: SimpleTokenEmbedder = None,
        vision_encoder: SuryaEncoderModel = None,
        decoder: SuryaDecoderModel = None,
    ):
        super().__init__(config)

        if vision_encoder is None:
            vision_encoder = SuryaEncoderModel(config.vision_encoder)

        if decoder is None:
            decoder = SuryaDecoderModel(config.decoder)

        if embedder is None:
            embedder = SimpleTokenEmbedder(config)

        self.vision_encoder = vision_encoder
        self.decoder = decoder
        self.embedder = embedder

        # Simple encoding for image patches
        self.img_w_embed = nn.Embedding(
            self.config.image_embed_encoding_size,
            self.config.hidden_size,
        )

        self.img_h_embed = nn.Embedding(
            self.config.image_embed_encoding_size,
            self.config.hidden_size,
        )

        # Tying configs
        self.vision_encoder.config = self.config.vision_encoder
        self.decoder.config = self.config.decoder

        self.bbox_head = nn.Linear(config.hidden_size, 6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def tie_weights(self):
        self._tie_weights()

    def _tie_weights(self):
        # Tie weights of lm head and token embedder
        self._tie_or_clone_weights(self.lm_head, self.embedder.token_embed)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.embedder.token_embed

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.embedder.token_embed = new_embeddings

    def get_image_embeddings(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        encoder_chunk_size: int | None,
    ):
        # embed all images with the vision encoder after they have already been tiled and flattened into a single batch
        chunks = [0]
        grid_chunks = [0]
        curr_chunk_len = 0
        curr_seq_len = 0
        for i in range(len(grid_thw)):
            curr_chunk_len += (grid_thw[i][0] * grid_thw[i][1] * grid_thw[i][2]).item()
            if curr_chunk_len > encoder_chunk_size:
                chunks.append(curr_chunk_len + curr_seq_len)
                curr_seq_len += curr_chunk_len
                curr_chunk_len = 0
                grid_chunks.append(i + 1)

        if curr_chunk_len > 0:
            chunks.append(pixel_values.shape[0])
            grid_chunks.append(len(grid_thw))

        assert curr_chunk_len + curr_seq_len == pixel_values.shape[0], (
            f"Mismatch in encoder chunking, {curr_chunk_len} + {curr_seq_len} != {pixel_values.shape[0]}"
        )

        logger.debug(
            f"Chunking encoder sequence into {len(chunks) - 1} chunks of size {encoder_chunk_size} with lengths {chunks} and grids {grid_chunks}"
        )
        embeddings = []
        for i in range(len(chunks) - 1):
            start = chunks[i]
            end = chunks[i + 1]
            grid_start = grid_chunks[i]
            grid_end = grid_chunks[i + 1]
            chunk_embeddings = self.vision_encoder.embed_images(
                image_batch=pixel_values[start:end],
                grid_thw=grid_thw[grid_start:grid_end],
            )
            embeddings.append(chunk_embeddings)

        if len(embeddings) == 0:
            raise ValueError(
                "No image embeddings were generated. Check the input images and grid sizes."
            )
        elif len(embeddings) == 1:
            embeddings = embeddings[0]
        else:
            embeddings = torch.cat(embeddings, dim=0)

        encoding_2d = self.get_2d_learned_embeddings(
            grid_thw,
            device=embeddings.device,
            bbox_size=self.config.image_embed_encoding_multiplier,
        )
        assert embeddings.shape[0] == encoding_2d.shape[0], (
            f"Mismatch in image embedding seq len: {embeddings.shape} vs {encoding_2d.shape}"
        )
        assert embeddings.shape[1] == encoding_2d.shape[1], (
            f"Mismatch in image embedding token counts: {embeddings.shape} vs {encoding_2d.shape}"
        )

        embeddings = embeddings + encoding_2d

        return embeddings

    def embed_ids_boxes_images(
        self, input_ids, pixel_values, grid_thw, encoder_chunk_size: int
    ):
        """
        Insert embedded image tiles into the corresponding positions into the full input sequence

        Positions to insert new tokens are indicated by the special image token index
        """
        # This is batched in the inner call
        inputs_embeds = self.embedder.embed(input_tokens=input_ids)
        if pixel_values is not None:
            image_features = self.get_image_embeddings(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                encoder_chunk_size=encoder_chunk_size,
            )

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = torch.sum((input_ids == self.config.image_token_id))
                n_image_features = image_features.shape[0] * image_features.shape[1]
                warnings.warn(
                    f"Image features and image tokens do not match: tokens {n_image_tokens}, features {n_image_features}. This may lead to unexpected results"
                )
            image_features = image_features.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )
        else:
            assert (input_ids == self.config.image_token_id).sum() == 0, (
                "Image tokens were present in the input but no input images were provided"
            )

        return inputs_embeds

    def get_2d_learned_embeddings(
        self,
        grid_thw,
        device: str | torch.device = "cpu",
        bbox_size: int = 256,
    ):
        all_embeddings = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.config.merge_size,
                grid_w // self.config.merge_size,
            )

            # Scale to 0-1024
            llm_grid_h = (
                torch.arange(llm_grid_h, device=device)
                / max(1, (llm_grid_h - 1))
                * bbox_size
            )
            llm_grid_w = (
                torch.arange(llm_grid_w, device=device)
                / max(1, (llm_grid_w - 1))
                * bbox_size
            )

            llm_grid_w_idx = llm_grid_w.to(torch.long)
            llm_grid_h_idx = llm_grid_h.to(torch.long)

            llm_grid_w = self.img_w_embed(llm_grid_w_idx)
            llm_grid_h = self.img_h_embed(llm_grid_h_idx)

            full_grid = llm_grid_h[:, None] + llm_grid_w[None, :]

            flattened = full_grid.flatten(
                0, 1
            )  # Flatten first dimension, so they are seq_len x embed_dim
            all_embeddings.append(flattened)
        return torch.concat(
            all_embeddings, dim=0
        )  # Shape is num_image_tokens x embed_dim

    def forward(
        self,
        input_ids=None,
        image_tiles=None,
        grid_thw=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
        logits_to_keep=None,
        encoder_chunk_size=None,
        **kwargs: KwargsForCausalLM,
    ):
        # Process the mixed batch if provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_ids_boxes_images(
                input_ids, image_tiles, grid_thw, encoder_chunk_size
            )

        # Handling flash attention kwargs outside the decoder to speed up + avoid graph breaks inside the decoder
        # Skipped during decoding since not required
        if (
            self.decoder.config._attn_implementation == "flash_attention_2"
            and inputs_embeds.shape[1] != 1
        ):
            batch_size, query_length, _ = inputs_embeds.shape
            indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
                attention_mask
            )
            kwargs["batch_size"] = batch_size
            kwargs["query_length"] = query_length
            kwargs["indices_k"] = indices_k
            kwargs["cu_seqlens_k"] = cu_seqlens_k
            kwargs["max_seqlen_in_batch_k"] = max_seqlen_in_batch_k

        outputs = self.decoder(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only keep the last `logits_to_keep` logits, should bring down memory usage during inference
        if logits_to_keep is not None:
            hidden_states = hidden_states[:, -logits_to_keep:, :]

        hidden_states = hidden_states.contiguous()
        bbox_logits = F.sigmoid(self.bbox_head(hidden_states))
        lm_logits = self.lm_head(hidden_states)

        return SuryaModelOutput(
            bbox_logits=bbox_logits,
            lm_logits=lm_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            past_key_values=outputs.past_key_values,
        )
