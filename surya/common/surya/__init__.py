import warnings
from dataclasses import dataclass
from typing import Optional, Unpack, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.config import SuryaModelConfig
from surya.common.surya.decoder.__init__ import SuryaDecoderModel
from surya.common.surya.embedder.__init__ import SimpleTokenEmbedder
from surya.common.surya.encoder.__init__ import SuryaEncoderModel


@dataclass
class SuryaModelOutput(CausalLMOutputWithPast):
    bbox_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None


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

    def __init__(
        self,
        config: SuryaModelConfig,
        embedder: SimpleTokenEmbedder = None,
        vision_encoder: SuryaEncoderModel = None,
        decoder: SuryaDecoderModel = None,
    ):
        super().__init__(config)

        if vision_encoder is None:
            vision_encoder = SuryaEncoderModel(
                config.vision_encoder
            )

        if decoder is None:
            decoder = SuryaDecoderModel(config.decoder)

        if embedder is None:
            embedder = SimpleTokenEmbedder(config)

        self.vision_encoder = vision_encoder
        self.decoder = decoder
        self.embedder = embedder

        # Tying configs
        self.vision_encoder.config = self.config.vision_encoder
        self.decoder.config = self.config.decoder

        self.vision_projector = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, self.decoder.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size),
        )

        self.bbox_head = nn.Linear(config.hidden_size, 6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def get_image_embeddings(self, image_tiles):
        # embed all images with the vision encoder after they have already been tiled and flattened into a single batch
        # TODO - Batch processing, since the total #image_tiles could be much higher than the original batch size
        return self.vision_projector(
            self.vision_encoder.embed_images(image_batch=image_tiles)
        )

    def embed_ids_boxes_images(self, input_ids, input_boxes, image_tiles):
        """
        Insert embedded image tiles into the corresponding positions into the full input sequence

        Positions to insert new tokens are indicated by the special image token index
        """
        inputs_embeds = self.embedder.embed(
            input_tokens=input_ids, input_bboxes=input_boxes
        )
        if image_tiles is not None:
            image_features = self.get_image_embeddings(image_tiles=image_tiles)

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = torch.sum((input_ids == self.config.image_token_id))
                n_image_features = image_features.shape[0] * image_features.shape[1]
                warnings.warn(
                    f"Image features and image tokens do not match: tokens {n_image_tokens}, features {n_image_features}. This may lead to unexpected results"
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )
        else:
            assert (input_ids == self.config.image_token_id).sum() == 0, (
                "Image tokens were present in the input but no input images were provided"
            )

        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        input_boxes=None,
        image_tiles=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        # Process the mixed batch if provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_ids_boxes_images(
                input_ids, input_boxes, image_tiles
            )

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
        bbox_logits = F.sigmoid(self.bbox_head(hidden_states))
        lm_logits = self.lm_head(hidden_states)

        return SuryaModelOutput(
            bbox_logits=bbox_logits,
            lm_logits=lm_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            past_key_values=outputs.past_key_values,
        )

    # Always uses caching
    # TODO Add static caching + torch compile
    # TODO add stopping criteria.
    def custom_generate(
        self,
        input_ids: torch.Tensor = None,
        input_boxes: Optional[torch.Tensor] = None,
        image_tiles: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        needs_boxes: List[bool] = None,
        eos_token_id: int = None,
        blank_bbox_token_id: int = None,
        ignore_bbox_token_ids: List[int] = None,
        max_new_tokens: int = 100,
    ):
        config: SuryaModelConfig = self.config
        if eos_token_id is None:
            eos_token_id = config.eos_token_id
        if blank_bbox_token_id is None:
            blank_bbox_token_id = config.blank_bbox_token_id

        batch_size = input_ids.shape[0]
        new_tokens = [[]] * batch_size
        new_boxes = [[]] * batch_size
        past_key_values = DynamicCache()
        if needs_boxes is None:
            needs_boxes = [False] * batch_size

        skip_bbox_idxs = ~np.array(needs_boxes)
        all_done = np.array([False] * batch_size)
        for _ in range(max_new_tokens):
            # This model always returns a dict
            outputs = self(
                input_ids=input_ids,
                input_boxes=input_boxes,
                image_tiles=image_tiles,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=None,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Update cache
            past_key_values = outputs["past_key_values"]

            # Prepare logits for tokens and bboxes - Clone is from this line - https://github.com/huggingface/transformers/blob/bc30dd1efb99f571d45b2e2131a555d09285ddd8/src/transformers/generation/utils.py#L3252
            next_token_logits = outputs["lm_logits"][:, -1:, :].clone().float()
            next_bbox_logits = outputs["bbox_logits"][:, -1:, :].clone().float()

            next_tokens = torch.argmax(next_token_logits, dim=-1)
            next_bboxes = next_bbox_logits * self.config.bbox_size

            updated_attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_tokens, dtype=torch.long)], dim=1
            ).to(attention_mask.device)
            updated_position_ids = position_ids[:, -1:] + 1

            for j in range(batch_size):
                # Ignore the generated bbox and force it to blank if the token is special
                tok = next_tokens[j].item()
                if tok in ignore_bbox_token_ids:
                    next_bboxes[j, -1, :] = blank_bbox_token_id

                new_tokens[j].append(tok)
                new_boxes[j].append(next_bboxes[j][0].tolist())

            done = (next_tokens == eos_token_id).cpu().numpy()
            all_done = all_done | done
            if all_done.all():
                break

            # Removing image tiles after prefill since they are no longer required
            input_ids = next_tokens
            input_boxes = next_bboxes.to(torch.long)
            input_boxes[skip_bbox_idxs, -1, :] = blank_bbox_token_id
            attention_mask = updated_attention_mask
            position_ids = updated_position_ids
            image_tiles = None

        return new_tokens, new_boxes
