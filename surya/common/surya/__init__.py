import warnings
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
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
        **kwargs: KwargsForCausalLM,
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