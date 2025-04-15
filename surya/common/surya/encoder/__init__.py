from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from surya.common.donut.encoder import (
    DonutSwinPreTrainedModel,
    DonutSwinEmbeddings,
    DonutSwinEncoder,
    DonutSwinModelOutput,
)
from surya.common.surya.encoder.config import SuryaEncoderConfig


class SwinModel(DonutSwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = DonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DonutSwinModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return DonutSwinModelOutput(
            last_hidden_state=sequence_output,
        )


class SuryaEncoderModel(SwinModel):
    @property
    def image_size(self) -> int:
        config: SuryaEncoderConfig = self.config
        if isinstance(config.image_size, tuple) and len(config.image_size) == 2:
            return config.image_size
        elif isinstance(config.image_size, int):
            return (config.image_size, config.image_size)

        raise ValueError(
            f"The `image_size` for SwinConfig should be a tuple of (int, int) or a single int but found {type(config.image_size)}"
        )

    @property
    def hidden_size(self) -> int:
        config: SuryaEncoderConfig = self.config
        return config.hidden_size

    def embed_images(self, image_batch: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(
                pixel_values=image_batch,
                bool_masked_pos=None,
                output_attentions=False,
                output_hidden_states=False,
                interpolate_pos_encoding=False,
                return_dict=True,
            )
            .last_hidden_state
        )

    @property
    def num_patches(self) -> int:
        # Final swin stage does not downsample, so we can get the value from its last layer
        final_layer_input_resolution = (
            self.encoder.layers[-1].blocks[-1].input_resolution
        )
        return final_layer_input_resolution[0] * final_layer_input_resolution[1]
