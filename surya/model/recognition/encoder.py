from torch import nn
import torch
from typing import Optional, Tuple

from transformers.models.donut.modeling_donut_swin import DonutSwinPatchEmbeddings, DonutSwinEmbeddings, DonutSwinModel, \
    DonutSwinEncoder

from surya.model.recognition.config import VariableDonutSwinConfig


class VariableDonutSwinEmbeddings(DonutSwinEmbeddings):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__(config, use_mask_token)

        self.patch_embeddings = DonutSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None
        self.position_embeddings = None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:

        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # Layernorm across the last dimension (each patch is a single row)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, embed_dim = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings[:, :seq_len, :]

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class VariableDonutSwinModel(DonutSwinModel):
    config_class = VariableDonutSwinConfig
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = VariableDonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()