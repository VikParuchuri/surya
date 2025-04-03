from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import (
    BackboneConfigMixin,
    get_aligned_output_features_output_indices,
)

logger = logging.get_logger(__name__)


class SuryaEncoderConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = "swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=128,
        depths=[2, 2, 12, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_stride=32,
        out_features=None,
        out_indices=None,
        num_patches=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = 0  # Set to zero for inference
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        self.num_patches = num_patches
        # we set the hidden_size attribute in order to make Swin work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.stage_names = ["stem"] + [
            f"stage{idx}" for idx in range(1, len(depths) + 1)
        ]
        self._out_features, self._out_indices = (
            get_aligned_output_features_output_indices(
                out_features=out_features,
                out_indices=out_indices,
                stage_names=self.stage_names,
            )
        )
