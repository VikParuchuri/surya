from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SuryaEncoderConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "depth",
    }

    def __init__(
        self,
        depth=8,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=1,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=1280,
        fullatt_block_indexes=(3, 7),
        initializer_range=0.02,
        image_size=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self.spatial_patch_size = spatial_patch_size
        self.image_size = image_size
