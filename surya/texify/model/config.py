from dataclasses import dataclass
from typing import Dict

import torch
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

from surya.settings import settings


class TexifyConfig(PretrainedConfig):
    model_type = "vision-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "encoder" in kwargs:
            encoder_config = kwargs.pop("encoder")
            decoder_config = kwargs.pop("decoder")
        else:
            encoder_config = TexifyEncoderConfig()
            decoder_config = TexifyDecoderConfig()

        self.encoder = encoder_config
        self.decoder = decoder_config
        self.is_encoder_decoder = True

        if isinstance(decoder_config, dict):
            self.decoder_start_token_id = decoder_config["bos_token_id"]
            self.decoder_end_token_id = decoder_config["eos_token_id"]
            self.pad_token_id = decoder_config["pad_token_id"]
            self.eos_token_id = decoder_config["eos_token_id"]
        else:
            self.decoder_start_token_id = decoder_config.bos_token_id
            self.decoder_end_token_id = decoder_config.eos_token_id
            self.pad_token_id = decoder_config.pad_token_id
            self.eos_token_id = decoder_config.eos_token_id


@dataclass
class TexifyModelOutput(ModelOutput):
    logits: Dict[str, torch.Tensor]
    hidden_states: torch.Tensor | None = None


class TexifyEncoderConfig(PretrainedConfig):
    model_type = "donut-swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=(settings.TEXIFY_IMAGE_SIZE["height"], settings.TEXIFY_IMAGE_SIZE["width"]),
        patch_size=4,
        num_channels=3,
        embed_dim=128,
        depths=[2, 2, 12, 2],
        num_heads=[4, 8, 16, 32],
        num_kv_heads=[4, 8, 16, 32],
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
        encoder_length=1024,
        use_positional_embeddings=True,
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
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # we set the hidden_size attribute in order to make Swin work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.encoder_length = encoder_length
        self.use_positional_embeddings = use_positional_embeddings


class TexifyDecoderConfig(PretrainedConfig):
    model_type = "texify"

    def __init__(
        self,
        num_hidden_layers=6,
        vocab_size=68549,
        hidden_size=512,
        intermediate_size=4 * 512,
        encoder_hidden_size=1024,
        num_attention_heads=8,
        lru_width=None,
        attention_window_size=16,
        conv1d_width=4,
        logits_soft_cap=30.0,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
        hidden_activation="gelu_pytorch_tanh",
        rope_theta=10000.0,
        block_types=("attention",),
        cross_attn_layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        encoder_cross_attn_layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        self_attn_layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        global_attn_layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        attention_dropout=0.0,
        num_key_value_heads=4,
        attention_bias=False,
        w_init_variance_scale=0.01,
        init_std=0.02,
        tie_word_embeddings=False,
        aux_heads=0, # How many n-token-ahead heads to add
        causal=True,
        layer_norm_eps=1e-5,
        dropout=.1,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.lru_width = lru_width if lru_width is not None else hidden_size
        self.attention_window_size = attention_window_size
        self.conv1d_width = conv1d_width
        self.logits_soft_cap = logits_soft_cap
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.block_types = list(block_types)
        self.hidden_activation = hidden_activation
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("The number of `num_key_value_heads` must be smaller than `num_attention_heads`")
        self.cross_attn_layers = cross_attn_layers
        self.self_attn_layers = self_attn_layers
        self.global_attn_layers = global_attn_layers
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.w_init_variance_scale = w_init_variance_scale
        self.final_w_init_variance_scale = 2.0 / self.num_hidden_layers
        self.init_std = init_std
        self.tie_word_embeddings = tie_word_embeddings
        self.aux_heads = aux_heads
        self.encoder_hidden_size=encoder_hidden_size
        self.causal = causal
        self.encoder_cross_attn_layers = encoder_cross_attn_layers
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.double_residual_flow = False

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return (self.block_types * 100)[: self.num_hidden_layers]