from dataclasses import dataclass

import torch
from transformers import PretrainedConfig
from transformers.utils import ModelOutput


class SuryaOCRConfig(PretrainedConfig):
    model_type = "vision-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        encoder_config = kwargs.pop("encoder")
        decoder_config = kwargs.pop("decoder")

        self.encoder = encoder_config
        self.decoder = decoder_config
        self.is_encoder_decoder = True

        if isinstance(decoder_config, dict):
            self.decoder_start_token_id = decoder_config["bos_token_id"]
            self.pad_token_id = decoder_config["pad_token_id"]
            self.eos_token_id = decoder_config["eos_token_id"]
        else:
            self.decoder_start_token_id = decoder_config.bos_token_id
            self.pad_token_id = decoder_config.pad_token_id
            self.eos_token_id = decoder_config.eos_token_id


class DonutSwinConfig(PretrainedConfig):
    model_type = "donut-swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=(256, 896),
        patch_size=4,
        num_channels=3,
        embed_dim=128,
        depths=[2, 2, 14, 2],
        num_heads=[4, 8, 16, 32],
        num_kv_heads=[1, 2, 4, 8],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_length=256,
        use_positional_embeddings=False,
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


class SuryaOCRDecoderConfig(PretrainedConfig):
    model_type = "surya_ocr"

    def __init__(
        self,
        num_hidden_layers=10,
        vocab_size=65792,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        num_attention_heads=16,
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
        self_attn_layers=(0, 1, 3, 5, 7, 9),
        global_attn_layers=(0, 1, 3, 5, 7, 9),
        attention_dropout=0.0,
        num_key_value_heads=2,
        attention_bias=False,
        w_init_variance_scale=0.01,
        init_std=0.02,
        tie_word_embeddings=False,
        aux_heads=0,  # How many n-token-ahead heads to add
        encoder_hidden_size=1024,
        causal=False,
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
        self.encoder_hidden_size = encoder_hidden_size
        self.causal = causal
        self.double_residual_flow = True # Residual flow slightly different

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return (self.block_types * 100)[: self.num_hidden_layers]


class SuryaOCRTextEncoderConfig(PretrainedConfig):
    model_type = "surya_ocr"

    def __init__(
        self,
        num_hidden_layers=10,
        vocab_size=65792,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        num_attention_heads=16,
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
        self_attn_layers=(0, 1, 3, 5, 7, 9),
        global_attn_layers=(0, 1, 3, 5, 7, 9),
        attention_dropout=0.0,
        num_key_value_heads=2,
        attention_bias=False,
        w_init_variance_scale=0.01,
        init_std=0.02,
        tie_word_embeddings=False,
        aux_heads=0,  # How many n-token-ahead heads to add
        encoder_hidden_size=1024,
        iteration_count=1,
        causal=False,
        query_token_count=128,
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
        self.encoder_hidden_size = encoder_hidden_size
        self.iteration_count = iteration_count
        self.causal = causal
        self.query_token_count = query_token_count

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return (self.block_types * 100)[: self.num_hidden_layers]

TOTAL_TOKENS = 65536
TOKEN_OFFSET = 3 # Pad, eos, bos
SPECIAL_TOKENS = 253
TOTAL_VOCAB_SIZE = TOTAL_TOKENS + TOKEN_OFFSET + SPECIAL_TOKENS
LANGUAGE_MAP = {
    'af': 0,
    'am': 1,
    'ar': 2,
    'as': 3,
    'az': 4,
    'be': 5,
    'bg': 6,
    'bn': 7,
    'br': 8,
    'bs': 9,
    'ca': 10,
    'cs': 11,
    'cy': 12,
    'da': 13,
    'de': 14,
    'el': 15,
    'en': 16,
    'eo': 17,
    'es': 18,
    'et': 19,
    'eu': 20,
    'fa': 21,
    'fi': 22,
    'fr': 23,
    'fy': 24,
    'ga': 25,
    'gd': 26,
    'gl': 27,
    'gu': 28,
    'ha': 29,
    'he': 30,
    'hi': 31,
    'hr': 32,
    'hu': 33,
    'hy': 34,
    'id': 35,
    'is': 36,
    'it': 37,
    'ja': 38,
    'jv': 39,
    'ka': 40,
    'kk': 41,
    'km': 42,
    'kn': 43,
    'ko': 44,
    'ku': 45,
    'ky': 46,
    'la': 47,
    'lo': 48,
    'lt': 49,
    'lv': 50,
    'mg': 51,
    'mk': 52,
    'ml': 53,
    'mn': 54,
    'mr': 55,
    'ms': 56,
    'my': 57,
    'ne': 58,
    'nl': 59,
    'no': 60,
    'om': 61,
    'or': 62,
    'pa': 63,
    'pl': 64,
    'ps': 65,
    'pt': 66,
    'ro': 67,
    'ru': 68,
    'sa': 69,
    'sd': 70,
    'si': 71,
    'sk': 72,
    'sl': 73,
    'so': 74,
    'sq': 75,
    'sr': 76,
    'su': 77,
    'sv': 78,
    'sw': 79,
    'ta': 80,
    'te': 81,
    'th': 82,
    'tl': 83,
    'tr': 84,
    'ug': 85,
    'uk': 86,
    'ur': 87,
    'uz': 88,
    'vi': 89,
    'xh': 90,
    'yi': 91,
    'zh': 92,
    "_math": 93
}