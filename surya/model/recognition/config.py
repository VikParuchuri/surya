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


@dataclass
class EfficientViTModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None


class EfficientViTConfig(PretrainedConfig):
    r"""
    ```"""

    model_type = "efficientvit"

    def __init__(
        self,
        num_channels=3,
        widths=(32, 64, 128, 512, 1024),
        head_dim=32,
        num_stages=4,
        depths=(1, 1, 1, 6, 6),
        strides=(4, 2, 1, 2, 2),
        hidden_sizes=(32, 64, 160, 512),
        patch_size=(7, 7),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        encoder_length=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.widths = widths
        self.head_dim = head_dim
        self.hidden_size = widths[-1]

        self.num_channels = num_channels
        self.num_stages = num_stages
        self.depths = depths
        self.strides = strides
        self.hidden_sizes = hidden_sizes
        self.patch_size = patch_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.encoder_length = encoder_length

        self.initializer_range = initializer_range


class SuryaOCRDecoderConfig(PretrainedConfig):
    model_type = "surya_ocr"

    def __init__(
        self,
        num_hidden_layers=8,
        vocab_size=65792,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        num_attention_heads=16,
        lru_width=None,
        attention_window_size=16,
        max_tokens=256,
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
        cross_attn_every=2,
        global_attn_every=2,
        attention_dropout=0.0,
        num_key_value_heads=2,
        attention_bias=False,
        w_init_variance_scale=0.01,
        init_std=0.02,
        tie_word_embeddings=False,
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
        self.cross_attn_every = cross_attn_every
        self.global_attn_every = global_attn_every
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.w_init_variance_scale = w_init_variance_scale
        self.final_w_init_variance_scale = 2.0 / self.num_hidden_layers
        self.init_std = init_std
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return (self.block_types * 100)[: self.num_hidden_layers]