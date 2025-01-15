from transformers import PretrainedConfig


class EfficientViTConfig(PretrainedConfig):
    r"""
    ```"""

    model_type = "efficientvit"

    def __init__(
        self,
        num_classes=2,
        num_channels=3,
        widths=(32, 64, 128, 256, 512),
        head_dim=32,
        num_stages=4,
        depths=(1, 1, 1, 6, 6),
        strides=(2, 2, 2, 2, 2),
        hidden_sizes=(32, 64, 160, 256),
        patch_size=(7, 7),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        decoder_layer_hidden_size=128,
        decoder_hidden_size=512,
        semantic_loss_ignore_index=255,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.widths = widths
        self.head_dim = head_dim

        self.num_channels = num_channels
        self.num_stages = num_stages
        self.depths = depths
        self.strides = strides
        self.hidden_sizes = hidden_sizes
        self.patch_size = patch_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layer_hidden_size = decoder_layer_hidden_size
        self.semantic_loss_ignore_index = semantic_loss_ignore_index

        self.initializer_range = initializer_range