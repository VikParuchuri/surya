from transformers import PretrainedConfig

# 对于 EfficientViT 的配置类


class EfficientViTConfig(PretrainedConfig):
    r"""
    EfficientViT 模型的配置类，继承自 PretrainedConfig。
    """

    # 模型类型
    model_type = "efficientvit"

    def __init__(
        self,
        num_classes=2,  # 分类数量，默认为2
        num_channels=3,  # 输入图像的通道数，默认为3（RGB图像）
        widths=(32, 64, 128, 256, 512),  # 每个阶段的宽度
        head_dim=32,  # 注意力头的维度
        num_stages=4,  # 模型的阶段数
        depths=(1, 1, 1, 6, 6),  # 每个阶段的深度
        strides=(2, 2, 2, 2, 2),  # 每个阶段的步幅
        hidden_sizes=(32, 64, 160, 256),  # 隐藏层的大小
        patch_size=(7, 7),  # 图像补丁的大小
        hidden_dropout_prob=0.0,  # 隐藏层的dropout概率
        attention_probs_dropout_prob=0.0,  # 注意力概率的dropout概率
        classifier_dropout_prob=0.0,  # 分类器的dropout概率
        layer_norm_eps=1e-6,  # 层归一化的epsilon值
        decoder_layer_hidden_size=128,  # 解码器层的隐藏大小
        decoder_hidden_size=512,  # 解码器的隐藏大小
        semantic_loss_ignore_index=255,  # 语义损失忽略的索引
        initializer_range=0.02,  # 初始化范围
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置分类数量
        self.num_classes = num_classes
        # 设置每个阶段的宽度
        self.widths = widths
        # 设置注意力头的维度
        self.head_dim = head_dim

        # 设置输入图像的通道数
        self.num_channels = num_channels
        # 设置模型的阶段数
        self.num_stages = num_stages
        # 设置每个阶段的深度
        self.depths = depths
        # 设置每个阶段的步幅
        self.strides = strides
        # 设置隐藏层的大小
        self.hidden_sizes = hidden_sizes
        # 设置图像补丁的大小
        self.patch_size = patch_size
        # 设置隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置分类器的dropout概率
        self.classifier_dropout_prob = classifier_dropout_prob
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置解码器层的隐藏大小
        self.decoder_hidden_size = decoder_hidden_size
        # 设置解码器的隐藏大小
        self.decoder_layer_hidden_size = decoder_layer_hidden_size
        # 设置语义损失忽略的索引
        self.semantic_loss_ignore_index = semantic_loss_ignore_index

        # 设置初始化范围
        self.initializer_range = initializer_range
