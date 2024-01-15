from typing import Optional, Union

import torch
from torch import nn
from transformers import (
    SegformerConfig,
    SegformerDecodeHead,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerModel,
)
from transformers.modeling_outputs import SemanticSegmenterOutput

from surya.settings import settings


def load_model(
    checkpoint=settings.DETECTOR_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
):
    config = SegformerConfig.from_pretrained(checkpoint)
    model = SegformerForRegressionMask.from_pretrained(checkpoint, torch_dtype=dtype, config=config)
    if "mps" in device:
        print(
            "Warning: MPS may have poor results. This is a bug with MPS, see here - https://github.com/pytorch/pytorch/issues/84936"
        )
    model = model.to(device)
    model = model.eval()
    return model


def load_processor(checkpoint=settings.DETECTOR_MODEL_CHECKPOINT):
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    return processor


class SegformerForMaskMLP(nn.Module):
    def __init__(self, config: SegformerConfig, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerForMaskDecodeHead(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        decoder_layer_hidden_size = getattr(config, "decoder_layer_hidden_size", config.decoder_hidden_size)

        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerForMaskMLP(config, input_dim=config.hidden_sizes[i], output_dim=decoder_layer_hidden_size)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_layer_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config


class SegformerForRegressionMask(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerForMaskDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)
        # Apply sigmoid to get 0-1 output
        logits = torch.special.expit(logits)

        return SemanticSegmenterOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
