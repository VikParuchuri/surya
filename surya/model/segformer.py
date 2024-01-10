import math
from typing import Optional, Tuple, Union, List

from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor, \
    SegformerDecodeHead, SegformerModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss, BCELoss
from transformers.modeling_outputs import SemanticSegmenterOutput
from surya.settings import settings


def load_model(checkpoint=settings.MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
    model = SegformerForRegressionMask.from_pretrained(checkpoint, torch_dtype=dtype)
    model = model.to(device)
    model = model.eval()
    return model


def load_processor(checkpoint=settings.MODEL_CHECKPOINT):
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    return processor


class SegformerForRegressionMask(SegformerForSemanticSegmentation):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
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

        loss = None
        if labels is not None:
            # upsample logits to the images' original size if needed
            if logits.shape[-2:] != labels.shape[-2:]:
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
            valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
            loss_fct = BCEWithLogitsLoss(reduction='none')

            assert upsampled_logits.shape == labels.shape
            loss = loss_fct(upsampled_logits, labels.float())
            loss = (loss * valid_mask).mean()

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Apply sigmoid to get 0-1 output
        logits = torch.special.expit(logits)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )