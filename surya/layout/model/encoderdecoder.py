from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from transformers import PreTrainedModel, VisionEncoderDecoderConfig, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from surya.layout.model.encoder import DonutSwinLayoutModel
from surya.layout.model.decoder import SuryaLayoutDecoder
from transformers.utils import ModelOutput

@dataclass
class LayoutBboxOutput(ModelOutput):
    bbox_logits: torch.FloatTensor = None
    class_logits: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None


class SuryaLayoutModel(PreTrainedModel):
    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        config.decoder.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = DonutSwinLayoutModel(config.encoder)

        if decoder is None:
            decoder = SuryaLayoutDecoder(config.decoder, attn_implementation=config._attn_implementation)

        self.encoder = encoder
        self.decoder = decoder

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_boxes: torch.LongTensor = None, # Shape (batch_size, num_boxes, 7), first 6 values all coords scaled 0 - 1024, with 1025 as padding, last value is the label, 0 to 11
        decoder_cache_position: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_input_boxes_counts: torch.LongTensor = None,  # Shape (batch_size), number of boxes in each image
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], LayoutBboxOutput]:
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # We need a start token as the first token
        assert decoder_input_boxes[0][0][0] == self.config.decoder_start_token_id
        assert decoder_input_boxes[0][0].shape == (7,)

        decoder_outputs = self.decoder(
            input_boxes=decoder_input_boxes,
            input_boxes_counts=decoder_input_boxes_counts,
            cache_position=decoder_cache_position,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            use_cache=use_cache,
            **kwargs_decoder,
        )

        return LayoutBboxOutput(
            bbox_logits=decoder_outputs.bbox_logits,
            class_logits=decoder_outputs.class_logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)