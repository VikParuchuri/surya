import random
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, VisionEncoderDecoderConfig, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right
from surya.model.table_rec.decoder import SuryaTableRecTextEncoder, SuryaTableRecDecoder
from surya.model.recognition.encoder import DonutSwinModel
import torch.nn.functional as F
from transformers.utils import ModelOutput


@dataclass
class TableRecOutput(ModelOutput):
    row_logits: torch.FloatTensor = None
    col_logits: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TableRecEncoderDecoderModel(PreTrainedModel):
    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        config.decoder.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = DonutSwinModel(config.encoder)

        if text_encoder is None:
            text_encoder = SuryaTableRecTextEncoder(config.text_encoder, attn_implementation=config._attn_implementation)

        if decoder is None:
            decoder = SuryaTableRecDecoder(config.decoder, attn_implementation=config._attn_implementation)

        self.encoder = encoder
        self.decoder = decoder
        self.text_encoder = text_encoder

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder
        self.text_encoder.config = self.config.text_encoder

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
        decoder_input_ids: torch.LongTensor = None,
        decoder_cache_position: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], TableRecOutput]:
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # Decode
        decoder_outputs = self.decoder(
            input_labels=decoder_input_ids,
            input_boxes_counts=None,
            cache_position=decoder_cache_position,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=None,
            use_cache=use_cache,
            **kwargs_decoder,
        )

        return TableRecOutput(
            row_logits=decoder_outputs.row_logits,
            col_logits=decoder_outputs.col_logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)