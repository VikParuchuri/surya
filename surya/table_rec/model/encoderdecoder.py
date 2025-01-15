from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict

import torch
from transformers import PreTrainedModel, VisionEncoderDecoderConfig, PretrainedConfig
from surya.table_rec.model.decoder import SuryaTableRecDecoder
from surya.table_rec.model.encoder import DonutSwinModel
from transformers.utils import ModelOutput


@dataclass
class TableRecOutput(ModelOutput):
    box_property_logits: Dict[str, torch.FloatTensor]
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
        decoder: Optional[PreTrainedModel] = None,
    ):
        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        config.decoder.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = DonutSwinModel(config.encoder)

        if decoder is None:
            decoder = SuryaTableRecDecoder(config.decoder, attn_implementation=config._attn_implementation)

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
            box_property_logits=decoder_outputs.box_property_logits,
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