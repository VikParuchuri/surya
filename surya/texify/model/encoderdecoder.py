from typing import Optional, Union, Tuple

import torch
from transformers import PreTrainedModel, VisionEncoderDecoderConfig, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from surya.texify.model.decoder import TexifyDecoder
from surya.texify.model.encoder import TexifyEncoder


class TexifyModel(PreTrainedModel):
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
            encoder = TexifyEncoder(config.encoder)

        if decoder is None:
            decoder = TexifyDecoder(config.decoder, attn_implementation=config._attn_implementation)

        self.encoder: TexifyEncoder = encoder
        self.decoder: TexifyDecoder = decoder

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
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_cache_position: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

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

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            cache_position=decoder_cache_position,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            **kwargs_decoder,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)