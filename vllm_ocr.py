from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Set, TypedDict, Union, Tuple

import torch
from transformers import PreTrainedModel, VisionEncoderDecoderConfig, PretrainedConfig
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers import DonutSwinConfig
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers import AutoConfig, AutoTokenizer


from surya.common.adetr.decoder import SuryaADETRDecoderPreTrainedModel
from surya.settings import settings
from surya.recognition.model.config import DonutSwinConfig, SuryaOCRDecoderConfig, SuryaOCRConfig
from surya.recognition.model.encoder import DonutSwinModel
from surya.recognition.model.decoder import SuryaOCRTextEncoder, SuryaOCRTextEncoderConfig
from surya.recognition.tokenizer import Byt5LangTokenizer

from vllm.config import VllmConfig, CacheConfig
from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsMultiModal

from vllm.inputs import (INPUT_REGISTRY, DummyData, EncoderDecoderInputs,
                         InputContext, TokenInputs, token_inputs)
from vllm.sequence import SequenceData
from vllm.multimodal import MULTIMODAL_REGISTRY

class WrappedEmbedding(nn.Embedding):
    def forward(self, input_ids, *args, **kwargs):
        return super().forward(input_ids)

class SuryaADETRDecoderRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.register_buffer('zero_tensor', torch.tensor(0.0))

    def _norm(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)

        # Add clipping to prevent division by zero
        variance = torch.clamp(variance, min=self.eps)
        return x * torch.rsqrt(variance)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst SuryaADETRDecoder is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        # Clamp to float16 range
        f16_info = torch.finfo(x.dtype)
        output = output.clamp(min=f16_info.min, max=f16_info.max)
        output = torch.where(torch.isnan(output),
                             self.zero_tensor.to(output.device),
                             output)
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


ALL_LAYERNORM_LAYERS.append(SuryaADETRDecoderRMSNorm)

class SuryaADETRDecoderCrossAttentionVLLM(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    Modified for GQA
    """

    def __init__(self, config: PretrainedConfig, cache_config: CacheConfig, quant_config: QuantizationConfig, prefix: str=""):
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.qkv_dec_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.qkv_enc_proj = QKVParallelLinear(
            hidden_size=self.config.encoder_hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = Attention(
            num_heads=self.num_attention_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.ENCODER_DECODER,
            prefix=f'{prefix}.attn'
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Encoder attention mask currently ignored
        qkv_dec, _ = self.qkv_dec_proj(hidden_states)
        q, k, v = qkv_dec.split([self.num_attention_heads*self.head_dim, self.num_key_value_heads*self.head_dim, self.num_key_value_heads*self.head_dim], dim=-1)

        #TODO FIX
        # qkv_enc, _ = self.qkv_enc_proj(encoder_hidden_states)
        # qkv_enc, _ = self.qkv_dec_proj(hidden_states)
        # _, k, v = qkv_enc.split([self.num_attention_heads*self.head_dim, self.num_key_value_heads*self.head_dim, self.num_key_value_heads*self.head_dim], dim=-1)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    def _setup_cache(self, batch_size, device, dtype=None):
        # Setup initial caches
        self.value_states = None
        self.key_states = None

    @torch.no_grad()
    def _update_cache(self, key_states, value_states, **cache_kwargs):
        self.value_states = value_states
        self.key_states = key_states


class SuryaADETRDecoderSelfAttentionVLLM(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PretrainedConfig, quant_config: QuantizationConfig, cache_config: CacheConfig, max_boxes=None, prefix: str=""):
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        #TODO - Figure out is_neox_style!!!!
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_boxes,
            base=config.rope_theta,
            rope_scaling=None,
            is_neox_style=False,
        )

        self.attn = Attention(
            num_heads=self.num_attention_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            prefix=f'{prefix}.attn'
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.num_attention_heads*self.head_dim, self.num_key_value_heads*self.head_dim, self.num_key_value_heads*self.head_dim], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(
            q, k, v, kv_cache, attn_metadata
        )
        output, _ = self.o_proj(attn_output)
        return output

class SuryaADETRDecoderMlp(nn.Module):
    def __init__(self, config, prefix:str=""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_activation is None:
            config.hidden_activation = "gelu_pytorch_tanh"
        hidden_activation = config.hidden_activation
        self.act_fn = ACT2FN[hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SuryaADETRDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, layer_idx, max_boxes=None, prefix:str=""):
        super().__init__()
        
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.cross_pre_norm = SuryaADETRDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps, prefix=f'{prefix}.cross_pre_norm')
        self.temporal_pre_norm = SuryaADETRDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps, prefix=f'{prefix}.temporal_pre_norm')

        self.temporal_block = None
        if layer_idx in config.self_attn_layers:
            self.temporal_block = SuryaADETRDecoderSelfAttentionVLLM(config, cache_config=cache_config, quant_config=quant_config, max_boxes=max_boxes, prefix=f'{prefix}.self_attn')

        self.cross_attn_block = None
        if layer_idx in config.cross_attn_layers:
            self.cross_attn_block = SuryaADETRDecoderCrossAttentionVLLM(config, cache_config=cache_config, quant_config=quant_config, prefix=f'{prefix}.cross_attn')

        self.window_attn = layer_idx not in config.global_attn_layers
        self.channel_pre_norm = SuryaADETRDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps, prefix=f'{prefix}.channel_pre_norm')
        self.mlp_block = SuryaADETRDecoderMlp(config, prefix=f'{prefix}.mlp')

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            encoder_hidden_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if self.cross_attn_block is not None:
            # Do cross-attention on encoder outputs
            cross_attn_inputs = self.cross_pre_norm(hidden_states)
            cross_attn_path = self.cross_attn_block(
                hidden_states=cross_attn_inputs, positions=positions, encoder_hidden_states=encoder_hidden_states, kv_cache=kv_cache, attn_metadata=attn_metadata
            )
            hidden_states = cross_attn_path + hidden_states

        if self.temporal_block is not None:
            temporal_inputs = self.temporal_pre_norm(hidden_states)  # RMSNorm introduces slight slight differences
            temporal_path = self.temporal_block(
                hidden_states=temporal_inputs, positions=positions, kv_cache=kv_cache, attn_metadata=attn_metadata
            )

            hidden_states = temporal_path + hidden_states

        block_input = hidden_states
        hidden_states = self.channel_pre_norm(block_input)
        hidden_states = self.mlp_block(hidden_states)
        hidden_states = hidden_states + block_input

        return hidden_states

class SuryaADETRDecoderModelVLLM(SuryaADETRDecoderPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SuryaADETRDecoderDecoderLayer`]

    Args:
        config: PretrainedConfig
    """

    def __init__(
            self,
            *,
            config: PretrainedConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            embedder: nn.Module = None,
            max_boxes: int = None,
            static_cache: bool = False,
            prefix:str = ""
    ):
        super().__init__(config)
        
        self.cache_config = cache_config
        self.quant_config = quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.causal = config.causal

        self.embed_tokens = embedder
        self.max_boxes = max_boxes
        self.static_cache = static_cache

        self.layers = nn.ModuleList(
            [SuryaADETRDecoderLayer(config, cache_config=cache_config, quant_config=quant_config, layer_idx=layer_idx, max_boxes=max_boxes, prefix=f'{prefix}.layers.{layer_idx}') for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = SuryaADETRDecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps, prefix=f'{prefix}.final_norm')

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches:List[torch.Tensor],
        attn_metadata:AttentionMetadata,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for i, residual_block in enumerate(self.layers):
            hidden_states = residual_block(
                hidden_states=hidden_states,
                positions=positions,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.final_norm(hidden_states)

        return hidden_states


def dummy_image(num_images: int, ):
    width = height = 1024
    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}

def dummy_encoder_seq_data(ctx: InputContext, num_images: int):
    num_tokens = num_images

    return SequenceData.from_prompt_token_counts(
        (0, num_tokens))

def dummy_decoder_seq_data(seq_len: int, num_images: int):
    # <|image|> * num_images + 0 * (seq_len - num_images)
    assert seq_len >= num_images, \
        "seq_len should be greater than or equal to num_images"

    return SequenceData.from_prompt_token_counts(
        (0, num_images),
        (0, seq_len - num_images),
    )

def dummy_encoder_data_for_surya_ocr_decoder(ctx: InputContext, seq_len: int,
                                  mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]
    return DummyData(dummy_encoder_seq_data(ctx, num_images),
                     dummy_image(num_images))

def dummy_decoder_data_for_surya_ocr_decoder(ctx: InputContext, seq_len: int,
                                  mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]
    return DummyData(dummy_decoder_seq_data(seq_len, num_images))

def input_processor_for_surya_ocr_decoder(
    ctx: InputContext,
    inputs: EncoderDecoderInputs,
) -> EncoderDecoderInputs:
    import ipdb; ipdb.set_trace()

class SuryaOCRDecoderConfigVLLM(SuryaOCRDecoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_encoder_decoder = True
        self.decoder_start_token_id = 0

class SuryaOCRDecoderVLLM(SuryaADETRDecoderPreTrainedModel, SupportsMultiModal):
    _tied_weights_keys = None
    config_class = SuryaOCRDecoderConfigVLLM

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        super().__init__(config)

        embed_tokens = WrappedEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model = SuryaADETRDecoderModelVLLM(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            embedder=embed_tokens,
            static_cache=settings.RECOGNITION_STATIC_CACHE,
            max_boxes=settings.RECOGNITION_MAX_TOKENS,
            prefix=maybe_prefix(prefix, 'model')
        )
        self.vocab_size = config.vocab_size
        aux_heads = config.aux_heads if config.aux_heads is not None else 0
        lm_heads = aux_heads + 1
        self.lm_head = ParallelLMHead(num_embeddings=config.vocab_size*lm_heads, embedding_dim=config.hidden_size, bias=False, prefix=f'{prefix}.lm_head')

        self.logits_processor = LogitsProcessor(vocab_size=self.vocab_size)
        self.sampler = get_sampler()

        # Initialize weights and apply final processing
        self.post_init()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        encoder_hidden_states = None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            encoder_hidden_states=encoder_hidden_states
        )

    def load_weights(self, weights):
        pass

    def get_input_embeddings(self, input_ids:torch.Tensor):
        return self.model.embed_tokens(input_ids)

# class OCREncoderDecoderModel(PreTrainedModel):
#     config_class = VisionEncoderDecoderConfig
#     base_model_prefix = "vision_encoder_decoder"
#     main_input_name = "pixel_values"
#     supports_gradient_checkpointing = True
#     _supports_param_buffer_assignment = False

#     def __init__(
#         self,
#         config: Optional[PretrainedConfig] = None,
#         encoder: Optional[PreTrainedModel] = None,
#         decoder: Optional[PreTrainedModel] = None,
#         text_encoder: Optional[PreTrainedModel] = None,
#     ):
#         # initialize with config
#         # make sure input & output embeddings is not tied
#         config.tie_word_embeddings = False
#         config.decoder.tie_word_embeddings = False
#         super().__init__(config)

#         if encoder is None:
#             encoder = DonutSwinModel(config.encoder)

#         if decoder is None:
#             decoder = SuryaOCRDecoderVLLM(config.decoder, attn_implementation=config._attn_implementation)

#         if text_encoder is None:
#             text_encoder = SuryaOCRTextEncoder(config.text_encoder, attn_implementation=config._attn_implementation)

#         self.encoder: DonutSwinModel = encoder
#         self.decoder: SuryaOCRDecoderVLLM = decoder
#         self.text_encoder: SuryaOCRTextEncoder = text_encoder

#         # make sure that the individual model's config refers to the shared config
#         # so that the updates to the config will be synced
#         self.encoder.config = self.config.encoder
#         self.decoder.config = self.config.decoder
#         self.text_encoder.config = self.config.text_encoder

'''
  TODO

- [ ] Weight loading
- [ ] Switch to encoder hidden states in cross attention + Pass encoder hidden states as an input to the LLM engine - Shouldn't be too hard
'''


NUM_IMAGES = 10000
BATCH_SIZE = 256
if __name__ == '__main__':
    # checkpoint = 'vikp/surya_rec2'
    # config = SuryaOCRConfig.from_pretrained(checkpoint)
    # decoder_config = config.decoder
    # decoder = SuryaOCRDecoderConfig(**decoder_config)
    # config.decoder = decoder

    # encoder_config = config.encoder
    # encoder = DonutSwinConfig(**encoder_config)
    # config.encoder = encoder

    # text_encoder_config = config.text_encoder
    # text_encoder = SuryaOCRTextEncoderConfig(**text_encoder_config)
    # config.text_encoder = text_encoder
    # model = OCREncoderDecoderModel.from_pretrained(checkpoint, config=config)
    # model = model.cuda()
    # model = model.eval()
    # print('Loaded')

    from tqdm import tqdm
    from PIL import Image
    image = Image.open('/home/ubuntu/datalab/marker/conversion_results/crop.png')

    AutoConfig.register('surya_ocr', SuryaOCRDecoderConfigVLLM)
    AutoTokenizer.register(SuryaOCRDecoderConfigVLLM, Byt5LangTokenizer)

    from vllm import ModelRegistry, LLM, SamplingParams
    ModelRegistry.register_model("SuryaOCRDecoder", SuryaOCRDecoderVLLM)

    llm = LLM('/home/ubuntu/surya_ocr_decoder', tokenizer=None, skip_tokenizer_init=True)
    sampling_params = SamplingParams(max_tokens=50)

    from vllm.inputs import TokensPrompt, ExplicitEncoderDecoderPrompt
    input_ids = list(range(10))
    single_tokens_prompt = TokensPrompt(prompt_token_ids=input_ids)
    vllm_prompt = ExplicitEncoderDecoderPrompt(encoder_prompt=single_tokens_prompt, decoder_prompt=single_tokens_prompt)

    # outputs = llm.generate(prompts=[vllm_input]*NUM_IMAGES, sampling_params=sampling_params)
    outputs = llm.generate([{
        "prompt": vllm_prompt,
        "multi_modal_data": {
            'image': image
        }
    }]*NUM_IMAGES)