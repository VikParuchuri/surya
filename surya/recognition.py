from typing import List
import torch
from PIL import Image
from transformers import GenerationConfig

from surya.postprocessing.math.latex import fix_math, contains_math
from surya.postprocessing.text import truncate_repetitions
from surya.settings import settings
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def get_batch_size():
    batch_size = settings.RECOGNITION_BATCH_SIZE
    if batch_size is None:
        batch_size = 32
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 64 # 12GB RAM max
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 256
    return batch_size


def batch_recognition(images: List, languages: List[List[str]], model, processor, batch_size=None):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)

    if batch_size is None:
        batch_size = get_batch_size()

    output_text = []
    confidences = []

    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_langs = languages[i:i+batch_size]
        has_math = ["_math" in lang for lang in batch_langs]
        batch_images = images[i:i+batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]

        model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

        batch_pixel_values = model_inputs["pixel_values"]
        batch_langs = model_inputs["langs"]
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

        batch_langs = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

        token_count = 0
        encoder_outputs = None
        batch_predictions = [[] for _ in range(len(batch_images))]
        sequence_scores = None

        attention_mask = torch.ones_like(batch_decoder_input, device=model.device)
        all_done = torch.zeros(len(batch_images), dtype=torch.bool, device=model.device)

        # Decoder kv cache
        # 7 (layers) x 2 (kv) x bs x 4 (heads) x max tokens x 64 (head dim)
        dec_config = model.config.decoder
        layer_count = dec_config.decoder_layers
        kv_heads = dec_config.kv_heads
        head_dim = int(dec_config.d_model / dec_config.decoder_attention_heads)
        decoder_cache = torch.zeros((layer_count, 2, len(batch_images), kv_heads, settings.RECOGNITION_MAX_TOKENS, head_dim), dtype=model.dtype, device=model.device)
        kv_mask = torch.zeros((len(batch_images), settings.RECOGNITION_MAX_TOKENS), device=model.device)

        # Encoder kv cache
        # 7 (layers) x 2 (kv) x bs x 4 (heads) x 196 (max tokens) x 64 (head dim)
        encoder_cache = torch.zeros((layer_count, 2, len(batch_images), kv_heads, 196, head_dim), dtype=model.dtype, device=model.device)

        with torch.inference_mode():
            while token_count < settings.RECOGNITION_MAX_TOKENS:
                is_prefill = token_count == 0
                inference_token_count = batch_decoder_input.shape[-1]
                return_dict = model(
                    decoder_input_ids=batch_decoder_input,
                    decoder_attention_mask=attention_mask,
                    decoder_kv_caches=None if token_count == 0 else [decoder_cache, encoder_cache],
                    decoder_past_token_count=token_count,
                    decoder_langs=batch_langs,
                    pixel_values=batch_pixel_values,
                    encoder_outputs=encoder_outputs,
                    return_dict=True,
                )

                logits = return_dict["logits"]
                preds = torch.argmax(logits[:, -1], dim=-1)
                scores = torch.max(F.softmax(logits, dim=-1), dim=-1).values
                done = preds == processor.tokenizer.eos_id
                all_done = all_done | done

                if sequence_scores is None:
                    sequence_scores = scores
                else:
                    scores[all_done == 1] = 0
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                encoder_outputs = (return_dict["encoder_last_hidden_state"],)
                past_key_values = return_dict["past_key_values"]
                for layer_idx, layer in enumerate(past_key_values):
                    decoder_cache[layer_idx, 0, :, :, token_count:(token_count + inference_token_count), :] = layer[0]
                    decoder_cache[layer_idx, 1, :, :, token_count:(token_count + inference_token_count), :] = layer[1]

                    if is_prefill:
                        encoder_cache[layer_idx, 0, :, :, :, :] = layer[2]
                        encoder_cache[layer_idx, 1, :, :, :, :] = layer[3]

                if all_done.all():
                    break

                kv_mask[:, token_count:(token_count + inference_token_count)] = 1
                attention_mask = torch.cat([kv_mask, ~all_done.unsqueeze(1)], dim=1)

                for j, (pred, status) in enumerate(zip(preds, all_done)):
                    if not status:
                        batch_predictions[j].append(int(pred))

                batch_decoder_input = preds.unsqueeze(1)
                token_count += inference_token_count

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)
        detected_text = processor.tokenizer.batch_decode(batch_predictions)
        detected_text = [truncate_repetitions(dt) for dt in detected_text]

        # Postprocess to fix LaTeX output (add $$ signs, etc)
        detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
        output_text.extend(detected_text)
        confidences.extend(sequence_scores.tolist())

    return output_text, confidences






