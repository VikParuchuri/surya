import torch
from typing import List
from PIL import Image

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
            batch_size = 512
    return batch_size


def pad_to_batch_size(tensor, batch_size):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

    return F.pad(tensor, padding, mode='constant', value=0)


def batch_recognition(images: List, languages: List[List[str] | None], model, processor, batch_size=None):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)

    if len(images) == 0:
        return [], []

    if batch_size is None:
        batch_size = get_batch_size()

    # Sort images by width, so similar length ones go together
    sorted_pairs = sorted(enumerate(images), key=lambda x: x[1].width, reverse=False)
    indices, images = zip(*sorted_pairs)
    indices = list(indices)
    images = list(images)

    output_text = []
    confidences = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_images = images[i:i+batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images

        batch_langs = languages[i:i+batch_size]
        has_math = [lang and "_math" in lang for lang in batch_langs]

        processed_batch = processor(text=[""] * len(batch_images), images=batch_images, langs=batch_langs)

        batch_pixel_values = processed_batch["pixel_values"]
        batch_langs = processed_batch["langs"]
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]
        max_input_length = max([len(tokens) for tokens in batch_decoder_input])

        # Pad decoder input to max length if needed, to ensure we can convert to a tensor
        for token_idx in range(len(batch_decoder_input)):
            lang_len = len(batch_decoder_input[token_idx])
            if lang_len < max_input_length:
                batch_decoder_input[token_idx] = [processor.tokenizer.pad_id] * (max_input_length - lang_len) + batch_decoder_input[token_idx]

        current_batch_size = len(batch_pixel_values)

        batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=model.dtype, device=model.device)
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)

        token_count = 0
        inference_token_count = batch_decoder_input.shape[-1]
        batch_predictions = [[] for _ in range(current_batch_size)]

        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64, device=model.device).cumsum(0) - 1
        model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)
        model.text_encoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)

        sequence_scores = None
        all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)
        encoder_hidden_states = None

        with torch.no_grad(): # inference_mode doesn't work with torch.compile
            encoder_batch_size = batch_size // settings.RECOGNITION_ENCODER_BATCH_DIVISOR + 1
            for z in range(0, batch_pixel_values.shape[0], encoder_batch_size):
                encoder_pixel_values = batch_pixel_values[z:min(z + encoder_batch_size, batch_pixel_values.shape[0])]
                encoder_hidden_states_batch = model.encoder(pixel_values=encoder_pixel_values).last_hidden_state
                if encoder_hidden_states is None:
                    encoder_hidden_states = encoder_hidden_states_batch
                else:
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_batch], dim=0)

            text_encoder_input_ids = torch.arange(
                model.text_encoder.config.query_token_count,
                device=encoder_hidden_states.device,
                dtype=torch.long
            ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)

            encoder_text_hidden_states = model.text_encoder(
                input_ids=text_encoder_input_ids,
                cache_position=None,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                use_cache=False
            ).hidden_states
            del encoder_hidden_states

            if settings.RECOGNITION_STATIC_CACHE:
                # Pad inputs to max batch size for static cache
                encoder_text_hidden_states = pad_to_batch_size(encoder_text_hidden_states, batch_size)
                batch_decoder_input = pad_to_batch_size(batch_decoder_input, batch_size)

            while token_count < settings.RECOGNITION_MAX_TOKENS - 1:
                is_prefill = token_count == 0
                #TODO: add attention mask
                return_dict = model.decoder(
                    input_ids=batch_decoder_input,
                    encoder_hidden_states=encoder_text_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True,
                    prefill=is_prefill
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1
                logits = return_dict["logits"][:current_batch_size] # Ignore batch padding
                aux_logits = return_dict.get("aux_logits", None)

                preds = torch.argmax(logits[:, -1], dim=-1)
                scores = torch.max(F.softmax(logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)
                done = (preds == processor.tokenizer.eos_id) | (preds == processor.tokenizer.pad_id)
                done = done
                all_done = all_done | done

                if is_prefill:
                    sequence_scores = scores
                else:
                    scores = scores.masked_fill(all_done, 0)
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                if all_done.all():
                    break

                batch_decoder_input = preds.unsqueeze(1)

                for j, (pred, status) in enumerate(zip(preds, all_done)):
                    if not status:
                        batch_predictions[j].append(int(pred))

                token_count += inference_token_count
                inference_token_count = batch_decoder_input.shape[-1]
                max_position_id = torch.max(decoder_position_ids).item()
                decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64, device=model.device).cumsum(0) - 1 + max_position_id

                if settings.RECOGNITION_STATIC_CACHE:
                    batch_decoder_input = pad_to_batch_size(batch_decoder_input, batch_size)

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)
        detected_text = processor.tokenizer.batch_decode(batch_predictions)
        detected_text = [truncate_repetitions(dt) for dt in detected_text]

        # Postprocess to fix LaTeX output (add $$ signs, etc)
        detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
        output_text.extend(detected_text)
        confidences.extend(sequence_scores.tolist())

        del encoder_text_hidden_states

    output_text = sorted(zip(indices, output_text), key=lambda x: x[0])
    confidences = sorted(zip(indices, confidences), key=lambda x: x[0])
    output_text = [text for _, text in output_text]
    confidences = [conf for _, conf in confidences]
    return output_text, confidences






