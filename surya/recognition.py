from typing import List
import torch
from PIL import Image

from surya.input.processing import convert_if_not_rgb
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


def get_encoder_batch_size():
    batch_size = settings.RECOGNITION_BATCH_SIZE
    if batch_size is not None:
        batch_size = int(batch_size/4)
        batch_size = max(1, batch_size)
    else:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = get_batch_size()
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = get_batch_size()
    return batch_size


def batch_recognition(images: List, languages: List[List[str]], model, processor, batch_size=None):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)

    images = [image.convert("RGB") for image in images] # also copies the images
    if batch_size is None:
        batch_size = get_batch_size()

    # Sort images by width, so similar length ones go together
    sorted_pairs = sorted(enumerate(images), key=lambda x: x[1].width, reverse=False)
    indices, images = zip(*sorted_pairs)
    indices = list(indices)
    images = list(images)

    output_text = []
    confidences = []

    processed_batches = processor(text=[""] * len(images), images=images)

    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_langs = languages[i:i+batch_size]
        has_math = ["_math" in lang for lang in batch_langs]

        batch_pixel_values = processed_batches["pixel_values"][i:i+batch_size]

        batch_decoder_input = [[model.config.decoder_start_token_id] for lang in batch_langs]
        current_batch_size = len(batch_pixel_values)

        batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=model.dtype, device=model.device)
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)

        token_count = 0
        inference_token_count = batch_decoder_input.shape[-1]
        batch_predictions = [[] for _ in range(current_batch_size)]

        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64, device=model.device).cumsum(0) - 1
        model.decoder.model._setup_cache(model.config, 1, model.device, model.dtype)

        sequence_scores = None
        all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)

        with torch.no_grad(): # inference_mode doesn't work with torch.compile
            # Run post-prefill tokens
            encoder_hidden_states = []
            encoder_batch_size = get_encoder_batch_size()
            for i in range(0, len(batch_pixel_values), encoder_batch_size):
                pixel_batch = batch_pixel_values[i:i+encoder_batch_size]
                encoder_hidden_states.append(model.encoder(pixel_values=pixel_batch).last_hidden_state)
            encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)

            while token_count < settings.RECOGNITION_MAX_TOKENS:
                is_prefill = token_count == 0
                return_dict = model.decoder(
                    input_ids=batch_decoder_input,
                    encoder_hidden_states=encoder_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1
                logits = return_dict["logits"][:current_batch_size] # Ignore batch padding
                preds = torch.argmax(logits[:, -1], dim=-1)
                scores = torch.max(F.softmax(logits, dim=-1), dim=-1).values
                done = (preds == processor.tokenizer.eos_id) | (preds == processor.tokenizer.pad_id)
                done = done
                all_done = all_done | done

                scores[all_done == 1] = 0

                if is_prefill:
                    sequence_scores = scores
                else:
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                if all_done.all():
                    break

                batch_decoder_input = preds.unsqueeze(1)

                for j, (pred, status) in enumerate(zip(preds, all_done)):
                    if not status:
                        batch_predictions[j].append(int(pred))

                token_count += inference_token_count
                inference_token_count = batch_decoder_input.shape[-1]

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)
        detected_text = processor.tokenizer.batch_decode(batch_predictions)
        detected_text = [truncate_repetitions(dt) for dt in detected_text]

        # Postprocess to fix LaTeX output (add $$ signs, etc)
        detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
        output_text.extend(detected_text)
        confidences.extend(sequence_scores.tolist())

    output_text = sorted(zip(indices, output_text), key=lambda x: x[0])
    confidences = sorted(zip(indices, confidences), key=lambda x: x[0])
    output_text = [text for _, text in output_text]
    confidences = [conf for _, conf in confidences]
    return output_text, confidences






