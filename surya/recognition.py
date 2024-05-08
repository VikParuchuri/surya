from typing import List
import torch
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
            batch_size = 256
    return batch_size


def batch_recognition(images: List, languages: List[List[str]], model, processor, batch_size=None):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)

    if batch_size is None:
        batch_size = get_batch_size()

    images = [image.convert("RGB") for image in images]

    output_text = []
    confidences = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_langs = languages[i:i+batch_size]
        has_math = ["_math" in lang for lang in batch_langs]
        batch_images = images[i:i+batch_size]
        model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

        batch_pixel_values = model_inputs["pixel_values"]
        batch_langs = model_inputs["langs"]
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

        batch_langs = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

        with torch.inference_mode():
            return_dict = model.generate(
                pixel_values=batch_pixel_values,
                decoder_input_ids=batch_decoder_input,
                decoder_langs=batch_langs,
                eos_token_id=processor.tokenizer.eos_id,
                max_new_tokens=settings.RECOGNITION_MAX_TOKENS,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_ids = return_dict["sequences"]

            # Find confidence scores
            scores = return_dict["scores"] # Scores is a tuple, one per new sequence position.  Each tuple element is bs x vocab_size
            sequence_scores = torch.zeros(generated_ids.shape[0])
            sequence_lens = torch.where(
                generated_ids > processor.tokenizer.eos_id,
                torch.ones_like(generated_ids),
                torch.zeros_like(generated_ids)
            ).sum(axis=-1).cpu()
            prefix_len = generated_ids.shape[1] - len(scores) # Length of passed in tokens (bos, langs)
            for token_idx, score in enumerate(scores):
                probs = F.softmax(score, dim=-1)
                max_probs = torch.max(probs, dim=-1).values
                max_probs = torch.where(
                    generated_ids[:, token_idx + prefix_len] <= processor.tokenizer.eos_id,
                    torch.zeros_like(max_probs),
                    max_probs
                ).cpu()
                sequence_scores += max_probs
            sequence_scores /= sequence_lens

        detected_text = processor.tokenizer.batch_decode(generated_ids)
        detected_text = [truncate_repetitions(dt) for dt in detected_text]
        # Postprocess to fix LaTeX output (add $$ signs, etc)
        detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
        output_text.extend(detected_text)
        confidences.extend(sequence_scores.tolist())

    return output_text, confidences






