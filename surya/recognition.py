from typing import List
import torch
from PIL import Image
from surya.settings import settings
from tqdm import tqdm
import numpy as np


def get_batch_size():
    batch_size = settings.RECOGNITION_BATCH_SIZE
    if batch_size is None:
        batch_size = 32
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 64 # 12GB RAM max
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 256
    return batch_size


def batch_recognition(images: List, languages: List[List[str]], model, processor):
    assert all([isinstance(image, Image.Image) for image in images])
    assert len(images) == len(languages)
    batch_size = get_batch_size()

    images = [image.convert("RGB") for image in images]

    output_text = []
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
        batch_langs = languages[i:i+batch_size]
        batch_images = images[i:i+batch_size]
        model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

        batch_pixel_values = model_inputs["pixel_values"]
        batch_langs = model_inputs["langs"]
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

        batch_langs = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
        batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values=batch_pixel_values,
                decoder_input_ids=batch_decoder_input,
                decoder_langs=batch_langs,
                eos_token_id=processor.tokenizer.eos_id,
                max_new_tokens=settings.RECOGNITION_MAX_TOKENS
            )

        output_text.extend(processor.tokenizer.batch_decode(generated_ids))

    return output_text






