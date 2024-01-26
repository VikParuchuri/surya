from typing import List
import torch
from PIL import Image
from surya.settings import settings
from tqdm import tqdm


def batch_recognition(images: List, languages: List[List[str]], model, processor):
    assert all([isinstance(image, Image.Image) for image in images])

    images = [image.convert("RGB") for image in images]
    model_inputs = processor(text=[""] * len(languages), images=images, lang=languages)

    output_text = []
    for i in tqdm(range(0, len(model_inputs["pixel_values"]), settings.RECOGNITION_BATCH_SIZE), desc="Recognizing Text"):
        batch_langs = model_inputs["langs"][i:i+settings.RECOGNITION_BATCH_SIZE]
        batch_pixel_values = model_inputs["pixel_values"][i:i+settings.RECOGNITION_BATCH_SIZE]
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

        batch_langs = torch.tensor(batch_langs, dtype=torch.long).to(model.device)
        batch_pixel_values = torch.tensor(batch_pixel_values, dtype=model.dtype).to(model.device)
        batch_decoder_input = torch.tensor(batch_decoder_input, dtype=torch.long).to(model.device)

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






