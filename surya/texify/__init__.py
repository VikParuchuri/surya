import re
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from surya.common.predictor import BasePredictor
from surya.settings import settings
from surya.texify.loader import TexifyModelLoader
from surya.texify.schema import TexifyResult


class TexifyPredictor(BasePredictor):
    model_loader_cls = TexifyModelLoader
    batch_size = settings.TEXIFY_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 2,
        "mps": 6,
        "cuda": 48
    }

    def __call__(self, images: List[Image.Image], batch_size: int | None = None) -> List[TexifyResult]:
        text, confidences = self.batch_texify(images, batch_size=batch_size)
        return [TexifyResult(text=self.fix_fences(t), confidence=c) for t, c in zip(text, confidences)]

    def prepare_input(self, images: List[Image.Image], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_images = [img.convert("RGB") for img in images]
        processed = self.processor(batch_images)
        batch_pixel_values = processed["pixel_values"].to(self.model.device).to(self.model.dtype)
        batch_input_ids = processed["input_ids"].to(self.model.device).to(torch.long)

        if settings.TEXIFY_STATIC_CACHE:
            batch_pixel_values = self.pad_to_batch_size(batch_pixel_values, batch_size)
            batch_input_ids = self.pad_to_batch_size(batch_input_ids, batch_size)

        return batch_pixel_values, batch_input_ids

    def fix_fences(self, text: str) -> str:
        if text.count("$") % 2 != 0:
            return text

        text = re.sub(r'\$\$(.*?)\$\$', r'<math display="block">\1</math>', text, flags=re.DOTALL)
        text = re.sub(r'\$(.*?)\$', r'<math>\1</math>', text, flags=re.DOTALL)
        return text


    def batch_texify(self, images: List[Image.Image], batch_size: int | None) -> Tuple[List[str], List[float]]:
        if batch_size is None:
            batch_size = self.get_batch_size()

        # Sort images by width, so similar length ones go together
        sorted_pairs = sorted(enumerate(images), key=lambda x: x[1].width, reverse=False)
        indices, images = zip(*sorted_pairs)
        indices = list(indices)
        images = list(images)

        output_text = []
        confidences = []
        for i in tqdm(range(0, len(images), batch_size), desc="Texify inference"):
            batch = images[i:i+batch_size]
            batch_pixel_values, batch_input_ids = self.prepare_input(batch, batch_size)
            current_batch_size = len(batch)

            token_count = 0
            inference_token_count = batch_input_ids.shape[-1]
            batch_predictions = [[] for _ in range(current_batch_size)]

            decoder_position_ids = torch.ones_like(batch_input_ids[0, :], dtype=torch.int64, device=self.model.device).cumsum(0) - 1
            self.model.decoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)

            sequence_scores = None
            all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=self.model.device)

            with torch.inference_mode():
                encoder_hidden_states = self.model.encoder(pixel_values=batch_pixel_values).last_hidden_state

                while token_count < settings.TEXIFY_MAX_TOKENS - 1:
                    is_prefill = token_count == 0

                    return_dict = self.model.decoder(
                        input_ids=batch_input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        cache_position=decoder_position_ids,
                        use_cache=True,
                        prefill=is_prefill
                    )

                    decoder_position_ids = decoder_position_ids[-1:] + 1
                    logits = return_dict["logits"][:current_batch_size]  # Ignore batch padding

                    preds = torch.argmax(logits[:, -1], dim=-1)
                    scores = torch.max(F.softmax(logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)
                    done = (preds == self.processor.tokenizer.eos_token_id) | (preds == self.processor.tokenizer.pad_token_id)
                    all_done = all_done | done

                    if is_prefill:
                        sequence_scores = scores
                    else:
                        scores = scores.masked_fill(all_done, 0)
                        sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                    if all_done.all():
                        break

                    batch_input_ids = preds.unsqueeze(1)

                    for j, (pred, status) in enumerate(zip(preds, all_done)):
                        if not status:
                            batch_predictions[j].append(int(pred))

                    token_count += inference_token_count
                    inference_token_count = batch_input_ids.shape[-1]
                    max_position_id = torch.max(decoder_position_ids).item()
                    decoder_position_ids = torch.ones_like(batch_input_ids[0, :], dtype=torch.int64,
                                                           device=self.model.device).cumsum(0) - 1 + max_position_id

                    if settings.TEXIFY_STATIC_CACHE:
                        batch_input_ids = self.pad_to_batch_size(batch_input_ids, batch_size)

            batch_confidences = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)
            detected_text = self.processor.tokenizer.batch_decode(batch_predictions)

            batch_confidences = batch_confidences.tolist()

            if settings.TEXIFY_STATIC_CACHE:
                detected_text = detected_text[:current_batch_size]
                batch_confidences = batch_confidences[:current_batch_size]

            output_text.extend(detected_text)
            confidences.extend(batch_confidences)

        output_text = sorted(zip(indices, output_text), key=lambda x: x[0])
        confidences = sorted(zip(indices, confidences), key=lambda x: x[0])
        output_text = [text for _, text in output_text]
        confidences = [conf for _, conf in confidences]
        return output_text, confidences