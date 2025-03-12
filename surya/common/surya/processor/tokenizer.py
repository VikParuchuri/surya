import re
from typing import List, Union, Dict
import numpy as np
import torch

from transformers import ByT5Tokenizer, PreTrainedTokenizer

from surya.common.s3 import S3DownloaderMixin


def create_token_regex(tokens):
    escaped_tokens = [re.escape(token) for token in tokens]
    escaped_tokens.sort(key=len, reverse=True)
    pattern = r"^(" + "|".join(escaped_tokens) + r")"
    regex = re.compile(pattern)
    return regex


class SuryaOCRTokenizer(S3DownloaderMixin, PreTrainedTokenizer):
    def __init__(self, special_tokens: Dict[str, list] | None = None, **kwargs):
        if special_tokens is None:
            special_tokens = dict()

        all_special_tokens = special_tokens.get("all", [])

        self.special_tokens = special_tokens
        self.SPECIAL_TOKEN_MAPPING = {}

        idx = 0
        for tag in all_special_tokens:
            if tag in self.SPECIAL_TOKEN_MAPPING:
                continue
            self.SPECIAL_TOKEN_MAPPING[tag] = idx  # Assign token ID
            idx += 1

        self.REVERSE_SPECIAL_TOKEN_MAPPING = {
            v: k for k, v in self.SPECIAL_TOKEN_MAPPING.items()
        }
        self.SPECIAL_TOKEN_OFFSET = idx
        self.FORMAT_TAG_PATTERN = create_token_regex(special_tokens["formatting"])
        self.MATH_TOKEN_PATTERN = create_token_regex(special_tokens["math_internal"])
        self.MATH_TAG_PATTERN = create_token_regex(special_tokens["math_external"])
        self.MATH_TAG_START = "<math"
        self.MATH_END_TAG = "</math>"

        super().__init__()

    @property
    def vocab_size(self):
        return 65535 + self.SPECIAL_TOKEN_OFFSET

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        in_math = False
        while text:
            # Check for next math token
            if in_math:
                # If we're in a math block, check to see if we have a special math tag in the text
                match = self.MATH_TOKEN_PATTERN.search(text)
                if match:
                    # We found a math token
                    tokens.append(
                        self.SPECIAL_TOKEN_MAPPING[match.group(0)]
                    )  # Use special token ID
                    text = text[match.end() :]
                    continue

            # Check for math tag
            match = self.MATH_TAG_PATTERN.search(text)
            if match:
                # We found a tag
                tag = match.group(1)
                if tag.startswith(self.MATH_TAG_START):
                    in_math = True
                elif tag == self.MATH_END_TAG:
                    in_math = False
                tokens.append(self.SPECIAL_TOKEN_MAPPING[tag])  # Use special token ID
                text = text[match.end() :]
                continue

            # Check for tags
            match = self.FORMAT_TAG_PATTERN.search(text)
            if match:
                # We found a tag
                tag = match.group(1)
                tokens.append(self.SPECIAL_TOKEN_MAPPING[tag])  # Use special token ID
                text = text[match.end() :]
                continue

            # General case, utf-16 tokenization
            utf_16_tokens = self.text_to_utf16_numbers(text[0])
            tokens += [t + self.SPECIAL_TOKEN_OFFSET for t in utf_16_tokens]
            text = text[1:]

        return tokens

    def text_to_utf16_numbers(self, text: str):
        """Converts text to UTF-16 encoded numbers."""
        utf16_bytes = text.encode(
            "utf-16le"
        )  # Little-endian to simplify byte order handling
        numbers = []

        for i in range(0, len(utf16_bytes), 2):
            # Combine two adjacent bytes into a single number
            number = utf16_bytes[i] + (utf16_bytes[i + 1] << 8)
            numbers.append(number)

        return numbers

    def utf16_numbers_to_text(self, numbers):
        """Converts UTF-16 numbers back to text."""
        byte_array = bytearray()
        for number in numbers:
            byte_array.append(number & 0xFF)  # Lower byte
            byte_array.append((number >> 8) & 0xFF)  # Upper byte

        try:
            text = byte_array.decode("utf-16le", errors="ignore")
        except Exception as e:
            print(f"Error decoding utf16: {e}")
            text = ""

        return text

    def __call__(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Dict[str, List[List[int]]]:
        """Tokenizes text and returns input IDs."""
        tokenized = []

        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            tokens = self._tokenize(text)
            tokenized.append(tokens)

        return {"input_ids": tokenized}

    def decode(self, token_ids, **kwargs):
        """Decodes token IDs back to text."""
        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = token_ids.tolist()

        decoded_text = ""
        token_buffer = []
        for t in token_ids:
            if t >= self.SPECIAL_TOKEN_OFFSET:
                token_buffer.append(t - self.SPECIAL_TOKEN_OFFSET)
            elif t in self.REVERSE_SPECIAL_TOKEN_MAPPING:
                if token_buffer:
                    decoded_text += self.utf16_numbers_to_text(token_buffer)
                    token_buffer = []
                decoded_text += self.REVERSE_SPECIAL_TOKEN_MAPPING[t]
            else:
                raise ValueError(
                    f'Unexpected token value while decoding, got "{t}" in token_ids {token_ids}'
                )

        # Detokenize remaining tokens
        if token_buffer:
            decoded_text += self.utf16_numbers_to_text(token_buffer)

        return decoded_text
