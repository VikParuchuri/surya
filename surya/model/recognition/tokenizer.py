from itertools import chain
import random
from typing import List, Optional, Tuple, Union
from tokenizers import AddedToken
from transformers import ByT5Tokenizer
import numpy as np
import torch

TOTAL_TOKENS = 65536
TOKEN_OFFSET = 3 # Pad, eos, bos
SPECIAL_TOKENS = 253
TOTAL_VOCAB_SIZE = TOTAL_TOKENS + TOKEN_OFFSET + SPECIAL_TOKENS


def replace_random_chars(text, thresh=.01, shift_range=100):
    if random.random() < thresh and len(text) >= 3:
        try:
            pos = random.randint(1, len(text) - 2)
            char = text[pos]
            shift_amt = random.randint(-shift_range, shift_range)
            new_char_code = ord(char) + shift_amt
            if new_char_code < 0:
                new_char_code = 0
            elif new_char_code > 65535:
                new_char_code = 65535
            new_char = chr(new_char_code)
            char_choices = [random.choice(text), new_char]
            char_choice = random.choice(char_choices)
            text = text[:pos] + char_choice + text[pos + 1:]
        except ValueError:
            print(f"Failed to replace char in {text}")
            pass
    return text


def text_to_utf16_numbers(text):
    utf16_bytes = text.encode('utf-16le')  # Little-endian to simplify byte order handling

    numbers = []

    # Iterate through each pair of bytes and combine them into a single number
    for i in range(0, len(utf16_bytes), 2):
        # Combine two adjacent bytes into a single number
        number = utf16_bytes[i] + (utf16_bytes[i + 1] << 8)
        numbers.append(number)

    return numbers


def utf16_numbers_to_text(numbers):
    byte_array = bytearray()
    for number in numbers:
        # Extract the two bytes from the number and add them to the byte array
        byte_array.append(number & 0xFF)         # Lower byte
        byte_array.append((number >> 8) & 0xFF)  # Upper byte

    text = byte_array.decode('utf-16le', errors="ignore")
    return text


def _tokenize(text: str, eos_token_id: int = 1, add_eos: bool = False, add_bos: bool = False):
    tokens = text_to_utf16_numbers(text)
    tokens = [t + TOKEN_OFFSET for t in tokens] # Account for special pad, etc, tokens

    if add_eos:
        tokens.append(eos_token_id)
    if add_bos:
        tokens.insert(0, eos_token_id)

    return tokens


class Byt5LangTokenizer(ByT5Tokenizer):
    def __init__(self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=None,
        **kwargs,
    ):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.bos_token = eos_token
        self.offset = TOKEN_OFFSET

        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2

        self.model_max_length = model_max_length
        self.special_token_start = TOKEN_OFFSET + TOTAL_TOKENS

        super().__init__()

    def __call__(self, texts: List[str] | str, pad_token_id: int = 0, **kwargs):
        tokenized = []

        is_list = True
        # Convert to list of lists format
        if isinstance(texts, str):
            texts = [texts]
            is_list = False

        for text in texts:
            tokens = _tokenize(text)
            tokenized.append(tokens)

        # Convert back to flat format
        if not is_list:
            tokenized = tokenized[0]

        return {"input_ids": tokenized}

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = token_ids.tolist()

        token_ids = [t for t in token_ids if TOKEN_OFFSET <= t < self.special_token_start]
        token_ids = [t - TOKEN_OFFSET for t in token_ids]
        text = utf16_numbers_to_text(token_ids)
        return text
