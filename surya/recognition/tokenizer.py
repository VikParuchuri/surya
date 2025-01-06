from typing import List, Union
from transformers import ByT5Tokenizer
import numpy as np
import torch
from surya.recognition.model.config import LANGUAGE_MAP, TOTAL_TOKENS, TOKEN_OFFSET


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

    try:
        text = byte_array.decode('utf-16le', errors="ignore")
    except Exception as e:
        print(f"Error decoding utf16: {e}")
        text = ""

    return text


def _tokenize(text: str, langs: List[str] | None, eos_token_id: int = 1, add_eos: bool = False, add_bos: bool = True):
    tokens = text_to_utf16_numbers(text)
    tokens = [t + TOKEN_OFFSET for t in tokens] # Account for special pad, etc, tokens

    lang_list = []
    if langs:
        for lang in langs:
            code = LANGUAGE_MAP[lang]
            lang_list.append(code + TOKEN_OFFSET + TOTAL_TOKENS)

    tokens = lang_list + tokens

    if add_bos:
        tokens.insert(0, eos_token_id)

    return tokens, lang_list


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

    def __call__(self, texts: List[str] | str, langs: List[List[str]] | List[str] | None = None, pad_token_id: int = 0, **kwargs):
        tokenized = []
        all_langs = []

        is_list = True
        # Convert to list of lists format
        if isinstance(texts, str):
            texts = [texts]
            is_list = False

        if langs is None:
            langs = [None] * len(texts)

        if isinstance(langs[0], str):
            langs = [langs]

        assert len(langs) == len(texts)

        for text, lang in zip(texts, langs):
            tokens, lang_list = _tokenize(text, lang)
            tokenized.append(tokens)
            all_langs.append(lang_list)

        # Convert back to flat format
        if not is_list:
            tokenized = tokenized[0]
            all_langs = all_langs[0]

        return {"input_ids": tokenized, "langs": all_langs}

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
