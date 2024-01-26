from itertools import chain
from typing import List, Union
from transformers import ByT5Tokenizer
import numpy as np
import torch
from surya.model.recognition.config import LANGUAGE_MAP, TOTAL_TOKENS, TOKEN_OFFSET

def _tokenize(text: str, langs: List[str], eos_token_id: int = 1, add_eos: bool = True, add_bos: bool = True):
    byte_codes = []
    for char in text:
        # Add 3 to account for special tokens
        byte_codes.append([byte + TOKEN_OFFSET for byte in char.encode('utf-8')])

    tokens = list(chain.from_iterable(byte_codes))

    lang_list = []
    for lang in langs:
        code = LANGUAGE_MAP[lang]
        lang_list.append(code + TOKEN_OFFSET + TOTAL_TOKENS)

    tokens = lang_list + tokens

    if add_eos:
        tokens.append(eos_token_id)
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

    def __call__(self, texts: List[str] | str, langs: List[List[str]] | List[str], pad_token_id: int = 0, **kwargs):
        tokenized = []
        all_langs = []

        is_list = True
        # Convert to list of lists format
        if isinstance(texts, str):
            texts = [texts]
            is_list = False

        if isinstance(langs[0], str):
            langs = [langs]

        # One language input per text input
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
        text = bytes(token_ids).decode('utf-8', errors='ignore')
        return text
