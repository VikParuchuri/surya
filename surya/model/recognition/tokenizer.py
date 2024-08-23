import json
import os
from itertools import chain
import random
from typing import List, Optional, Tuple, Union
from tokenizers import AddedToken
from transformers import ByT5Tokenizer
import numpy as np
import torch
from surya.model.recognition.config import LANGUAGE_MAP, TOTAL_TOKENS, TOKEN_OFFSET

TOKENS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tokens.json")


class Byt5LangTokenizer(ByT5Tokenizer):
    def __init__(self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=None,
        token_path=TOKENS_FILE,
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
        with open(token_path, "r", encoding="utf-8") as f:
            token_map = json.load(f)
            self.token_map = {token_map[v]: v + TOKEN_OFFSET for v in range(len(token_map))}
            self.reverse_token_map = {v: k for k, v in self.token_map.items()}

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
            tokens, lang_list = self._inner_tokenize(text, lang)
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
        text = "".join([self.reverse_token_map[t] for t in token_ids])
        return text

    def _inner_tokenize(self, text: str, langs: List[str] | None, add_eos: bool = False, add_bos: bool = True):
        tokens = []
        i = 0
        while i < len(text):
            # Try to match a two-character token first
            if i + 1 < len(text) and text[i:i + 2] in self.token_map:
                tokens.append(self.token_map[text[i:i + 2]])
                i += 2
            # Fallback to one-character token
            elif text[i:i + 1] in self.token_map:
                tokens.append(self.token_map[text[i:i + 1]])
                i += 1
            else:
                tokens.append(self.unk_id)
                i += 1

        lang_list = []
        if langs:
            for lang in langs:
                code = LANGUAGE_MAP[lang]
                lang_list.append(code + TOKEN_OFFSET + TOTAL_TOKENS)

        tokens = lang_list + tokens

        if add_bos:
            tokens.insert(0, self.eos_id)

        return tokens, lang_list