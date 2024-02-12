from typing import List
from surya.languages import LANGUAGE_TO_CODE, CODE_TO_LANGUAGE


def replace_lang_with_code(langs: List[str]):
    for i in range(len(langs)):
        if langs[i].title() in LANGUAGE_TO_CODE:
            langs[i] = LANGUAGE_TO_CODE[langs[i].title()]
        if langs[i] not in CODE_TO_LANGUAGE:
            raise ValueError(f"Language code {langs[i]} not found.")


def get_unique_langs(langs: List[List[str]]):
    uniques = []
    for lang_list in langs:
        for lang in lang_list:
            if lang not in uniques:
                uniques.append(lang)
    return uniques