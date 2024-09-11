import re
from ftfy import fix_text
from surya.settings import settings


def contains_math(text):
    return text.startswith(settings.MATH_FENCE_CHAR) or text.endswith(settings.MATH_FENCE_CHAR)


def fix_math(text):
    # Fix any issues with the text
    text = fix_text(text)

    # Remove LaTeX labels and references
    text = remove_labels(text)
    text = replace_katex_invalid(text)
    text = fix_fences(text)
    return text


def remove_labels(text):
    pattern = r'\\label\{[^}]*\}'
    text = re.sub(pattern, '', text)

    ref_pattern = r'\\ref\{[^}]*\}'
    text = re.sub(ref_pattern, '', text)

    pageref_pattern = r'\\pageref\{[^}]*\}'
    text = re.sub(pageref_pattern, '', text)
    return text


def replace_katex_invalid(string):
    # KaTeX cannot render all LaTeX, so we need to replace some things
    string = re.sub(r'\\tag\{.*?\}', '', string)
    string = re.sub(r'\\(?:Bigg?|bigg?)\{(.*?)\}', r'\1', string)
    string = re.sub(r'\\quad\\mbox\{(.*?)\}', r'\1', string)
    string = re.sub(r'\\mbox\{(.*?)\}', r'\1', string)
    string = remove_inner_dollars(string)
    return string


def remove_inner_dollars(text):
    def replace_dollar(match):
        # Replace single $ with nothing, keep $$ intact
        math_block = match.group(1)
        return settings.MATH_FENCE_CHAR + math_block.replace("$", '') + settings.MATH_FENCE_CHAR

    pattern = rf'{settings.MATH_FENCE_CHAR}(.*?){settings.MATH_FENCE_CHAR}'
    return re.sub(pattern, replace_dollar, text, flags=re.DOTALL)


def extract_latex_with_positions(text):
    pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    matches = []
    for match in re.finditer(pattern, text, re.DOTALL):
        matches.append((match.group(), match.start(), match.end()))
    return matches


def is_latex(text):
    latex_patterns = [
        r'\\(?:begin|end)\{[a-zA-Z]*\}',
        rf'{settings.MATH_FENCE_CHAR}.*?{settings.MATH_FENCE_CHAR}',
        r'\$\$.*?\$\$',
        r'\\[a-zA-Z]+',
        r'\\[^a-zA-Z]',
    ]

    combined_pattern = '|'.join(latex_patterns)
    if re.search(combined_pattern, text, re.DOTALL):
        return True

    return False


def fix_fences(text):
    if text.startswith(settings.MATH_FENCE_CHAR) and not text.endswith(settings.MATH_FENCE_CHAR):
        text += settings.MATH_FENCE_CHAR

    if not text.startswith(settings.MATH_FENCE_CHAR) and text.endswith(settings.MATH_FENCE_CHAR):
        text = settings.MATH_FENCE_CHAR + text

    return text


