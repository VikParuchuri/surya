import re
from ftfy import fix_text


def contains_math(text):
    return text.startswith("$") or text.endswith("$")


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
        return '$$' + math_block.replace('$', '') + '$$'

    pattern = r'\$\$(.*?)\$\$'
    return re.sub(pattern, replace_dollar, text, flags=re.DOTALL)


def extract_latex_with_positions(text):
    pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    matches = []
    for match in re.finditer(pattern, text, re.DOTALL):
        matches.append((match.group(), match.start(), match.end()))
    return matches


def slice_latex(text):
    # Extract LaTeX blocks along with their positions
    latex_blocks_with_positions = extract_latex_with_positions(text)

    chunks = []
    last_position = 0
    for block, start, end in latex_blocks_with_positions:
        # Add text before the current LaTeX block, if any
        if start > last_position:
            chunks.append({"text": text[last_position:start], "type": "text"})
        # Add the LaTeX block
        chunks.append({"text": block, "type": "latex"})
        last_position = end
    # Add remaining text after the last LaTeX block, if any
    if last_position < len(text):
        chunks.append({"text": text[last_position:], "type": "text"})

    return chunks


def is_latex(text):
    latex_patterns = [
        r'\\(?:begin|end)\{[a-zA-Z]*\}',
        r'\$.*?\$',
        r'\$\$.*?\$\$',
        r'\\[a-zA-Z]+',
        r'\\[^a-zA-Z]',
    ]

    combined_pattern = '|'.join(latex_patterns)
    if re.search(combined_pattern, text, re.DOTALL):
        return True

    return False


def fix_fences(text):
    if text.startswith("$$") and not text.endswith("$$"):
        if text[-1] == "$":
            text += "$"
        else:
            text += "$$"

    if text.endswith("$$") and not text.startswith("$$"):
        if text[0] == "$":
            text = "$" + text
        else:
            text = "$$" + text

    if text.startswith("$") and not text.endswith("$"):
        text = "$" + text + "$$"

    if text.endswith("$") and not text.startswith("$"):
        text = "$$" + text + "$"

    return text


def strip_fences(text):
    while text.startswith("$"):
        text = text[1:]
    while text.endswith("$"):
        text = text[:-1]
    return text


