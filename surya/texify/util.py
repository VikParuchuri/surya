import re

def convert_math_delimiters(text):
    text = re.sub(r'<math display="block">(.*?)</math>', r'$$\1$$', text)
    text = re.sub(r'<math>(.*?)</math>', r'$\1$', text)
    return text