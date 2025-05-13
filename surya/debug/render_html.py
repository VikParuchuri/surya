import html as htmllib
import os.path
import re

filepath = os.path.abspath(__file__)

def render_text_as_html(
        bboxes: list[list[int]],
        texts: list[str],
        image_size: tuple[int, int],
        base_font_size: int = 16,
        scaler: int = 2
):
    katex_path = os.path.join(os.path.dirname(filepath), "katex.js")
    with open(katex_path, "r") as f:
        katex_script = f.read()

    html_content = []
    image_size = tuple([int(s * scaler) for s in image_size])
    width, height = image_size


    html_content.append(f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            width: {width}px;
            height: {height}px;
            position: relative;
            overflow: hidden;
            background: white;
            color: black;
        }}
        .text-box {{
            position: absolute;
            overflow: hidden;
            display: flex;
            justify-content: left;
            font-family: Arial, sans-serif;
            white-space: pre-wrap;
        }}
        .vertical-text {{
          writing-mode: vertical-rl;  /* Top to bottom, right to left */
        }}
    </style>
    {katex_script}
</head>
<body>
""")

    for i, (bbox, text) in enumerate(zip(bboxes, texts)):
        bbox = bbox.copy()
        bbox = [int(bb * scaler) for bb in bbox]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        min_dim = min(width, height)

        # Scale font size based on box height
        font_size = min(int(min_dim * 0.75), base_font_size)

        # Create div with absolute positioning
        div_style = (
            f"left: {x1}px; "
            f"top: {y1}px; "
            f"width: {width}px; "
            f"height: {height}px; "
            f"font-size: {font_size}px;"
        )

        class_ = "text-box"
        if height > width * 2:
            class_ += " vertical-text"

        # Determine if content is HTML/MathML or plain text
        if "<" in text and ">" in text and re.search(r"<(html|math|div|sub|sup|i|u|mark|small|del|b|br|code)\b", text.lower()):
            # Content is already HTML/MathML, include as-is
            html_content.append(f'<span class="{class_}" id="box-{i}" style="{div_style}">{text}</span>')
        else:
            # Plain text, escape it
            escaped_text = htmllib.escape(text)
            html_content.append(f'<span class="{class_}" id="box-{i}" style="{div_style}">{escaped_text}</span>')

    html_content.append("</body></html>")

    return "\n".join(html_content), image_size