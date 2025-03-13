import html as htmllib
import re


def render_text_as_html(
        bboxes: list[list[int]],
        texts: list[str],
        image_size: tuple[int, int],
        base_font_size: int = 24,
        scaler: int = 2
):
    html_content = []
    image_size = tuple([int(s * scaler) for s in image_size])

    html_content.append("""
<html>
<head>
    <style>
        body {
            margin: 0;
            padding: 0;
            width: %dpx;
            height: %dpx;
            position: relative;
            overflow: hidden;
            background: white;
            color: black;
        }
        .text-box {
            position: absolute;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
            white-space: pre-wrap;
        }
        /* Add MathML styling */
        math {
            display: inline-block;
        }
    </style>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
""" % image_size)

    for i, (bbox, text) in enumerate(zip(bboxes, texts)):
        bbox = bbox.copy()
        bbox = [int(bb * scaler) for bb in bbox]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Scale font size based on box height
        font_size = min(int(height * 0.75), base_font_size)

        # Create div with absolute positioning
        div_style = (
            f"left: {x1}px; "
            f"top: {y1}px; "
            f"width: {width}px; "
            f"height: {height}px; "
            f"font-size: {font_size}px;"
        )

        # Determine if content is HTML/MathML or plain text
        if "<" in text and ">" in text and re.search(r"<(html|math|div|sub|sup|i|u|mark|small|del|b|br|code)\b", text.lower()):
            # Content is already HTML/MathML, include as-is
            html_content.append(f'<span class="text-box" id="box-{i}" style="{div_style}">{text}</span>')
        else:
            # Plain text, escape it
            escaped_text = htmllib.escape(text)
            html_content.append(f'<span class="text-box" id="box-{i}" style="{div_style}">{escaped_text}</span>')

    html_content.append("</body></html>")

    return "\n".join(html_content), image_size