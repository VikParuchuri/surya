from playwright.sync_api import sync_playwright
from PIL import Image
import io


def latex_to_pil(latex_code, target_width, target_height, fontsize=18):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css">
        <script src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js"></script>
        <style>
            body {
                margin: 0;
                padding: 0;
                display: flex;
            }
            #content {
                font-size: {fontsize}px;
            }
        </style>
    </head>
    <body>
        <div id="content">{content}</div>
        <script>
            function renderMath() {
                let content = document.getElementById('content');
                let html = content.innerHTML;

                // Replace display equations
                html = html.replace(/\\$\\$(.*?)\\$\\$/gs, (match, equation) => {
                    let span = document.createElement('span');
                    katex.render(equation, span, { displayMode: true, throwOnError: false, errorColor: '#000' });
                    return span.outerHTML;
                });

                // Replace inline equations
                html = html.replace(/\\$(.*?)\\$/g, (match, equation) => {
                    if(match.startsWith('\\\\$')) return match; // Ignore escaped dollars
                    let span = document.createElement('span');
                    katex.render(equation, span, { displayMode: false, throwOnError: false, errorColor: '#000' });
                    return span.outerHTML;
                });

                content.innerHTML = html;
            }

            renderMath();
        </script>
    </body>
    </html>
    """

    formatted_latex = latex_code.replace('\n', '\\n').replace('"', '\\"')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({'width': target_width, 'height': target_height})

        while fontsize <= 30:
            html_content = html_template.replace("{content}", formatted_latex).replace("{fontsize}", str(fontsize))
            page.set_content(html_content)

            dimensions = page.evaluate("""() => {
                const render = document.getElementById('content');
                return {
                    width: render.offsetWidth,
                    height: render.offsetHeight
                };
            }""")

            if dimensions['width'] >= target_width or dimensions['height'] >= target_height:
                fontsize -= 1
                break
            else:
                fontsize += 1

        html_content = html_template.replace("{content}", formatted_latex).replace("{fontsize}", str(fontsize))
        page.set_content(html_content)

        screenshot_bytes = page.screenshot()
        browser.close()

        image_stream = io.BytesIO(screenshot_bytes)
        pil_image = Image.open(image_stream)
        pil_image.load()
        return pil_image