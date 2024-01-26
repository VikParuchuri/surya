from PIL import Image, ImageDraw, ImageFont
from surya.settings import settings


def get_text_size(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def draw_text_on_image(bboxes, texts, image_size=(1024, 1024), font_path=settings.RECOGNITION_RENDER_FONT, font_size=14):
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)

    for bbox, text in zip(bboxes, texts):
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # Shrink the text to fit in the bbox if needed
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = get_text_size(text, font)
        while (text_width > bbox_width or text_height > bbox_height) and font_size > 6:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = get_text_size(text, font)

        # Calculate text position (centered in bbox)
        text_width, text_height = get_text_size(text, font)
        x = bbox[0] + (bbox_width - text_width) / 2
        y = bbox[1] + (bbox_height - text_height) / 2

        draw.text((x, y), text, fill="black", font=font)

    return image
