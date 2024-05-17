import PIL

from surya.input.processing import open_pdf, get_page_images
import os
import filetype
from PIL import Image
import json


def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]


def load_pdf(pdf_path, max_pages=None, start_page=None):
    doc = open_pdf(pdf_path)
    last_page = len(doc)

    if start_page:
        assert start_page < last_page and start_page >= 0, f"Start page must be between 0 and {last_page}"
    else:
        start_page = 0

    if max_pages:
        assert max_pages >= 0, f"Max pages must be greater than 0"
        last_page = min(start_page + max_pages, last_page)

    page_indices = list(range(start_page, last_page))
    images = get_page_images(doc, page_indices)
    doc.close()
    names = [get_name_from_path(pdf_path) for _ in page_indices]
    return images, names


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    name = get_name_from_path(image_path)
    return [image], [name]


def load_from_file(input_path, max_pages=None, start_page=None):
    input_type = filetype.guess(input_path)
    if input_type.extension == "pdf":
        return load_pdf(input_path, max_pages, start_page)
    else:
        return load_image(input_path)


def load_from_folder(folder_path, max_pages=None, start_page=None):
    image_paths = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path) if not image_name.startswith(".")]
    image_paths = [ip for ip in image_paths if not os.path.isdir(ip)]

    images = []
    names = []
    for path in image_paths:
        extension = filetype.guess(path)
        if extension and extension.extension == "pdf":
            image, name = load_pdf(path, max_pages, start_page)
            images.extend(image)
            names.extend(name)
        else:
            try:
                image, name = load_image(path)
                images.extend(image)
                names.extend(name)
            except PIL.UnidentifiedImageError:
                print(f"Could not load image {path}")
                continue
    return images, names


def load_lang_file(lang_path, names):
    with open(lang_path, "r") as f:
        lang_dict = json.load(f)
    return [lang_dict[name].copy() for name in names]
