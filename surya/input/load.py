from surya.input.processing import open_pdf, get_page_images
import os
import filetype
from PIL import Image


def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]


def load_pdf(pdf_path, max_pages=None):
    doc = open_pdf(pdf_path)
    page_count = len(doc)
    if max_pages:
        page_count = min(max_pages, page_count)

    page_indices = list(range(page_count))

    images = get_page_images(doc, page_indices)
    doc.close()
    names = [get_name_from_path(pdf_path) for _ in page_indices]
    return images, names


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    name = get_name_from_path(image_path)
    return [image], [name]


def load_from_file(input_path, max_pages=None):
    input_type = filetype.guess(input_path)
    if input_type.extension == "pdf":
        return load_pdf(input_path, max_pages)
    else:
        return load_image(input_path)


def load_from_folder(folder_path, max_pages=None):
    image_paths = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)]
    image_paths = [ip for ip in image_paths if not os.path.isdir(ip) and not ip.startswith(".")]

    images = []
    names = []
    for path in image_paths:
        if filetype.guess(path).extension == "pdf":
            image, name = load_pdf(path, max_pages)
            images.extend(image)
            names.extend(name)
        else:
            image, name = load_image(path)
            images.extend(image)
            names.extend(name)
    return images, names