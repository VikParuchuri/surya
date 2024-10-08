from typing import List

import cv2
import numpy as np
import math
import pypdfium2
from PIL import Image, ImageOps, ImageDraw
import torch
from surya.settings import settings


def convert_if_not_rgb(images: List[Image.Image]) -> List[Image.Image]:
    new_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        new_images.append(image)
    return new_images


def get_total_splits(image_size, processor):
    img_height = list(image_size)[1]
    max_height = settings.DETECTOR_IMAGE_CHUNK_HEIGHT
    processor_height = processor.size["height"]
    if img_height > max_height:
        num_splits = math.ceil(img_height / processor_height)
        return num_splits
    return 1


def split_image(img, processor):
    # This will not modify/return the original image - it will either crop, or copy the image
    img_height = list(img.size)[1]
    max_height = settings.DETECTOR_IMAGE_CHUNK_HEIGHT
    processor_height = processor.size["height"]
    if img_height > max_height:
        num_splits = math.ceil(img_height / processor_height)
        splits = []
        split_heights = []
        for i in range(num_splits):
            top = i * processor_height
            bottom = (i + 1) * processor_height
            if bottom > img_height:
                bottom = img_height
            cropped = img.crop((0, top, img.size[0], bottom))
            height = bottom - top
            if height < processor_height:
                cropped = ImageOps.pad(cropped, (img.size[0], processor_height), color=255, centering=(0, 0))
            splits.append(cropped)
            split_heights.append(height)
        return splits, split_heights
    return [img.copy()], [img_height]


def prepare_image_detection(img, processor):
    new_size = (processor.size["width"], processor.size["height"])

    # This double resize actually necessary for downstream accuracy
    img.thumbnail(new_size, Image.Resampling.LANCZOS)
    img = img.resize(new_size, Image.Resampling.LANCZOS) # Stretch smaller dimension to fit new size

    img = np.asarray(img, dtype=np.uint8)
    img = processor(img)["pixel_values"][0]
    img = torch.from_numpy(img)
    return img


def open_pdf(pdf_filepath):
    return pypdfium2.PdfDocument(pdf_filepath)


def get_page_images(doc, indices: List, dpi=settings.IMAGE_DPI):
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=indices,
        scale=dpi / 72,
    )
    images = list(renderer)
    images = [image.convert("RGB") for image in images]
    return images


def slice_bboxes_from_image(image: Image.Image, bboxes):
    lines = []
    for bbox in bboxes:
        line = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        if line.size[0] == 0:
            print(f"Warning: found an empty line with bbox {bbox}")
        lines.append(line)
    return lines


def slice_polys_from_image(image: Image.Image, polys):
    image_array = np.array(image, dtype=np.uint8)
    lines = []
    for idx, poly in enumerate(polys):
        lines.append(slice_and_pad_poly(image_array, poly))
    return lines


def slice_and_pad_poly(image_array: np.array, coordinates):
    # Draw polygon onto mask
    coordinates = [(corner[0], corner[1]) for corner in coordinates]
    bbox = [min([x[0] for x in coordinates]), min([x[1] for x in coordinates]), max([x[0] for x in coordinates]), max([x[1] for x in coordinates])]

    # We mask out anything not in the polygon
    cropped_polygon = image_array[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    coordinates = [(x - bbox[0], y - bbox[1]) for x, y in coordinates]

    # Pad the area outside the polygon with the pad value
    mask = np.zeros(cropped_polygon.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(coordinates)], 1)
    mask = np.stack([mask] * 3, axis=-1)

    cropped_polygon[mask == 0] = settings.RECOGNITION_PAD_VALUE
    rectangle_image = Image.fromarray(cropped_polygon)

    return rectangle_image
