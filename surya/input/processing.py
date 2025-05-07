from typing import List

import cv2
import numpy as np
import pypdfium2
from PIL import Image

from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def convert_if_not_rgb(images: List[Image.Image]) -> List[Image.Image]:
    new_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        new_images.append(image)
    return new_images


def open_pdf(pdf_filepath):
    return pypdfium2.PdfDocument(pdf_filepath)


def get_page_images(doc, indices: List, dpi=settings.IMAGE_DPI):
    images = [
        doc[i].render(scale=dpi / 72, draw_annots=False).to_pil() for i in indices
    ]
    images = [image.convert("RGB") for image in images]
    return images


def slice_bboxes_from_image(image: np.ndarray, bboxes):
    lines = []
    for bbox in bboxes:
        bbox = np.array(bbox, dtype=np.int32)
        bbox = np.clip(bbox, 0, None)  # Ensure no negative indices
        # Ensure bbox is within the image bounds
        if bbox[3] <= bbox[1]:
            bbox[3] = bbox[1] + 1

        if bbox[2] <= bbox[0]:
            bbox[2] = bbox[0] + 1

        bbox[2] = min(bbox[2], image.shape[1])
        bbox[3] = min(bbox[3], image.shape[0])

        line = image[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
        if line.size == 0:
            logger.warning(f"Warning: found an empty line with bbox {bbox}")
        lines.append(line)
    return lines


def slice_polys_from_image(image: np.ndarray, polys):
    lines = []
    for idx, poly in enumerate(polys):
        lines.append(slice_and_pad_poly(image, poly))
    return lines


def slice_and_pad_poly(image_array: np.array, coordinates):
    # Draw polygon onto mask
    coordinates = [(corner[0], corner[1]) for corner in coordinates]
    bbox = [
        min([x[0] for x in coordinates]),
        min([x[1] for x in coordinates]),
        max([x[0] for x in coordinates]),
        max([x[1] for x in coordinates]),
    ]

    # We mask out anything not in the polygon
    cropped_polygon = image_array[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
    height, width = cropped_polygon.shape[:2]

    coordinates = [(x - bbox[0], y - bbox[1]) for x, y in coordinates]

    # Validate the cropped area
    if any(
        [
            bbox[3] <= bbox[1] or bbox[2] <= bbox[0],
            len(coordinates) < 3,
            height == 0,
            width == 0,
        ]
    ):
        return cropped_polygon

    # Pad the area outside the polygon with the pad value
    try:
        mask = np.zeros(cropped_polygon.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(coordinates)], 1)
        mask = np.stack([mask] * 3, axis=-1)

        cropped_polygon[mask == 0] = settings.RECOGNITION_PAD_VALUE
    except cv2.error as e:
        logger.warning(f"Warning: issue while processing polygon: {e}")

    return cropped_polygon
