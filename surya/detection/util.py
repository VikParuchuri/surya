import math
from PIL import ImageOps

from surya.settings import settings


def get_total_splits(image_size, height):
    img_height = list(image_size)[1]
    max_height = settings.DETECTOR_IMAGE_CHUNK_HEIGHT
    if img_height > max_height:
        num_splits = math.ceil(img_height / height)
        return num_splits
    return 1


def split_image(img, height):
    # This will not modify/return the original image - it will either crop, or copy the image
    img_height = list(img.size)[1]
    max_height = settings.DETECTOR_IMAGE_CHUNK_HEIGHT
    if img_height > max_height:
        num_splits = math.ceil(img_height / height)
        splits = []
        split_heights = []
        for i in range(num_splits):
            top = i * height
            bottom = (i + 1) * height
            if bottom > img_height:
                bottom = img_height
            cropped = img.crop((0, top, img.size[0], bottom))
            height = bottom - top
            if height < height:
                cropped = ImageOps.pad(cropped, (img.size[0], height), color=255, centering=(0, 0))
            splits.append(cropped)
            split_heights.append(height)
        return splits, split_heights
    return [img.copy()], [img_height]
