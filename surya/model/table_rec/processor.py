import cv2
import torch
import torch.nn.functional as F
from surya.settings import settings
import numpy as np


def load_processor():
    def _processor(imgs):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        imgs = [np.array(img) for img in imgs]
        imgs = [_normalize(_resize(img), mean_vals, std_vals) for img in imgs]
        pixel_values = torch.stack(imgs, dim=0)
        return pixel_values
    return _processor


def _resize(image, interpolation=cv2.INTER_LANCZOS4):
    max_height, max_width = settings.TABLE_REC_IMAGE_SIZE["height"], settings.TABLE_REC_IMAGE_SIZE["width"]
    resized_image = cv2.resize(image, (max_width, max_height), interpolation=interpolation)
    resized_image = resized_image.transpose(2, 0, 1).astype(np.float32)
    resized_image = torch.from_numpy(resized_image)
    resized_image /= 255.0
    return resized_image


def _normalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = (tensor - mean) / std
    return tensor