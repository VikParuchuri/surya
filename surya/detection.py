import os
from typing import List, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.postprocessing.affinity import get_vertical_lines, get_horizontal_lines
from surya.input.processing import prepare_image, split_image
from surya.schema import TextDetectionResult
from surya.settings import settings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def get_batch_size():
    batch_size = settings.DETECTOR_BATCH_SIZE
    if batch_size is None:
        batch_size = 6
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 24
    return batch_size


def batch_detection(images: List, model, processor, batch_size=None) -> Tuple[List[List[np.ndarray]], List[Tuple[int, int]]]:
    assert all([isinstance(image, Image.Image) for image in images])
    if batch_size is None:
        batch_size = get_batch_size()
    heatmap_count = model.config.num_labels

    images = [image.convert("RGB") for image in images]
    orig_sizes = [image.size for image in images]
    split_index = []
    split_heights = []
    image_splits = []
    for i, image in enumerate(images):
        image_parts, split_height = split_image(image, processor)
        image_splits.extend(image_parts)
        split_index.extend([i] * len(image_parts))
        split_heights.extend(split_height)

    image_splits = [prepare_image(image, processor) for image in image_splits]

    pred_parts = []
    for i in tqdm(range(0, len(image_splits), batch_size), desc="Detecting bboxes"):
        batch = image_splits[i:i+batch_size]
        # Batch images in dim 0
        batch = torch.stack(batch, dim=0)
        batch = batch.to(model.dtype)
        batch = batch.to(model.device)

        with torch.inference_mode():
            pred = model(pixel_values=batch)

        logits = pred.logits
        for j in range(logits.shape[0]):
            heatmaps = []
            for k in range(heatmap_count):
                heatmap = logits[j, k, :, :].detach().cpu().numpy().astype(np.float32)
                heatmap_shape = list(heatmap.shape)

                correct_shape = [processor.size["height"], processor.size["width"]]
                cv2_size = list(reversed(correct_shape)) # opencv uses (width, height) instead of (height, width)
                if heatmap_shape != correct_shape:
                    heatmap = cv2.resize(heatmap, cv2_size, interpolation=cv2.INTER_LINEAR)

                heatmaps.append(heatmap)
            pred_parts.append(heatmaps)

    preds = []
    for i, (idx, height) in enumerate(zip(split_index, split_heights)):
        if len(preds) <= idx:
            preds.append(pred_parts[i])
        else:
            heatmaps = preds[idx]
            pred_heatmaps = [pred_parts[i][k] for k in range(heatmap_count)]

            if height < processor.size["height"]:
                # Cut off padding to get original height
                pred_heatmaps = [pred_heatmap[:height, :] for pred_heatmap in pred_heatmaps]

            for k in range(heatmap_count):
                heatmaps[k] = np.vstack([heatmaps[k], pred_heatmaps[k]])
            preds[idx] = heatmaps

    assert len(preds) == len(images)
    assert all([len(pred) == heatmap_count for pred in preds])
    return preds, orig_sizes


def parallel_get_lines(preds, orig_sizes):
    heatmap, affinity_map = preds
    heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    aff_img = Image.fromarray((affinity_map * 255).astype(np.uint8))
    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))
    bboxes = get_and_clean_boxes(heatmap, heatmap_size, orig_sizes)
    vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes)
    horizontal_lines = get_horizontal_lines(affinity_map, affinity_size, orig_sizes)

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        horizontal_lines=horizontal_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]]
    )
    return result


def batch_text_detection(images: List, model, processor, batch_size=None) -> List[TextDetectionResult]:
    preds, orig_sizes = batch_detection(images, model, processor, batch_size=batch_size)
    results = []
    if len(images) == 1: # Ensures we don't parallelize with streamlit
        for i in range(len(images)):
            result = parallel_get_lines(preds[i], orig_sizes[i])
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=settings.DETECTOR_POSTPROCESSING_CPU_WORKERS) as executor:
            results = list(executor.map(parallel_get_lines, preds, orig_sizes))

    return results


