from typing import List, Tuple

import torch
import numpy as np
from PIL import Image

from surya.model.detection.segformer import SegformerForRegressionMask
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.postprocessing.affinity import get_vertical_lines
from surya.input.processing import prepare_image_detection, split_image, get_total_splits
from surya.schema import TextDetectionResult
from surya.settings import settings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F


def get_batch_size():
    batch_size = settings.DETECTOR_BATCH_SIZE
    if batch_size is None:
        batch_size = 6
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 24
    return batch_size


def batch_detection(images: List, model: SegformerForRegressionMask, processor, batch_size=None) -> Tuple[List[List[np.ndarray]], List[Tuple[int, int]]]:
    assert all([isinstance(image, Image.Image) for image in images])
    if batch_size is None:
        batch_size = get_batch_size()
    heatmap_count = model.config.num_labels

    orig_sizes = [image.size for image in images]
    splits_per_image = [get_total_splits(size, processor) for size in orig_sizes]

    batches = []
    current_batch_size = 0
    current_batch = []
    for i in range(len(images)):
        if current_batch_size + splits_per_image[i] > batch_size:
            if len(current_batch) > 0:
                batches.append(current_batch)
            current_batch = []
            current_batch_size = 0
        current_batch.append(i)
        current_batch_size += splits_per_image[i]

    if len(current_batch) > 0:
        batches.append(current_batch)

    all_preds = []
    for batch_idx in tqdm(range(len(batches)), desc="Detecting bboxes"):
        batch_image_idxs = batches[batch_idx]
        batch_images = [images[j].convert("RGB") for j in batch_image_idxs]

        split_index = []
        split_heights = []
        image_splits = []
        for image_idx, image in enumerate(batch_images):
            image_parts, split_height = split_image(image, processor)
            image_splits.extend(image_parts)
            split_index.extend([image_idx] * len(image_parts))
            split_heights.extend(split_height)

        image_splits = [prepare_image_detection(image, processor) for image in image_splits]
        # Batch images in dim 0
        batch = torch.stack(image_splits, dim=0).to(model.dtype).to(model.device)

        with torch.inference_mode():
            pred = model(pixel_values=batch)

        logits = pred.logits
        correct_shape = [processor.size["height"], processor.size["width"]]
        current_shape = list(logits.shape[2:])
        if current_shape != correct_shape:
            logits = F.interpolate(logits, size=correct_shape, mode='bilinear', align_corners=False)

        logits = logits.cpu().detach().numpy().astype(np.float32)
        preds = []
        for i, (idx, height) in enumerate(zip(split_index, split_heights)):
            # If our current prediction length is below the image idx, that means we have a new image
            # Otherwise, we need to add to the current image
            if len(preds) <= idx:
                preds.append([logits[i][k] for k in range(heatmap_count)])
            else:
                heatmaps = preds[idx]
                pred_heatmaps = [logits[i][k] for k in range(heatmap_count)]

                if height < processor.size["height"]:
                    # Cut off padding to get original height
                    pred_heatmaps = [pred_heatmap[:height, :] for pred_heatmap in pred_heatmaps]

                for k in range(heatmap_count):
                    heatmaps[k] = np.vstack([heatmaps[k], pred_heatmaps[k]])
                preds[idx] = heatmaps

        all_preds.extend(preds)

    assert len(all_preds) == len(images)
    assert all([len(pred) == heatmap_count for pred in all_preds])
    return all_preds, orig_sizes


def parallel_get_lines(preds, orig_sizes):
    heatmap, affinity_map = preds
    heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    aff_img = Image.fromarray((affinity_map * 255).astype(np.uint8))
    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))
    bboxes = get_and_clean_boxes(heatmap, heatmap_size, orig_sizes)
    vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes)

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]]
    )
    return result


def batch_text_detection(images: List, model, processor, batch_size=None) -> List[TextDetectionResult]:
    preds, orig_sizes = batch_detection(images, model, processor, batch_size=batch_size)
    results = []
    if settings.IN_STREAMLIT or len(images) < settings.DETECTOR_MIN_PARALLEL_THRESH: # Ensures we don't parallelize with streamlit, or with very few images
        for i in range(len(images)):
            result = parallel_get_lines(preds[i], orig_sizes[i])
            results.append(result)
    else:
        max_workers = min(settings.DETECTOR_POSTPROCESSING_CPU_WORKERS, len(images))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(parallel_get_lines, preds, orig_sizes))

    return results


