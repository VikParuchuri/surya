import cv2
import numpy as np
import torch
from PIL import Image

from surya.model.processing import prepare_image, split_image
from surya.postprocessing.affinity import get_horizontal_lines, get_vertical_lines
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.settings import settings


def batch_inference(images: list, model, processor):
    assert all([isinstance(image, Image.Image) for image in images])

    images = [image.copy().convert("RGB") for image in images]
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
    for i in range(0, len(image_splits), settings.DETECTOR_BATCH_SIZE):
        batch = image_splits[i : i + settings.DETECTOR_BATCH_SIZE]
        # Batch images in dim 0
        batch = torch.stack(batch, dim=0)
        batch = batch.to(model.dtype)
        batch = batch.to(model.device)

        with torch.inference_mode():
            pred = model(pixel_values=batch)

        logits = pred.logits
        for j in range(logits.shape[0]):
            heatmap = logits[j, 0, :, :].detach().cpu().numpy().astype(np.float32)
            affinity_map = logits[j, 1, :, :].detach().cpu().numpy().astype(np.float32)

            heatmap_shape = list(heatmap.shape)
            correct_shape = [processor.size["height"], processor.size["width"]]
            cv2_size = list(reversed(correct_shape))  # opencv uses (width, height) instead of (height, width)

            if heatmap_shape != correct_shape:
                heatmap = cv2.resize(heatmap, cv2_size, interpolation=cv2.INTER_LINEAR)

            affinity_shape = list(affinity_map.shape)
            if affinity_shape != correct_shape:
                affinity_map = cv2.resize(affinity_map, cv2_size, interpolation=cv2.INTER_LINEAR)

            pred_parts.append((heatmap, affinity_map))

    preds = []
    for i, (idx, height) in enumerate(zip(split_index, split_heights)):
        if len(preds) <= idx:
            preds.append(pred_parts[i])
        else:
            heatmap, affinity_map = preds[idx]
            pred_heatmap = pred_parts[i][0]
            pred_affinity = pred_parts[i][1]

            if height < processor.size["height"]:
                # Cut off padding to get original height
                pred_heatmap = pred_heatmap[:height, :]
                pred_affinity = pred_affinity[:height, :]

            heatmap = np.vstack([heatmap, pred_heatmap])
            affinity_map = np.vstack([affinity_map, pred_affinity])
            preds[idx] = (heatmap, affinity_map)

    assert len(preds) == len(images)
    results = []
    for i in range(len(images)):
        heatmap, affinity_map = preds[i]
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        aff_img = Image.fromarray((affinity_map * 255).astype(np.uint8))

        affinity_size = list(reversed(affinity_map.shape))
        heatmap_size = list(reversed(heatmap.shape))
        bboxes = get_and_clean_boxes(heatmap, heatmap_size, orig_sizes[i])
        vertical_lines = get_vertical_lines(affinity_map, affinity_size, orig_sizes[i])
        horizontal_lines = get_horizontal_lines(affinity_map, affinity_size, orig_sizes[i])

        results.append(
            {
                "bboxes": bboxes,
                "vertical_lines": vertical_lines,
                "horizontal_lines": horizontal_lines,
                "heatmap": heat_img,
                "affinity_map": aff_img,
            }
        )

    return results
