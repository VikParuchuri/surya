from concurrent.futures import ThreadPoolExecutor
from typing import List, Generator, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from surya.common.predictor import BasePredictor

from surya.detection.loader import DetectionModelLoader
from surya.detection.parallel import FakeExecutor
from surya.detection.util import get_total_splits, split_image
from surya.detection.schema import TextDetectionResult
from surya.settings import settings
from surya.detection.heatmap import parallel_get_boxes


class DetectionPredictor(BasePredictor):
    """Text detection, served from one shared instance.

    Public construction is client-backed: the model runs in a single shared server
    process (surya.detection.server) and this object POSTs images to it, so N worker
    processes don't each load their own copy. Use ``DetectionPredictor.local()`` for
    a process-local predictor that owns the model (that's what the server uses). The
    __call__ signature and TextDetectionResult output are unchanged.
    """

    model_loader_cls = DetectionModelLoader
    batch_size = settings.DETECTOR_BATCH_SIZE
    default_batch_sizes = {"cpu": 8, "mps": 8, "cuda": 36}

    def __init__(
        self, checkpoint=None, device=None, dtype=None, attention_implementation=None
    ):
        # Client-backed by default: no model load here. device/dtype/attn are
        # accepted for signature parity with BasePredictor but live server-side.
        from surya.detection.client import DetectionServerClient

        self._client = DetectionServerClient(checkpoint=checkpoint)
        self.model = None
        self.processor = None
        self._disable_tqdm = settings.DISABLE_TQDM

    @classmethod
    def local(cls, *args, **kwargs) -> "DetectionPredictor":
        """Process-local predictor that owns the model (used by the server)."""
        self = cls.__new__(cls)
        BasePredictor.__init__(self, *args, **kwargs)
        self._client = None
        return self

    def to(self, device_dtype=None):
        if self._client is not None:  # client-backed: device lives server-side
            return
        return super().to(device_dtype)

    def __call__(
        self, images: List[Image.Image], batch_size=None, include_maps=False
    ) -> List[TextDetectionResult]:
        if self._client is not None:
            return self._client(images, include_maps=include_maps)

        detection_generator = self.batch_detection(images, batch_size=batch_size)

        postprocessing_futures = []
        max_workers = min(settings.DETECTOR_POSTPROCESSING_CPU_WORKERS, len(images))
        parallelize = (
            not settings.IN_STREAMLIT
            and len(images) >= settings.DETECTOR_MIN_PARALLEL_THRESH
        )
        executor = ThreadPoolExecutor if parallelize else FakeExecutor
        with executor(max_workers=max_workers) as e:
            for preds, orig_sizes in detection_generator:
                for pred, orig_size in zip(preds, orig_sizes):
                    postprocessing_futures.append(
                        e.submit(parallel_get_boxes, pred, orig_size, include_maps)
                    )

        return [future.result() for future in postprocessing_futures]

    def prepare_image(self, img):
        new_size = (self.processor.size["width"], self.processor.size["height"])

        # This double resize is actually necessary for downstream accuracy
        img.thumbnail(new_size, Image.Resampling.LANCZOS)
        img = img.resize(
            new_size, Image.Resampling.LANCZOS
        )  # Stretch smaller dimension to fit new size

        img = np.asarray(img, dtype=np.uint8)
        img = self.processor(img)["pixel_values"][0]
        img = torch.from_numpy(img)
        return img

    def batch_detection(
        self, images: List, batch_size=None
    ) -> Generator[Tuple[List[List[np.ndarray]], List[Tuple[int, int]]], None, None]:
        assert all([isinstance(image, Image.Image) for image in images])
        if batch_size is None:
            batch_size = self.get_batch_size()
        heatmap_count = self.model.config.num_labels

        orig_sizes = [image.size for image in images]
        splits_per_image = [
            get_total_splits(size, self.processor.size["height"]) for size in orig_sizes
        ]

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

        for batch_idx in tqdm(
            range(len(batches)), desc="Detecting bboxes", disable=self.disable_tqdm
        ):
            batch_image_idxs = batches[batch_idx]
            batch_images = [images[j].convert("RGB") for j in batch_image_idxs]

            split_index = []
            split_heights = []
            image_splits = []
            for image_idx, image in enumerate(batch_images):
                image_parts, split_height = split_image(
                    image, self.processor.size["height"]
                )
                image_splits.extend(image_parts)
                split_index.extend([image_idx] * len(image_parts))
                split_heights.extend(split_height)

            image_splits = [self.prepare_image(image) for image in image_splits]
            # Batch images in dim 0
            batch = torch.stack(image_splits, dim=0).to(self.model.dtype)

            with settings.INFERENCE_MODE():
                pred = self.model(pixel_values=batch.to(self.model.device))

            logits = pred.logits
            correct_shape = [
                self.processor.size["height"],
                self.processor.size["width"],
            ]
            current_shape = list(logits.shape[2:])
            if current_shape != correct_shape:
                logits = F.interpolate(
                    logits, size=correct_shape, mode="bilinear", align_corners=False
                )

            logits = logits.to(torch.float32).cpu().numpy()
            preds = []
            for i, (idx, height) in enumerate(zip(split_index, split_heights)):
                # If our current prediction length is below the image idx, that means we have a new image
                # Otherwise, we need to add to the current image
                if len(preds) <= idx:
                    preds.append([logits[i][k] for k in range(heatmap_count)])
                else:
                    heatmaps = preds[idx]
                    pred_heatmaps = [logits[i][k] for k in range(heatmap_count)]

                    if height < self.processor.size["height"]:
                        # Cut off padding to get original height
                        pred_heatmaps = [
                            pred_heatmap[:height, :] for pred_heatmap in pred_heatmaps
                        ]

                    for k in range(heatmap_count):
                        heatmaps[k] = np.vstack([heatmaps[k], pred_heatmaps[k]])
                    preds[idx] = heatmaps

            yield preds, [orig_sizes[j] for j in batch_image_idxs]

        torch.cuda.empty_cache()
