import contextlib

import torch
from typing import List, Tuple, Generator

import numpy as np
from PIL import Image

from surya.model.detection.model import EfficientViTForSemanticSegmentation
from surya.postprocessing.heatmap import get_and_clean_boxes
from surya.postprocessing.affinity import get_vertical_lines
from surya.input.processing import prepare_image_detection, split_image, get_total_splits
from surya.schema import TextDetectionResult
from surya.settings import settings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

from surya.util.parallel import FakeExecutor

# 获取批处理大小


def get_batch_size():
    batch_size = settings.DETECTOR_BATCH_SIZE
    if batch_size is None:
        batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "mps":
            batch_size = 8
        if settings.TORCH_DEVICE_MODEL == "cuda":
            batch_size = 36
    return batch_size

# 将张量填充到批处理大小


def pad_to_batch_size(tensor, batch_size):
    current_batch_size = tensor.shape[0]
    if current_batch_size >= batch_size:
        return tensor

    pad_size = batch_size - current_batch_size
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)

    return F.pad(tensor, padding, mode='constant', value=0)

# 批量检测函数


def batch_detection(
    images: List,  # 输入图像列表
    model: EfficientViTForSemanticSegmentation,  # 检测模型
    processor,  # 处理器
    batch_size=None,  # 批处理大小
    static_cache=False  # 是否使用静态缓存
) -> Generator[Tuple[List[List[np.ndarray]], List[Tuple[int, int]]], None, None]:
    assert all([isinstance(image, Image.Image)
               for image in images])  # 确保所有输入都是图像
    if batch_size is None:
        batch_size = get_batch_size()  # 获取批处理大小
    heatmap_count = model.config.num_labels  # 获取热图数量

    orig_sizes = [image.size for image in images]  # 获取原始图像大小
    splits_per_image = [get_total_splits(
        size, processor) for size in orig_sizes]  # 获取每张图像的分割数量

    batches = []
    current_batch_size = 0
    current_batch = []
    for i in range(len(images)):
        if current_batch_size + splits_per_image[i] > batch_size:
            if len(current_batch) > 0:
                batches.append(current_batch)  # 添加当前批次
            current_batch = []
            current_batch_size = 0
        current_batch.append(i)
        current_batch_size += splits_per_image[i]

    if len(current_batch) > 0:
        batches.append(current_batch)  # 添加最后一个批次

    for batch_idx in tqdm(range(len(batches)), desc="Detecting bboxes"):
        batch_image_idxs = batches[batch_idx]
        batch_images = [images[j].convert("RGB")
                        for j in batch_image_idxs]  # 转换图像为RGB

        split_index = []
        split_heights = []
        image_splits = []
        for image_idx, image in enumerate(batch_images):
            image_parts, split_height = split_image(image, processor)  # 分割图像
            image_splits.extend(image_parts)
            split_index.extend([image_idx] * len(image_parts))
            split_heights.extend(split_height)

        image_splits = [prepare_image_detection(
            image, processor) for image in image_splits]  # 准备图像检测
        batch = torch.stack(image_splits, dim=0).to(
            model.dtype).to(model.device)  # 将图像堆叠成批次
        if static_cache:
            batch = pad_to_batch_size(batch, batch_size)  # 填充批次大小

        with torch.inference_mode():
            pred = model(pixel_values=batch)  # 进行推理

        logits = pred.logits
        correct_shape = [processor.size["height"], processor.size["width"]]
        current_shape = list(logits.shape[2:])
        if current_shape != correct_shape:
            logits = F.interpolate(
                logits, size=correct_shape, mode='bilinear', align_corners=False)  # 调整大小

        logits = logits.cpu().detach().numpy().astype(np.float32)
        preds = []
        for i, (idx, height) in enumerate(zip(split_index, split_heights)):
            if len(preds) <= idx:
                preds.append([logits[i][k]
                             for k in range(heatmap_count)])  # 添加新的预测
            else:
                heatmaps = preds[idx]
                pred_heatmaps = [logits[i][k] for k in range(heatmap_count)]

                if height < processor.size["height"]:
                    pred_heatmaps = [pred_heatmap[:height, :]
                                     for pred_heatmap in pred_heatmaps]  # 截断填充

                for k in range(heatmap_count):
                    heatmaps[k] = np.vstack(
                        [heatmaps[k], pred_heatmaps[k]])  # 叠加热图
                preds[idx] = heatmaps

        yield preds, [orig_sizes[j] for j in batch_image_idxs]  # 生成预测结果和原始大小

# 并行获取文本行


def parallel_get_lines(preds, orig_sizes, include_maps=False):
    heatmap, affinity_map = preds
    heat_img, aff_img = None, None
    if include_maps:
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))  # 生成热图图像
        aff_img = Image.fromarray(
            (affinity_map * 255).astype(np.uint8))  # 生成亲和图图像
    affinity_size = list(reversed(affinity_map.shape))
    heatmap_size = list(reversed(heatmap.shape))
    bboxes = get_and_clean_boxes(heatmap, heatmap_size, orig_sizes)  # 获取并清理边界框
    vertical_lines = get_vertical_lines(
        affinity_map, affinity_size, orig_sizes)  # 获取垂直线

    result = TextDetectionResult(
        bboxes=bboxes,
        vertical_lines=vertical_lines,
        heatmap=heat_img,
        affinity_map=aff_img,
        image_bbox=[0, 0, orig_sizes[0], orig_sizes[1]]
    )
    return result

# 批量文本检测 接受 图像列表、模型、处理器、批处理大小、是否包含地图 返回 文本检测结果列表


def batch_text_detection(images: List, model, processor, batch_size=None, include_maps=False) -> List[TextDetectionResult]:

    # 处理图像 批量处理图像并进行文本检测。 是一个生成器，用于逐批次生成检测结果
    detection_generator = batch_detection(
        images, model, processor, batch_size=batch_size, static_cache=settings.DETECTOR_STATIC_CACHE)

    # 存储后处理任务的未来对象。
    postprocessing_futures = []
    max_workers = min(
        settings.DETECTOR_POSTPROCESSING_CPU_WORKERS, len(images))

    # 如果不在 Streamlit 环境中，并且图像数量大于等于并行处理的最小阈值，则进行并行处理。
    parallelize = not settings.IN_STREAMLIT and len(
        images) >= settings.DETECTOR_MIN_PARALLEL_THRESH

    '''
        根据 parallelize 的值选择执行器：
        如果进行并行处理，使用 ThreadPoolExecutor。
        否则，使用 FakeExecutor（模拟并行处理）。
    '''
    executor = ThreadPoolExecutor if parallelize else FakeExecutor
    with executor(max_workers=max_workers) as e:
        for preds, orig_sizes in detection_generator:
            for pred, orig_size in zip(preds, orig_sizes):
                postprocessing_futures.append(
                    e.submit(parallel_get_lines, pred, orig_size, include_maps))

    '''
        等待所有后处理任务完成，并收集结果。
        返回包含所有文本检测结果的列表。
    '''
    return [future.result() for future in postprocessing_futures]
