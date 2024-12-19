from typing import List  # 导入 List 类型
# 导入必要的模块

import numpy as np  # 导入 numpy 库
import torch  # 导入 torch 库
from PIL import Image  # 从 PIL 库中导入 Image 模块

from tqdm import tqdm  # 从 tqdm 库中导入 tqdm 模块

from surya.input.slicing import ImageSlicer  # 从 surya.input.slicing 模块中导入 ImageSlicer 类
from surya.model.layout.config import ID_TO_LABEL  # 从 surya.model.layout.config 模块中导入 ID_TO_LABEL
from surya.postprocessing.heatmap import clean_boxes, intersects_other_boxes  # 从 surya.postprocessing.heatmap 模块中导入 clean_boxes 和 intersects_other_boxes 函数
from surya.schema import LayoutResult, LayoutBox  # 从 surya.schema 模块中导入 LayoutResult 和 LayoutBox 类
from surya.settings import settings  # 从 surya.settings 模块中导入 settings


def get_batch_size():
    # 获取批处理大小
    batch_size = settings.LAYOUT_BATCH_SIZE  # 从设置中获取批处理大小
    if batch_size is None:  # 如果批处理大小未设置
        batch_size = 4  # 默认批处理大小为 4
        if settings.TORCH_DEVICE_MODEL == "mps":  # 如果设备模型是 mps
            batch_size = 4  # 批处理大小为 4
        if settings.TORCH_DEVICE_MODEL == "cuda":  # 如果设备模型是 cuda
            batch_size = 32  # 批处理大小为 32
    return batch_size  # 返回批处理大小


def prediction_to_polygon(pred, img_size, bbox_scaler, skew_scaler, skew_min=.001):
    # 将预测转换为多边形
    w_scale = img_size[0] / bbox_scaler  # 计算宽度缩放比例
    h_scale = img_size[1] / bbox_scaler  # 计算高度缩放比例

    boxes = pred  # 获取预测框
    cx = boxes[0]  # 获取中心 x 坐标
    cy = boxes[1]  # 获取中心 y 坐标
    width = boxes[2]  # 获取宽度
    height = boxes[3]  # 获取高度
    x1 = cx - width / 2  # 计算左上角 x 坐标
    y1 = cy - height / 2  # 计算左上角 y 坐标
    x2 = cx + width / 2  # 计算右下角 x 坐标
    y2 = cy + height / 2  # 计算右下角 y 坐标
    skew_x = torch.floor((boxes[4] - skew_scaler) / 2)  # 计算 x 方向的倾斜
    skew_y = torch.floor((boxes[5] - skew_scaler) / 2)  # 计算 y 方向的倾斜

    # 确保我们不会得到稍微变形的框
    skew_x[torch.abs(skew_x) < skew_min] = 0  # 如果倾斜值小于最小值，则设为 0
    skew_y[torch.abs(skew_y) < skew_min] = 0  # 如果倾斜值小于最小值，则设为 0

    polygon = [x1 - skew_x, y1 - skew_y, x2 - skew_x, y1 + skew_y, x2 + skew_x, y2 + skew_y, x1 + skew_x, y2 - skew_y]  # 计算多边形的四个顶点
    poly = []  # 初始化多边形列表
    for i in range(4):  # 遍历四个顶点
        poly.append([
            polygon[2 * i].item() * w_scale,  # 计算顶点的 x 坐标
            polygon[2 * i + 1].item() * h_scale  # 计算顶点的 y 坐标
        ])
    return poly  # 返回多边形


def find_pause_items(preds):
    # 查找暂停项
    pause_sequence = []  # 初始化暂停序列
    for p in preds[::-1]:  # 逆序遍历预测结果
        if not p["paused"]:  # 如果预测结果未暂停
            return pause_sequence  # 返回暂停序列
        pause_sequence.insert(0, p)  # 将预测结果插入暂停序列的开头
    return pause_sequence  # 返回暂停序列


def batch_layout_detection(images: List, model, processor, batch_size=None) -> List[LayoutResult]:
    # 批量布局检测
    assert all([isinstance(image, Image.Image) for image in images])  # 确保所有图像都是 PIL Image 对象
    if batch_size is None:  # 如果批处理大小未设置
        batch_size = get_batch_size()  # 获取批处理大小

    slicer = ImageSlicer(settings.LAYOUT_SLICE_MIN, settings.LAYOUT_SLICE_SIZE)  # 创建图像切片器

    batches = []  # 初始化批次列表
    img_counts = [slicer.slice_count(image) for image in images]  # 获取每个图像的切片数量

    start_idx = 0  # 初始化起始索引
    end_idx = 1  # 初始化结束索引
    while end_idx < len(img_counts):  # 遍历图像切片数量
        if any([
            sum(img_counts[start_idx:end_idx]) >= batch_size,  # 如果当前批次的切片数量大于等于批处理大小
            sum(img_counts[start_idx:end_idx + 1]) > batch_size,  # 如果下一个批次的切片数量大于批处理大小
            ]):
            batches.append((start_idx, end_idx))  # 将当前批次添加到批次列表
            start_idx = end_idx  # 更新起始索引
        end_idx += 1  # 更新结束索引

    if start_idx < len(img_counts):  # 如果起始索引小于图像数量
        batches.append((start_idx, len(img_counts)))  # 将剩余的图像添加到批次列表

    results = []  # 初始化结果列表
    for (start_idx, end_idx) in tqdm(batches, desc="Recognizing layout"):  # 遍历批次列表
        batch_results = []  # 初始化批次结果列表
        batch_images = images[start_idx:end_idx]  # 获取当前批次的图像
        batch_images = [image.convert("RGB") for image in batch_images]  # 将图像转换为 RGB 模式
        batch_images, tile_positions = slicer.slice(batch_images)  # 切片图像
        current_batch_size = len(batch_images)  # 获取当前批次的大小

        orig_sizes = [image.size for image in batch_images]  # 获取原始图像大小
        model_inputs = processor(batch_images)  # 处理图像

        batch_pixel_values = model_inputs["pixel_values"]  # 获取像素值
        batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)  # 转换为张量并移动到设备

        pause_token = [model.config.decoder.pause_token_id] * 7  # 获取暂停标记
        start_token = [model.config.decoder.bos_token_id] * 7  # 获取起始标记
        batch_decoder_input = [
            [start_token] + [pause_token] * model.config.decoder.pause_token_count  # 初始化解码器输入
            for _ in range(current_batch_size)
        ]
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)  # 转换为张量并移动到设备
        inference_token_count = batch_decoder_input.shape[1]  # 获取推理标记数量

        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :, 0], dtype=torch.int64, device=model.device).cumsum(0) - 1  # 初始化解码器位置 ID
        model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)  # 设置解码器缓存

        batch_predictions = [[] for _ in range(current_batch_size)]  # 初始化批次预测结果

        with torch.inference_mode():  # 禁用梯度计算
            encoder_hidden_states = model.encoder(pixel_values=batch_pixel_values)[0]  # 获取编码器隐藏状态

            token_count = 0  # 初始化标记计数
            all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)  # 初始化完成标记

            while token_count < settings.LAYOUT_MAX_BOXES:  # 遍历标记
                is_prefill = token_count == 0  # 是否为预填充
                return_dict = model.decoder(
                    input_boxes=batch_decoder_input,  # 解码器输入
                    encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                    cache_position=decoder_position_ids,  # 缓存位置
                    use_cache=True,  # 使用缓存
                    prefill=is_prefill  # 是否为预填充
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1  # 更新解码器位置 ID
                box_logits = return_dict["bbox_logits"][:current_batch_size, -1, :].detach()  # 获取边界框 logits
                class_logits = return_dict["class_logits"][:current_batch_size, -1, :].detach()  # 获取类别 logits

                probs = torch.nn.functional.softmax(class_logits, dim=-1).detach().cpu()  # 计算类别概率
                entropy = torch.special.entr(probs).sum(dim=-1)  # 计算熵

                class_preds = class_logits.argmax(-1)  # 获取类别预测
                box_preds = box_logits * model.config.decoder.bbox_size  # 获取边界框预测

                done = (class_preds == model.decoder.config.eos_token_id) | (class_preds == model.decoder.config.pad_token_id)  # 判断是否完成

                all_done = all_done | done  # 更新完成标记
                if all_done.all():  # 如果所有都完成
                    break  # 退出循环

                batch_decoder_input = torch.cat([box_preds.unsqueeze(1), class_preds.unsqueeze(1).unsqueeze(1)], dim=-1)  # 更新解码器输入

                for j, (pred, status) in enumerate(zip(batch_decoder_input, all_done)):  # 遍历解码器输入和完成标记
                    if not status:  # 如果未完成
                        last_prediction = batch_predictions[j][-1] if len(batch_predictions[j]) > 0 else None  # 获取最后一个预测结果
                        preds = pred[0].detach().cpu()  # 获取预测结果
                        prediction = {
                            "preds": preds,  # 预测结果
                            "token": preds,  # 标记
                            "entropy": entropy[j].item(),  # 熵
                            "paused": False,  # 是否暂停
                            "pause_tokens": 0,  # 暂停标记数量
                            "polygon": prediction_to_polygon(
                                    preds,  # 预测结果
                                    orig_sizes[j],  # 原始大小
                                    model.config.decoder.bbox_size,  # 边界框大小
                                    model.config.decoder.skew_scaler  # 倾斜缩放器
                                ),
                            "label": preds[6].item() - model.decoder.config.special_token_count,  # 标签
                            "class_logits": class_logits[j].detach().cpu(),  # 类别 logits
                            "orig_size": orig_sizes[j]  # 原始大小
                        }
                        prediction["text_label"] = ID_TO_LABEL.get(int(prediction["label"]), None)  # 获取文本标签
                        if last_prediction and last_prediction["paused"]:  # 如果最后一个预测结果暂停
                            pause_sequence = find_pause_items(batch_predictions[j])  # 查找暂停项
                            entropies = [p["entropy"] for p in pause_sequence]  # 获取熵
                            min_entropy = min(entropies)  # 获取最小熵
                            max_pause_tokens = last_prediction["pause_tokens"]  # 获取最大暂停标记数量
                            if len(pause_sequence) < max_pause_tokens and prediction["entropy"] > min_entropy:  # 如果暂停序列长度小于最大暂停标记数量且熵大于最小熵
                                # 继续暂停
                                prediction["paused"] = True  # 设置暂停
                                prediction["pause_tokens"] = last_prediction["pause_tokens"]  # 设置暂停标记数量
                                prediction["token"].fill_(model.decoder.config.pause_token_id)  # 填充暂停标记
                                batch_decoder_input[j, :] = model.decoder.config.pause_token_id  # 更新解码器输入
                        elif intersects_other_boxes(prediction["polygon"], [p["polygon"] for p in batch_predictions[j]], thresh=.4):  # 如果预测结果与其他框相交
                            prediction["paused"] = True  # 设置暂停
                            prediction["pause_tokens"] = 1  # 设置暂停标记数量
                            prediction["token"].fill_(model.decoder.config.pause_token_id)  # 填充暂停标记
                            batch_decoder_input[j, :] = model.decoder.config.pause_token_id  # 更新解码器输入
                        elif all([
                                prediction["text_label"] in ["PageHeader", "PageFooter"],  # 如果文本标签是页眉或页脚
                                prediction["polygon"][0][1] < prediction["orig_size"][1] * .8,  # 如果多边形的 y 坐标小于原始大小的 80%
                                prediction["polygon"][2][1] > prediction["orig_size"][1] * .2,  # 如果多边形的 y 坐标大于原始大小的 20%
                                prediction["polygon"][0][0] < prediction["orig_size"][0] * .8,  # 如果多边形的 x 坐标小于原始大小的 80%
                                prediction["polygon"][2][0] > prediction["orig_size"][0] * .2  # 如果多边形的 x 坐标大于原始大小的 20%
                            ]):
                            # 确保页脚仅出现在页面底部，页眉仅出现在顶部
                            prediction["class_logits"][int(preds[6].item())] = 0  # 设置类别 logits 为 0
                            new_prediction = prediction["class_logits"].argmax(-1).item()  # 获取新的预测结果
                            prediction["label"] = new_prediction - model.decoder.config.special_token_count  # 更新标签
                            prediction["token"][6] = new_prediction  # 更新标记
                            batch_decoder_input[j, -1, 6] = new_prediction  # 更新解码器输入

                        batch_predictions[j].append(prediction)  # 将预测结果添加到批次预测结果

                token_count += inference_token_count  # 更新标记计数
                inference_token_count = batch_decoder_input.shape[1]  # 更新推理标记数量
                batch_decoder_input = batch_decoder_input.to(torch.long)  # 转换为长整型

        for j, (pred_dict, orig_size) in enumerate(zip(batch_predictions, orig_sizes)):  # 遍历批次预测结果和原始大小
            boxes = []  # 初始化边界框列表
            preds = [p for p in pred_dict if p["token"][6] > model.decoder.config.special_token_count]  # 移除特殊标记，如暂停
            if len(preds) > 0:  # 如果有预测结果
                polygons = [p["polygon"] for p in preds]  # 获取多边形
                labels = [p["label"] for p in preds]  # 获取标签

                for z, (poly, label) in enumerate(zip(polygons, labels)):  # 遍历多边形和标签
                    lb = LayoutBox(
                        polygon=poly,  # 多边形
                        label=ID_TO_LABEL[int(label)],  # 标签
                        position=z  # 位置
                    )
                    boxes.append(lb)  # 将边界框添加到列表
            boxes = clean_boxes(boxes)  # 清理边界框
            result = LayoutResult(
                bboxes=boxes,  # 边界框
                image_bbox=[0, 0, orig_size[0], orig_size[1]]  # 图像边界框
            )
            batch_results.append(result)  # 将结果添加到批次结果

        assert len(batch_results) == len(tile_positions)  # 确保批次结果和切片位置数量相同
        batch_results = slicer.join(batch_results, tile_positions)  # 合并批次结果
        results.extend(batch_results)  # 将批次结果添加到结果列表

    assert len(results) == len(images)  # 确保结果数量和图像数量相同
    return results  # 返回结果
