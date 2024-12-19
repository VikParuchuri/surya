import torch  # 导入 PyTorch
from typing import List  # 导入 List 类型提示
from PIL import Image  # 导入 PIL 库的 Image 模块

from surya.postprocessing.math.latex import fix_math, contains_math  # 导入数学后处理函数
from surya.postprocessing.text import truncate_repetitions  # 导入文本后处理函数
from surya.settings import settings  # 导入设置
from tqdm import tqdm  # 导入 tqdm 进度条库
import numpy as np  # 导入 NumPy
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API

def get_batch_size():  # 定义获取批处理大小的函数
    # 获取批处理大小
    batch_size = settings.RECOGNITION_BATCH_SIZE  # 从设置中获取批处理大小
    # 如果批处理大小未设置
    if batch_size is None:  
        batch_size = 32  # 默认批处理大小为 32
        if settings.TORCH_DEVICE_MODEL == "mps":  # 如果设备是 MPS
            batch_size = 64  # 设置批处理大小为 64
        if settings.TORCH_DEVICE_MODEL == "cuda":  # 如果设备是 CUDA
            batch_size = 256  # 设置批处理大小为 256
    return batch_size  # 返回批处理大小

def pad_to_batch_size(tensor, batch_size):  # 定义填充到批处理大小的函数
    # 填充到批处理大小
    current_batch_size = tensor.shape[0]  # 获取当前批处理大小
    if current_batch_size >= batch_size:  # 如果当前批处理大小大于等于目标批处理大小
        return tensor  # 返回原始张量

    pad_size = batch_size - current_batch_size  # 计算需要填充的大小
    padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)  # 创建填充元组

    return F.pad(tensor, padding, mode='constant', value=0)  # 使用常数填充张量

def batch_recognition(images: List[Image.Image], languages: List[List[str] | None], model, processor, batch_size=None):  # 定义批量识别函数
    # 批量识别
    assert all(isinstance(image, Image.Image) for image in images)  # 确保所有输入都是图像
    assert len(images) == len(languages)  # 确保图像和语言列表长度相同

    if len(images) == 0:  # 如果没有图像
        return [], []  # 返回空列表

    if batch_size is None:  # 如果批处理大小未设置
        batch_size = get_batch_size()  # 获取默认批处理大小

    # 按宽度排序图像
    sorted_pairs = sorted(enumerate(images), key=lambda x: x[1].width, reverse=False)  # 按宽度升序排序图像
    indices, images = zip(*sorted_pairs)  # 解压索引和图像
    indices = list(indices)  # 转换为列表
    images = list(images)  # 转换为列表

    output_text = []  # 初始化输出文本列表
    confidences = []  # 初始化置信度列表
    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):  # 遍历图像批次
        batch_images = images[i:i+batch_size]  # 获取当前批次图像
        batch_images = [image.convert("RGB") for image in batch_images]  # 将图像转换为 RGB
        real_batch_size = len(batch_images)  # 获取实际批处理大小
        batch_langs = languages[i:i+real_batch_size]  # 获取当前批次语言
        has_math = [lang and "_math" in lang for lang in batch_langs]  # 检查是否包含数学公式

        processed_batch = processor(text=[""] * len(batch_images), images=batch_images, langs=batch_langs)  # 处理批次图像

        batch_pixel_values = processed_batch["pixel_values"]  # 获取像素值
        batch_langs = processed_batch["langs"]  # 获取语言
        batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]  # 创建解码器输入
        max_input_length = max(len(tokens) for tokens in batch_decoder_input)  # 获取最大输入长度

        # 如果需要，填充解码器输入到最大长度
        for idx, tokens in enumerate(batch_decoder_input):  # 遍历解码器输入
            if len(tokens) < max_input_length:  # 如果输入长度小于最大长度
                padding_length = max_input_length - len(tokens)  # 计算填充长度
                batch_decoder_input[idx] = [processor.tokenizer.pad_id] * padding_length + tokens  # 填充输入
        current_batch_size = len(batch_pixel_values)  # 获取当前批处理大小

        batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=model.dtype, device=model.device)  # 转换为张量
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long, device=model.device)  # 转换为张量
        if settings.RECOGNITION_STATIC_CACHE:  # 如果启用静态缓存
            batch_pixel_values = pad_to_batch_size(batch_pixel_values, batch_size)  # 填充像素值
            batch_decoder_input = pad_to_batch_size(batch_decoder_input, batch_size)  # 填充解码器输入

        token_count = 0  # 初始化令牌计数
        inference_token_count = batch_decoder_input.shape[-1]  # 获取推理令牌计数
        batch_predictions = [[] for _ in range(current_batch_size)]  # 初始化批次预测

        decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64, device=model.device).cumsum(0) - 1  # 创建解码器位置 ID
        model.decoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)  # 设置解码器缓存
        model.text_encoder.model._setup_cache(model.config, batch_size, model.device, model.dtype)  # 设置文本编码器缓存

        sequence_scores = None  # 初始化序列分数
        all_done = torch.zeros(current_batch_size, dtype=torch.bool, device=model.device)  # 初始化完成标志
        encoder_hidden_states = None  # 初始化编码器隐藏状态

        with torch.inference_mode():  # 启用推理模式
            encoder_batch_size = batch_size // settings.RECOGNITION_ENCODER_BATCH_DIVISOR  # 计算编码器批处理大小
            for z in range(0, batch_pixel_values.shape[0], encoder_batch_size):  # 遍历编码器批次
                encoder_pixel_values = batch_pixel_values[z:min(z + encoder_batch_size, batch_pixel_values.shape[0])]  # 获取当前批次像素值
                encoder_hidden_states_batch = model.encoder(pixel_values=encoder_pixel_values).last_hidden_state  # 获取编码器隐藏状态
                if encoder_hidden_states is None:  # 如果编码器隐藏状态为空
                    encoder_hidden_states = encoder_hidden_states_batch  # 设置编码器隐藏状态
                else:
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_batch], dim=0)  # 连接编码器隐藏状态

            text_encoder_input_ids = torch.arange(
                model.text_encoder.config.query_token_count,
                device=encoder_hidden_states.device,
                dtype=torch.long
            ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)  # 创建文本编码器输入 ID

            encoder_text_hidden_states = model.text_encoder(
                input_ids=text_encoder_input_ids,
                cache_position=None,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                use_cache=False
            ).hidden_states  # 获取文本编码器隐藏状态
            del encoder_hidden_states  # 删除编码器隐藏状态

            if settings.RECOGNITION_STATIC_CACHE:  # 如果启用静态缓存
                # 填充输入到最大批处理大小以进行静态缓存
                encoder_text_hidden_states = pad_to_batch_size(encoder_text_hidden_states, batch_size)  # 填充文本编码器隐藏状态
                batch_decoder_input = pad_to_batch_size(batch_decoder_input, batch_size)  # 填充解码器输入

            while token_count < settings.RECOGNITION_MAX_TOKENS - 1:  # 当令牌计数小于最大令牌数时
                is_prefill = token_count == 0  # 检查是否为预填充
                #TODO: 添加注意力掩码
                return_dict = model.decoder(
                    input_ids=batch_decoder_input,
                    encoder_hidden_states=encoder_text_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True,
                    prefill=is_prefill
                )  # 调用解码器

                decoder_position_ids = decoder_position_ids[-1:] + 1  # 更新解码器位置 ID
                logits = return_dict["logits"][:current_batch_size]  # 忽略批处理填充
                aux_logits = return_dict.get("aux_logits", None)  # 获取辅助 logits

                preds = torch.argmax(logits[:, -1], dim=-1)  # 获取预测
                scores = torch.max(F.softmax(logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)  # 获取分数
                done = (preds == processor.tokenizer.eos_id) | (preds == processor.tokenizer.pad_id)  # 检查是否完成
                all_done = all_done | done  # 更新完成标志

                if is_prefill:  # 如果为预填充
                    sequence_scores = scores  # 设置序列分数
                else:
                    scores = scores.masked_fill(all_done, 0)  # 填充完成的分数
                    sequence_scores = torch.cat([sequence_scores, scores], dim=1)  # 连接序列分数

                if all_done.all():  # 如果所有都完成
                    break  # 退出循环

                batch_decoder_input = preds.unsqueeze(1)  # 更新解码器输入

                for j, (pred, status) in enumerate(zip(preds, all_done)):  # 遍历预测和完成标志
                    if not status:  # 如果未完成
                        batch_predictions[j].append(int(pred))  # 添加预测

                token_count += inference_token_count  # 更新令牌计数
                inference_token_count = batch_decoder_input.shape[-1]  # 获取推理令牌计数
                max_position_id = torch.max(decoder_position_ids).item()  # 获取最大位置 ID
                decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64, device=model.device).cumsum(0) - 1 + max_position_id  # 更新解码器位置 ID

                if settings.RECOGNITION_STATIC_CACHE:  # 如果启用静态缓存
                    batch_decoder_input = pad_to_batch_size(batch_decoder_input, batch_size)  # 填充解码器输入

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)  # 计算序列分数
        detected_text = processor.tokenizer.batch_decode(batch_predictions)  # 解码批次预测
        detected_text = [truncate_repetitions(dt) for dt in detected_text]  # 截断重复

        # 后处理以修复 LaTeX 输出
        detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]  # 修复数学公式

        # 将 sequence_scores 转换为当前批处理的列表
        batch_confidences = sequence_scores.tolist()  # 转换为列表

        # 如果实际批处理大小小于批处理大小，则排除填充结果
        if settings.RECOGNITION_STATIC_CACHE:  # 如果启用静态缓存
            detected_text = detected_text[:real_batch_size]  # 排除填充结果
            batch_confidences = batch_confidences[:real_batch_size]  # 排除填充结果

        output_text.extend(detected_text)  # 添加检测到的文本
        confidences.extend(batch_confidences)  # 添加置信度

        del encoder_text_hidden_states  # 删除文本编码器隐藏状态

    output_text = sorted(zip(indices, output_text), key=lambda x: x[0])  # 按索引排序输出文本
    confidences = sorted(zip(indices, confidences), key=lambda x: x[0])  # 按索引排序置信度
    output_text = [text for _, text in output_text]  # 获取输出文本
    confidences = [conf for _, conf in confidences]  # 获取置信度
    return output_text, confidences  # 返回输出文本和置信度


