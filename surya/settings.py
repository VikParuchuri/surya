from typing import Dict, Optional  # 导入类型提示模块

from dotenv import find_dotenv  # 导入 dotenv 模块以查找环境文件
from pydantic import computed_field  # 导入 pydantic 的 computed_field 装饰器
# 导入 pydantic_settings 的 BaseSettings 类
# pydantic_settings 提供了一个方便的方式来定义配置类,并支持从各种源（如环境变量、JSON 文件、YAML 文件）加载配置数据。
from pydantic_settings import BaseSettings
import torch  # 导入 PyTorch
import os  # 导入 os 模块以进行文件和目录操作

# 定义 Settings 类,继承自 BaseSettings


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None  # PyTorch 设备,可选
    '''
    DPI 的全称是 Dots Per Inch(每英寸点数),是一个用于衡量分辨率的指标,主要用于描述图像、打印输出或显示设备的精细程度。
    通常情况下,DPI 越高,图像的精细程度就越高。
    '''
    IMAGE_DPI: int = 96  # 用于检测、布局、阅读顺序的图像 DPI
    IMAGE_DPI_HIGHRES: int = 192  # 用于 OCR 和表格识别的高分辨率图像 DPI
    IN_STREAMLIT: bool = False  # 是否在 Streamlit 中运行
    ENABLE_EFFICIENT_ATTENTION: bool = True  # 是否启用高效注意力机制
    ENABLE_CUDNN_ATTENTION: bool = False  # 是否启用 cuDNN 注意力机制
    FLATTEN_PDF: bool = True  # 是否在处理前合并 PDF 表单字段

    # Paths
    DATA_DIR: str = "data"  # 数据目录
    RESULT_DIR: str = "results"  # 结果目录
    BASE_DIR: str = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))  # 基础目录
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")  # 字体目录

    @computed_field  # 使用 computed_field 装饰器
    def TORCH_DEVICE_MODEL(self) -> str:  # 定义 TORCH_DEVICE_MODEL 方法,返回字符串
        # 获取 PyTorch 设备模型
        if self.TORCH_DEVICE is not None:  # 如果 TORCH_DEVICE 不为空
            return self.TORCH_DEVICE  # 返回 TORCH_DEVICE

        if torch.cuda.is_available():  # 如果 CUDA 可用
            return "cuda"  # 返回 "cuda"

        if torch.backends.mps.is_available():  # 如果 MPS 可用 MPS是苹果的GPU加速技术
            return "mps"  # 返回 "mps"

        return "cpu"  # 否则返回 "cpu" 使用 CPU进行计算

    # 文本检测
    DETECTOR_BATCH_SIZE: Optional[int] = None  # 检测器批处理大小,可选
    DETECTOR_MODEL_CHECKPOINT: str = "vikp/surya_det3"  # 检测器模型检查点
    DETECTOR_BENCH_DATASET_NAME: str = "vikp/doclaynet_bench"  # 检测器基准数据集名称
    DETECTOR_IMAGE_CHUNK_HEIGHT: int = 1400  # 垂直切片图像的高度
    DETECTOR_TEXT_THRESHOLD: float = 0.6  # 文本检测阈值
    DETECTOR_BLANK_THRESHOLD: float = 0.35  # 空白检测阈值
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = min(
        8, os.cpu_count())  # 后处理的 CPU 工作线程数
    DETECTOR_MIN_PARALLEL_THRESH: int = 3  # 并行处理的最小图像数量
    COMPILE_DETECTOR: bool = False  # 是否编译检测器

    # 文本识别
    RECOGNITION_MODEL_CHECKPOINT: str = "vikp/surya_rec2"  # 识别模型检查点
    RECOGNITION_MAX_TOKENS: int = 175  # 最大令牌数
    RECOGNITION_BATCH_SIZE: Optional[int] = None  # 识别批处理大小,可选
    RECOGNITION_IMAGE_SIZE: Dict = {"height": 256, "width": 896}  # 识别图像大小
    RECOGNITION_RENDER_FONTS: Dict[str, str] = {
        # 所有语言的字体路径
        "all": os.path.join(FONT_DIR, "GoNotoCurrent-Regular.ttf"),
        "zh": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),  # 中文字体路径
        "ja": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),  # 日文字体路径
        "ko": os.path.join(FONT_DIR, "GoNotoCJKCore.ttf"),  # 韩文字体路径
    }  # 识别渲染字体
    RECOGNITION_FONT_DL_BASE: str = "https://github.com/satbyy/go-noto-universal/releases/download/v7.0"  # 字体下载基础 URL
    RECOGNITION_BENCH_DATASET_NAME: str = "vikp/rec_bench"  # 识别基准数据集名称
    RECOGNITION_PAD_VALUE: int = 255  # 填充值
    COMPILE_RECOGNITION: bool = False  # 是否编译识别器
    RECOGNITION_ENCODER_BATCH_DIVISOR: int = 1  # 编码器批处理大小除数

    # Layout
    LAYOUT_MODEL_CHECKPOINT: str = "datalab-to/surya_layout0"  # 布局模型检查点
    LAYOUT_IMAGE_SIZE: Dict = {"height": 768, "width": 768}  # 布局图像大小
    LAYOUT_SLICE_MIN: Dict = {"height": 1500, "width": 1500}  # 开始切片图像的最小尺寸
    LAYOUT_SLICE_SIZE: Dict = {"height": 1200, "width": 1200}  # 切片大小
    LAYOUT_BATCH_SIZE: Optional[int] = None  # 布局批处理大小,可选
    LAYOUT_BENCH_DATASET_NAME: str = "vikp/publaynet_bench"  # 布局基准数据集名称
    LAYOUT_MAX_BOXES: int = 100  # 最大框数
    COMPILE_LAYOUT: bool = False  # 是否编译布局

    # Table Rec
    TABLE_REC_MODEL_CHECKPOINT: str = "vikp/surya_tablerec"  # 表格识别模型检查点
    TABLE_REC_IMAGE_SIZE: Dict = {"height": 640, "width": 640}  # 表格识别图像大小
    TABLE_REC_MAX_BOXES: int = 512  # 最大框数
    TABLE_REC_MAX_ROWS: int = 384  # 最大行数
    TABLE_REC_BATCH_SIZE: Optional[int] = None  # 表格识别批处理大小,可选
    TABLE_REC_BENCH_DATASET_NAME: str = "vikp/fintabnet_bench"  # 表格识别基准数据集名称
    COMPILE_TABLE_REC: bool = False  # 是否编译表格识别

    # Tesseract (for benchmarks only)
    TESSDATA_PREFIX: Optional[str] = None  # Tesseract 数据前缀

    COMPILE_ALL: bool = False  # 是否编译所有模型

    @computed_field  # 使用 computed_field 装饰器
    # 定义 DETECTOR_STATIC_CACHE 方法,返回布尔值
    def DETECTOR_STATIC_CACHE(self) -> bool:
        # 检测器静态缓存
        # 如果 COMPILE_ALL 或 COMPILE_DETECTOR 为真,则返回真
        return self.COMPILE_ALL or self.COMPILE_DETECTOR

    @computed_field  # 使用 computed_field 装饰器
    # 定义 RECOGNITION_STATIC_CACHE 方法,返回布尔值
    def RECOGNITION_STATIC_CACHE(self) -> bool:
        # 识别器静态缓存
        # 如果 COMPILE_ALL 或 COMPILE_RECOGNITION 为真,则返回真
        return self.COMPILE_ALL or self.COMPILE_RECOGNITION

    @computed_field  # 使用 computed_field 装饰器
    def LAYOUT_STATIC_CACHE(self) -> bool:  # 定义 LAYOUT_STATIC_CACHE 方法,返回布尔值
        # 布局静态缓存
        # 如果 COMPILE_ALL 或 COMPILE_LAYOUT 为真,则返回真
        return self.COMPILE_ALL or self.COMPILE_LAYOUT

    @computed_field  # 使用 computed_field 装饰器
    # 定义 TABLE_REC_STATIC_CACHE 方法,返回布尔值
    def TABLE_REC_STATIC_CACHE(self) -> bool:
        # 表格识别静态缓存
        # 如果 COMPILE_ALL 或 COMPILE_TABLE_REC 为真,则返回真
        return self.COMPILE_ALL or self.COMPILE_TABLE_REC

    @computed_field  # 使用 computed_field 装饰器
    @property  # 使用 property 装饰器
    def MODEL_DTYPE(self) -> torch.dtype:  # 定义 MODEL_DTYPE 方法,返回 torch.dtype
        # 模型数据类型
        # 如果 TORCH_DEVICE_MODEL 为 "cpu",则返回 torch.float32,否则返回 torch.float16
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:  # 定义 Config 类
        env_file = find_dotenv("local.env")  # 环境文件
        extra = "ignore"  # 忽略额外字段


settings = Settings()  # 实例化设置
