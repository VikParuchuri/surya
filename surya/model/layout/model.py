import torch

from surya.model.detection.model import EfficientViTForSemanticLayoutSegmentation
from surya.model.detection.config import EfficientViTConfig
from surya.model.detection.processor import SegformerImageProcessor
from surya.settings import settings

def load_model(checkpoint=settings.DETECTOR_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE, compile=False) -> EfficientViTForSemanticLayoutSegmentation:
    config = EfficientViTConfig.from_pretrained(checkpoint)
    model = EfficientViTForSemanticLayoutSegmentation.from_pretrained(checkpoint, torch_dtype=dtype, config=config, ignore_mismatched_sizes=True)
    model = model.to(device)
    model = model.eval()

    if compile:
        assert settings.DETECTOR_STATIC_CACHE, "You must set DETECTOR_STATIC_CACHE to compile the model."
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model)

    print(f"Loaded detection model {checkpoint} on device {device} with dtype {dtype}")
    return model


def load_processor(checkpoint=settings.DETECTOR_MODEL_CHECKPOINT):
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    return processor