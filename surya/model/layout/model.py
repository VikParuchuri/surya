import torch

from surya.model.detection.model import EfficientViTForSemanticLayoutSegmentation
from surya.model.detection.config import EfficientViTConfig
from surya.model.detection.processor import SegformerImageProcessor
from surya.settings import settings

def load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE) -> EfficientViTForSemanticLayoutSegmentation:
    config = EfficientViTConfig.from_pretrained(checkpoint)
    model = EfficientViTForSemanticLayoutSegmentation.from_pretrained(checkpoint, torch_dtype=dtype, config=config, ignore_mismatched_sizes=True)
    model = model.to(device)
    model = model.eval()

    if settings.LAYOUT_STATIC_CACHE:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 1
        torch._dynamo.config.suppress_errors = False


        print(f"Compiling layout model {checkpoint} on device {device} with dtype {dtype}")
        model = torch.compile(model)

    print(f"Loaded layout model {checkpoint} on device {device} with dtype {dtype}")
    return model


def load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT):
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    return processor
