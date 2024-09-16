from transformers import TableTransformerForObjectDetection

from surya.settings import settings


def load_model(checkpoint=settings.TABLE_REC_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
    model = TableTransformerForObjectDetection.from_pretrained(checkpoint, torch_dtype=dtype)

    model = model.to(device)
    model = model.eval()

    print(f"Loaded table model {checkpoint} on device {device} with dtype {dtype}")

    return model

