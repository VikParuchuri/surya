import torch
from transformers import DetrFeatureExtractor, AutoModelForObjectDetection
from surya.settings import settings

from PIL import Image
import numpy as np


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def to_tensor(image):
    # Convert PIL Image to NumPy array
    np_image = np.array(image).astype(np.float32)

    # Rearrange dimensions to [C, H, W] format
    np_image = np_image.transpose((2, 0, 1))

    # Normalize to [0.0, 1.0]
    np_image /= 255.0

    return torch.from_numpy(np_image)


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def structure_transform(image):
    image = MaxResize(1000)(image)
    tensor = to_tensor(image)
    normalized_tensor = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return normalized_tensor


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes


def outputs_to_objects(outputs, img_sizes, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    batch_labels = list(m.indices.detach().cpu().numpy())
    batch_scores = list(m.values.detach().cpu().numpy())
    batch_bboxes = outputs['pred_boxes'].detach().cpu()

    batch_objects = []
    for i in range(len(img_sizes)):
        pred_bboxes = [elem.tolist() for elem in rescale_bboxes(batch_bboxes[i], img_sizes[i])]
        pred_scores = batch_scores[i]
        pred_labels = batch_labels[i]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({
                    'label': class_label,
                    'score': float(score),
                    'bbox': [float(elem) for elem in bbox]}
                )

        rows = []
        cols = []
        for i, cell in enumerate(objects):
            if cell["label"] == "table column":
                cols.append(cell)

            if cell["label"] == "table row":
                rows.append(cell)
        batch_objects.append({
            "rows": rows,
            "cols": cols
        })

    return batch_objects


def load_tatr():
    return AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(settings.TORCH_DEVICE_MODEL)


def batch_inference_tatr(model, images, batch_size):
    device = model.device
    rows_cols = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        pixel_values = torch.stack([structure_transform(img) for img in batch_images], dim=0).to(device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values)

        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"
        rows_cols.extend(outputs_to_objects(outputs, [img.size for img in batch_images], id2label))
    return rows_cols