from typing import List

from PIL import Image

from surya.input.processing import convert_if_not_rgb
from surya.foundation import FoundationPredictor
from surya.common.surya.schema import TaskNames
from surya.layout.schema import LayoutBox, LayoutResult
from surya.foundation.util import prediction_to_polygon_batch
from surya.settings import settings


class LayoutPredictor(FoundationPredictor):
    batch_size = settings.LAYOUT_BATCH_SIZE
    default_batch_sizes = {
        "cpu": 4,
        "mps": 4,
        "cuda": 32,
        "xla": 16
    }
    def __init__(self, checkpoint=None, device=settings.TORCH_DEVICE_MODEL, dtype=None):
        super().__init__(checkpoint, device, dtype)
        self.ID2LABEL = {
            self.processor.ocr_tokenizer(tag, "layout")['input_ids'][0][0]: tag for tag in self.processor.ocr_tokenizer.special_tokens['layout']
        }

    def __call__(
        self,
        images: List[Image.Image],
        batch_size: int | None = None,
        top_k: int = 5
    ) -> List[LayoutResult]:
        if batch_size is None:
            batch_size = self.get_batch_size()

        images = convert_if_not_rgb(images)
        orig_sizes = [image.size for image in images]
        images = [self.processor.image_processor(image) for image in images]
        task_names = [TaskNames.layout] * len(images)
        input_texts = [""] * len(images)
        
        predicted_tokens, batch_bboxes, predicted_scores = self.prediction_loop(
            processed_images=images,
            input_texts=input_texts,
            task_names=task_names,
            batch_size=batch_size,
            math_mode=False
        )

        bbox_size = self.model.config.bbox_size
        # Need to reverse the tuples because the function expects (h, w) and PIL does (w,h)
        predicted_polygons = prediction_to_polygon_batch(
            batch_bboxes, [o[::-1] for o in orig_sizes], bbox_size, bbox_size // 2
        )

        results = []
        for tokens, polygons, scores, orig_size in zip(predicted_tokens, predicted_polygons, predicted_scores, orig_sizes):
            layout_boxes = []
            polygons = polygons.cpu().numpy().tolist()
            for idx, (tok, poly, score) in enumerate(zip(tokens, polygons, scores)):
                # TODO EOS token handling
                if tok not in self.ID2LABEL:
                    break
                label = self.ID2LABEL[tok]
                top_k_dict = {
                    label: score
                }
                layout_boxes.append(LayoutBox(
                    label=label,
                    polygon=poly,
                    position=idx,
                    top_k=top_k_dict
                ))
            results.append(LayoutResult(
                bboxes=layout_boxes,
                image_bbox=[0, 0, *orig_size],
                sliced=False
            ))

        return results