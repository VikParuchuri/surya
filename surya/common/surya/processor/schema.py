from typing import TypedDict, Literal, List, Any, Tuple

import torch
from PIL import Image


class TaskDict(TypedDict):
    datasets: List[str]
    img_size: Tuple[int, int]


class TasksDict(TypedDict):
    ocr_with_boxes: TaskDict
    ocr_without_boxes: TaskDict
    block_without_boxes: TaskDict


class ProcessorInput(TypedDict):
    type: Literal["image", "ocr", "text", "empty_output"]


class ImageInput(ProcessorInput):
    type: Literal["image"]
    image: Image.Image
    rotated: bool


class TextInput(ProcessorInput):
    type: Literal["text"]
    text: str


class OCRInput(ProcessorInput):
    type: Literal["ocr"]
    tokens: List[int]
    bboxes: List[List[int]] | List[Any]


class EmptyInput(ProcessorInput):
    type: Literal["empty_output"]


class ProcessorOutput(TypedDict):
    input_ids: List[int]
    input_boxes: List[List[int]]
    image_tiles: torch.Tensor | None
