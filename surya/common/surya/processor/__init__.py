import torch
import einops
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from typing import List, Optional, Tuple

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizer

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.processor.schema import (
    TextInput,
    OCRInput,
    ImageInput,
    EmptyInput,
    ProcessorOutput,
)

# Task agnostic tokens - Every task will use these in some form or another
EOS_TOKEN = "</S>"
EOI_TOKEN = "<EOI>"  # This is end of INPUT, not image. Images are always followed by a task specific BOS token, so that serves as a delimiter anyways.
IMAGE_TOKEN = "<IMAGE>"
PAD_TOKEN = "<PAD>"
NO_OUTPUT_TOKEN = "<NOP>"
IMAGE_ROTATED_TOKEN = "<ROT>"

# Task specific tokens
OCR_WITH_BOXES_BOS_TOKEN = "<OCR-WB>"
OCR_WITHOUT_BOXES_BOS_TOKEN = "<OCR-WOB>"
BLOCK_WITHOUT_BOXES_TOKEN = "<BLOCKS-WOB>"

TOTAL_SPECIAL_TOKENS = 32

class SuryaOCRProcessor(S3DownloaderMixin, ProcessorMixin):
    attributes = ["image_processor", "ocr_tokenizer"]
    image_processor_class = "BaseImageProcessor"
    ocr_tokenizer_class = "PreTrainedTokenizer"

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        ocr_tokenizer: PreTrainedTokenizer,
        tile_size: Tuple[int, int],
        image_tokens_per_tile: int,
        blank_bbox_token_id: int,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.ocr_tokenizer = ocr_tokenizer
        self.tile_size = tile_size
        self.image_tokens_per_tile = image_tokens_per_tile

        self.tokenizer_vocab_size = 0
        for attr in self.attributes:
            if "tokenizer" in attr:
                self.tokenizer_vocab_size += getattr(self, attr).vocab_size

        self.offsets = {"ocr": 0}

        # Create special token mapping
        self.special_token_mapping = {
            k: self.tokenizer_vocab_size + i
            for i, k in enumerate(
                [
                    EOS_TOKEN,
                    PAD_TOKEN,
                    IMAGE_TOKEN,
                    EOI_TOKEN,
                    NO_OUTPUT_TOKEN,
                    IMAGE_ROTATED_TOKEN,  # Task agnostic
                    OCR_WITH_BOXES_BOS_TOKEN,
                    OCR_WITHOUT_BOXES_BOS_TOKEN,
                    BLOCK_WITHOUT_BOXES_TOKEN,  # Task specific
                ]
            )
        }

        # Add in extra reserved tokens for later
        reserved_token_len = TOTAL_SPECIAL_TOKENS - len(self.special_token_mapping)
        special_token_mapping_len = len(self.special_token_mapping)
        for i in range(reserved_token_len):
            self.special_token_mapping[f"<SYS_RESERVED_{i}>"] = (
                self.tokenizer_vocab_size + special_token_mapping_len + i
            )

        self.image_token_id = self.special_token_mapping[IMAGE_TOKEN]
        self.pad_token_id = self.special_token_mapping[PAD_TOKEN]
        self.eos_token_id = self.special_token_mapping[EOS_TOKEN]
        self.eoi_token_id = self.special_token_mapping[EOI_TOKEN]
        self.no_output_token = self.special_token_mapping[NO_OUTPUT_TOKEN]
        self.image_rotated_token = self.special_token_mapping[IMAGE_ROTATED_TOKEN]

        self.bos_token_id = {
            "ocr_with_boxes": self.special_token_mapping[OCR_WITH_BOXES_BOS_TOKEN],
            "ocr_without_boxes": self.special_token_mapping[
                OCR_WITHOUT_BOXES_BOS_TOKEN
            ],
            "block_without_boxes": self.special_token_mapping[
                BLOCK_WITHOUT_BOXES_TOKEN
            ],
        }

        self.blank_bbox_token_id = blank_bbox_token_id
        self.bbox_pad_token_id = self.blank_bbox_token_id

        # Tells us where to insert a bbox blank token
        self.ignore_bbox_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k not in self.ocr_tokenizer.special_tokens["math_external"]
        ]
        math_end_token = "</math>"
        self.math_start_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k in self.ocr_tokenizer.special_tokens["math_external"] and k != math_end_token
        ]
        self.math_end_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k == math_end_token
        ]

        super().__init__(image_processor, ocr_tokenizer)

    @property
    def vocab_size(self):
        return self.tokenizer_vocab_size + len(self.special_token_mapping)

    def _process_and_tile(self, image: Image.Image) -> torch.Tensor:
        """
        Resizes the input image to the closest multiple of tile_size while preserving the aspect ratio
        and returns a tensor of image tiles.

        #TODO Pin to closest aspect ratio  - Currently pins to the next biggest grid of tiles that can fit the full image
        """
        tile_width, tile_height = self.tile_size
        orig_width, orig_height = image.size

        # Compute the scaling factor to maintain aspect ratio
        scale_w = (orig_width + tile_width - 1) // tile_width
        scale_h = (orig_height + tile_height - 1) // tile_height

        new_width = scale_w * tile_width
        new_height = scale_h * tile_height
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Do not perform resizing from the image processor, since resizing is already handled
        img_tensor = self.image_processor(
            resized_image, return_tensors="pt", do_resize=False
        )["pixel_values"][0]
        tiles = einops.rearrange(
            img_tensor,
            "c (ny th) (nx tw) -> (ny nx) c th tw",
            th=tile_height,
            tw=tile_width,
        )
        return tiles

    # Handle image input dictionaries - Process image, tile accordingly, and setup the input ids and boxes correspondingly
    def _process_image_input(self, image_input: ImageInput) -> ProcessorOutput:
        rotated = image_input.get("rotated", False)

        image = image_input.get("image", None)
        assert image is not None, (
            "A PIL Image must be provided when the input type is `image`"
        )
        image_tiles = self._process_and_tile(image)

        num_tiles = image_tiles.shape[0]
        input_ids = [self.image_token_id] * num_tiles * self.image_tokens_per_tile

        # Handle the image being rotated in the imdataset
        if rotated:
            input_ids = [self.image_rotated_token] + input_ids

        input_bboxes = [[self.blank_bbox_token_id] * 6] * len(input_ids)
        lm_labels = [-100] * len(input_ids)
        bbox_labels = [[-100] * 6] * len(input_ids)

        return ProcessorOutput(
            input_ids=input_ids,
            input_boxes=input_bboxes,
            lm_labels=lm_labels,
            bbox_labels=bbox_labels,
            image_tiles=image_tiles,
        )

    def _process_text_input(self, text_input: TextInput) -> ProcessorOutput:
        input_text = text_input.get("text", None)
        input_ids = self.ocr_tokenizer(input_text)["input_ids"][0]
        input_ids = [self.offsets["ocr"] + id for id in input_ids]

        input_bboxes = [[self.blank_bbox_token_id] * 6] * len(input_ids)
        bbox_labels = [[-100] * 6] * len(input_ids)

        return ProcessorOutput(
            input_ids=input_ids,
            input_boxes=input_bboxes,
            lm_labels=input_ids,
            bbox_labels=bbox_labels,
            image_tiles=None,
        )

    def _process_ocr_input(self, ocr_input: OCRInput) -> ProcessorOutput:
        input_ids = ocr_input["tokens"]
        input_bboxes = ocr_input["bboxes"]

        input_ids = [self.offsets["ocr"] + id for id in input_ids]

        if input_bboxes:
            # Replace empty boxes with the corresponding blank token
            input_bboxes = [
                [self.blank_bbox_token_id] * 6 if not b else b for b in input_bboxes
            ]
        else:
            input_bboxes = [[self.blank_bbox_token_id] * 6] * len(input_ids)

        if len(input_bboxes) != len(input_ids):
            print(
                f"Length mismatch - input ids: {len(input_ids)} Input bboxes: {len(input_bboxes)} "
            )
            min_length = min(len(input_ids), len(input_bboxes))
            input_ids = input_ids[:min_length]
            input_bboxes = input_bboxes[:min_length]

        # Return `None` for the image tiles
        return ProcessorOutput(
            input_ids=input_ids,
            input_boxes=input_bboxes,
            image_tiles=None,
        )

    def _process_input(self, input_dict: dict):
        input_type = input_dict["type"]
        if input_type == "image":
            return self._process_image_input(input_dict)
        elif input_type == "ocr":
            return self._process_ocr_input(input_dict)
        elif input_type == "text":
            return self._process_text_input(input_dict)

        raise NotImplementedError(f"Input of type `{input_type}` is not implemented")

    # Peprocessing for OCR task
    # The task is expected to have - image_dict, user_input_dict, output_dict
    # use_input_dict is allowed to have an empty input which is fine, but needs to be present
    def _process_ocr_with_boxes(self, mixed_input: List[dict], bos_token_id: int):
        processed_input_ids = []
        processed_input_boxes = []
        all_image_tiles = []

        # 1. Process the image input
        for i, input_dict in enumerate(mixed_input):
            processor_output = self._process_input(input_dict)
            input_ids = processor_output["input_ids"]
            input_boxes = processor_output["input_boxes"]
            image_tiles = processor_output["image_tiles"]

            # Special handling of some delimiter tokens
            if i == 1:
                assert input_dict["type"] == "text", (
                    "Expected text input for model input."
                )
                # Case for input - Add task specific bos token + end_of_input token
                # We do not want the model to learn how to predict inputs. Hence IGNORE_INDEX for these
                input_ids = [bos_token_id] + input_ids + [self.eoi_token_id]
                input_boxes = (
                    [[self.blank_bbox_token_id] * 6]
                    + input_boxes
                    + [[self.blank_bbox_token_id] * 6]
                )
            elif i == 2:
                # Case for output - No specific bos token, but need to add EOS token
                assert input_dict["type"] == "ocr", (
                    "Expected OCR inputfor model output."
                )

                input_ids = input_ids + [self.eos_token_id]
                input_boxes = input_boxes + [[self.blank_bbox_token_id] * 6]

            # Some input types don't return any image tiles, accounting for that
            if image_tiles is not None:
                all_image_tiles.append(image_tiles)

            processed_input_ids.extend(input_ids)
            processed_input_boxes.append(torch.tensor(input_boxes, dtype=torch.int64))

        return (
            torch.tensor(processed_input_ids, dtype=torch.int64),
            torch.cat(processed_input_boxes, dim=0),
            all_image_tiles,
        )

    def _process_ocr_without_boxes(self, mixed_input: List[dict], bos_token_id: int):
        # Boxes are set to None, so this will work
        # TODO: improve this behavior
        return self._process_ocr_with_boxes(mixed_input, bos_token_id=bos_token_id)

    def _process_block_without_boxes(self, mixed_input: List[dict], bos_token_id: int):
        return self._process_ocr_with_boxes(mixed_input, bos_token_id=bos_token_id)

    def __call__(self, mixed_batch: List[dict], padding_side: Optional[str] = "left"):
        """
        Process a batch of mixed inputs
        Each batch element has two keys:
            'task':
                String that determines how the input is processed
            'inputs':
                list of dictionaries which can contain tokens OR text for a certain type of task (with/without bounding boxes), or an image
                    [{"type": "X", "text": "<YOUR TEXT HERE>}, {"type":"image", "image": PIL.Image}, ....]
                                                                OR
                    [{"type": "X", "tokens"; [1,2,3,4], "bboxes": [bbox1, bbox2, bbox3, bbox4]}, {"type":"image", "image": PIL.Image}, ....]

        NOTE: When inputting raw text, bounding boxes cannot be specified, instead input tokens + aligned bboxes

        This function processes the mixed inputs into a batched set of input_ids + input_boxes + image tiles that the model can consume

        TODO Add row and column separators for image tiles if needed
        TODO Fix - EOS and BOS is incorrect for interleaved inputs
        """

        all_image_tiles = []
        all_input_ids = []
        all_input_boxes = []

        for b in mixed_batch:
            mixed_input = b["inputs"]
            task = b["task"]
            assert task in self.bos_token_id, f"Task {task} has no bos token defined."

            # Select the correct processing function based on the task type
            input_ids, input_boxes, image_tiles = getattr(
                self, f"_process_{task}"
            )(mixed_input, self.bos_token_id[task])

            all_input_ids.append(input_ids)
            all_input_boxes.append(input_boxes)
            all_image_tiles.extend(image_tiles)

        # If max sequence length is None, this slicing does nothing
        batched_input_ids = pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_side=padding_side,
            padding_value=self.pad_token_id,
        )

        batched_input_boxes = pad_sequence(
            all_input_boxes,
            batch_first=True,
            padding_side=padding_side,
            padding_value=self.bbox_pad_token_id,
        )

        attention_mask = batched_input_ids.ne(self.pad_token_id)

        # Generating position IDs that are independent of left and right padding;
        # This should ensure same results for either padding side. Exact position id for the pad tokens themselves don't matter since they are masked
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids[position_ids < 0] = (
            0  # For left padding, the position ids for padding will become -1 because of the shift; Setting to 0
        )
        position_ids = (
            attention_mask.to(torch.long) * position_ids
        )  # Ensure right pad ids get set to zero

        batched_image_tiles = torch.cat(all_image_tiles, dim=0)

        # Returning lm labels as labels, since this is used by HF to calculate num_items_per_batch which is super important for gradient accumulation
        return BatchFeature(
            {
                "input_ids": batched_input_ids,
                "input_boxes": batched_input_boxes,
                "image_tiles": batched_image_tiles,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

    # Decode model outputs; Strips special tokens
    def decode(self, tokens: List[int]):
        filtered_tokens = [
            t
            for t in tokens
            if t not in self.special_token_mapping.values() and t != -100
        ]  # Skip special tokens and loss ignore index
        return self.ocr_tokenizer.decode(filtered_tokens)

