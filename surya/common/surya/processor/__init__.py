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
    ImageInput,
    ProcessorOutput,
)
from surya.common.surya.schema import TaskNames

# Task agnostic tokens - Every task will use these in some form or another
EOS_TOKEN = "</S>"
EOI_TOKEN = "<EOI>"  # This is end of INPUT, not image. Images are always followed by a task specific BOS token, so that serves as a delimiter anyways.
IMAGE_TOKEN = "<IMAGE>"
PAD_TOKEN = "<PAD>"
NO_OUTPUT_TOKEN = "<NOP>"
IMAGE_ROTATED_TOKEN = "<ROT>"
REGISTER_TOKENS = ["<REG1>", "<REG2>", "<REG3>", "<REG4>"]
NOMATH_TOKEN = "<NO-MATH>"

# Task specific tokens
OCR_WITH_BOXES_BOS_TOKEN = "<OCR-WB>"
OCR_WITHOUT_BOXES_BOS_TOKEN = "<OCR-WOB>"
BLOCK_WITHOUT_BOXES_TOKEN = "<BLOCKS-WOB>"


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
        num_register_tokens: int,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.ocr_tokenizer = ocr_tokenizer
        self.tile_size = tile_size
        self.image_tokens_per_tile = image_tokens_per_tile
        self.num_register_tokens = num_register_tokens

        self.tokenizer_vocab_size = 0
        for attr in self.attributes:
            if "tokenizer" in attr:
                self.tokenizer_vocab_size += getattr(self, attr).vocab_size

        self.offsets = {"ocr": 0}

        # Create special token mapping
        self.special_token_mapping = self.ocr_tokenizer.system_tokens

        self.register_token_ids = [
            self.special_token_mapping.get(r) for r in REGISTER_TOKENS
        ]
        self.image_token_id = self.special_token_mapping.get(IMAGE_TOKEN)
        self.pad_token_id = self.special_token_mapping.get(PAD_TOKEN)
        self.eos_token_id = self.special_token_mapping.get(EOS_TOKEN)
        self.eoi_token_id = self.special_token_mapping.get(EOI_TOKEN)
        self.no_output_token = self.special_token_mapping.get(NO_OUTPUT_TOKEN)
        self.image_rotated_token = self.special_token_mapping.get(IMAGE_ROTATED_TOKEN)
        self.nomath_token = self.special_token_mapping.get(NOMATH_TOKEN)

        self.bos_token_id = {
            TaskNames.ocr_with_boxes: self.special_token_mapping.get(
                OCR_WITH_BOXES_BOS_TOKEN
            ),
            TaskNames.ocr_without_boxes: self.special_token_mapping.get(
                OCR_WITHOUT_BOXES_BOS_TOKEN
            ),
            TaskNames.block_without_boxes: self.special_token_mapping.get(
                BLOCK_WITHOUT_BOXES_TOKEN
            ),
        }

        if self.image_token_id is None:
            print("Warning: Image token not found in special tokens")

        self.blank_bbox_token_id = blank_bbox_token_id
        self.bbox_pad_token_id = self.blank_bbox_token_id

        self.ignore_bbox_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k not in self.ocr_tokenizer.special_tokens["math_external"]
        ]
        math_end_token = "</math>"
        self.math_start_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k in self.ocr_tokenizer.special_tokens["math_external"]
            and k != math_end_token
        ]
        self.math_end_token_ids = [
            v
            for (k, v) in self.ocr_tokenizer.SPECIAL_TOKEN_MAPPING.items()
            if k == math_end_token
        ]

        super().__init__(image_processor, ocr_tokenizer)

        if self.num_register_tokens > len(self.register_token_ids):
            raise ValueError(
                "The number of register tokens requested exceeds the number of register tokens defined in the special token mapping."
            )

    @property
    def vocab_size(self):
        return self.tokenizer_vocab_size

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
        input_ids += [self.register_token_ids][: self.num_register_tokens]

        # Handle the image being rotated in the imdataset
        if rotated:
            input_ids = [self.image_rotated_token] + input_ids

        return ProcessorOutput(
            input_ids=input_ids,
            image_tiles=image_tiles,
        )

    def _process_text_input(self, text_input: TextInput, task: str) -> ProcessorOutput:
        input_text = text_input.get("text", None)
        math_mode = text_input.get("math", False)

        input_ids = self.ocr_tokenizer(input_text, tasks=task)["input_ids"][0]
        input_ids = [self.offsets["ocr"] + id for id in input_ids]

        if not math_mode:
            input_ids.insert(0, self.nomath_token)

        return ProcessorOutput(
            input_ids=input_ids,
            image_tiles=None,
        )

    def _process_input(self, input_dict: dict, task: str):
        input_type = input_dict["type"]
        if input_type == "image":
            return self._process_image_input(input_dict)
        elif input_type == "text":
            return self._process_text_input(input_dict, task)

        raise NotImplementedError(f"Input of type `{input_type}` is not implemented")

    # Peprocessing for OCR task
    # The task is expected to have - image_dict, user_input_dict, output_dict
    # use_input_dict is allowed to have an empty input which is fine, but needs to be present
    def _process_ocr_with_boxes(
        self,
        mixed_input: List[dict],
        bos_token_id: int,
        task: str = TaskNames.ocr_with_boxes,
    ):
        processed_input_ids = []
        all_image_tiles = []

        # 1. Process the image input
        for i, input_dict in enumerate(mixed_input):
            processor_output = self._process_input(input_dict, task)
            input_ids = processor_output["input_ids"]
            image_tiles = processor_output["image_tiles"]

            # Special handling of some delimiter tokens
            if i == 1:
                assert input_dict["type"] == "text", (
                    "Expected text input for model input."
                )
                # Case for input - Add task specific bos token + end_of_input token
                # We do not want the model to learn how to predict inputs. Hence IGNORE_INDEX for these
                input_ids = [bos_token_id] + input_ids + [self.eoi_token_id]
            elif i > 1:
                raise ValueError(
                    f"Unexpected input type encountered in mixed input processing. We only accept input image and text. {mixed_input}"
                )

            # Some input types don't return any image tiles, accounting for that
            if image_tiles is not None:
                all_image_tiles.append(image_tiles)

            processed_input_ids.extend(input_ids)

        return (
            torch.tensor(processed_input_ids, dtype=torch.int64),
            all_image_tiles,
        )

    def _process_ocr_without_boxes(
        self,
        mixed_input: List[dict],
        bos_token_id: int,
        task: str = "ocr_without_boxes",
    ):
        # Boxes are set to None, so this will work
        # TODO: improve this behavior
        return self._process_ocr_with_boxes(
            mixed_input, bos_token_id=bos_token_id, task=task
        )

    def _process_block_without_boxes(
        self,
        mixed_input: List[dict],
        bos_token_id: int,
        task: str = "block_without_boxes",
    ):
        return self._process_ocr_with_boxes(
            mixed_input, bos_token_id=bos_token_id, task=task
        )

    def align_long_axis(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        width, height = image.size
        if height > width:  # Rotate vertical lines
            return image.rotate(90, expand=True), True

        return image, False

    def __call__(self, mixed_batch: List[dict], padding_side: Optional[str] = "left"):
        all_image_tiles = []
        all_input_ids = []

        for b in mixed_batch:
            mixed_input = b["inputs"]
            task = b["task"]
            assert task in self.bos_token_id, f"Task {task} has no bos token defined."

            # Select the correct processing function based on the task type
            input_ids, image_tiles = getattr(self, f"_process_{task}")(
                mixed_input, self.bos_token_id[task]
            )

            all_input_ids.append(input_ids)
            all_image_tiles.extend(image_tiles)

        # If max sequence length is None, this slicing does nothing
        batched_input_ids = pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_side=padding_side,
            padding_value=self.pad_token_id,
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
                "image_tiles": batched_image_tiles,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

    # Decode model outputs; Strips special tokens
    def decode(self, tokens: List[int], task: str):
        filtered_tokens = [
            t
            for t in tokens
            if t not in self.special_token_mapping.values() and t != -100
        ]  # Skip special tokens and loss ignore index
        return self.ocr_tokenizer.decode(filtered_tokens, task=task)
