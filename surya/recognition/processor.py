from transformers import DonutProcessor

from surya.common.donut.processor import SuryaEncoderImageProcessor
from surya.recognition.tokenizer import Byt5LangTokenizer
from surya.settings import settings


class SuryaProcessor(DonutProcessor):
    def __init__(self, checkpoint, revision, image_processor=None, tokenizer=None, **kwargs):
        image_processor = SuryaEncoderImageProcessor.from_pretrained(checkpoint, revision=revision)
        image_processor.do_align_long_axis = True
        image_processor.max_size = settings.RECOGNITION_IMAGE_SIZE
        tokenizer = Byt5LangTokenizer()
        tokenizer.model_max_length = settings.RECOGNITION_MAX_TOKENS
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        langs = kwargs.pop("langs", None)

        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)

        if text is not None:
            encodings = self.tokenizer(text, langs, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            inputs["langs"] = encodings["langs"]
            return inputs