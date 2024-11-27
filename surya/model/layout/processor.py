import logging

from surya.model.common.donut.processor import SuryaEncoderImageProcessor
from surya.settings import settings

logger = logging.getLogger()


def load_processor():
    processor = SuryaEncoderImageProcessor(max_size=settings.LAYOUT_IMAGE_SIZE)
    return processor