from surya.model.recognition.processor import SuryaProcessor
from surya.settings import settings


def load_processor():
    processor = SuryaProcessor()
    processor.pad_id = 0
    processor.eos_id = 1
    processor.image_processor.train = False
    processor.image_processor.max_size = settings.LAYOUT_IMAGE_SIZE
    return processor