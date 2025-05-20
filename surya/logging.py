import logging
import warnings
from surya.settings import settings


def configure_logging():
    # Setup surya logger
    logger = get_logger()

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(settings.LOGLEVEL)
    warnings.simplefilter(action="ignore", category=FutureWarning)


def get_logger():
    return logging.getLogger("surya")
