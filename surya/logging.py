import logging
import warnings


def configure_logging():
    # Setup surya logger
    logger = logging.getLogger("surya")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)
    warnings.simplefilter(action="ignore", category=FutureWarning)


def get_logger():
    return logging.getLogger("surya")
