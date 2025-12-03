import logging
import sys


def setup_logger(level: str = "INFO"):
    logger = logging.getLogger("camera_app")
    logger.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
