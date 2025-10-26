import logging
import os
import sys


def init_logging() -> logging.Logger:
    """ロギングを初期化してルートロガーを返す。"""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger_keep = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )
    logger_keep.addHandler(handler)
    logger_keep.setLevel(logging._nameToLevel.get(log_level, logging.INFO))
    return logger_keep

logger = init_logging()