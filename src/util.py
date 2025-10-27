from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import tiktoken


def init_logger() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("comment_translator")
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_path = Path("run.log")
        try:
            log_path.write_text("", encoding="utf-8")
        except Exception:
            pass
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging._nameToLevel.get(level_name, logging.INFO))
    return logger


logger = init_logger()


@lru_cache(maxsize=1)
def _encoding(name: str = "o200k_base"):
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    return len(_encoding(encoding_name).encode(text))
