from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import tiktoken


def init_logger() -> logging.Logger:
    """ロガーを初期化（stdout出力のみ）"""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("comment_translator")
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(logging._nameToLevel.get(level_name, logging.INFO))
    return logger


logger = init_logger()


@lru_cache(maxsize=1)
def _encoding(name: str = "o200k_base"):
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    return len(_encoding(encoding_name).encode(text))
