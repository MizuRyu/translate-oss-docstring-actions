# Copyright (c) Microsoft. All rights reserved.
import importlib.metadata

from ._chat_message_store import RedisChatMessageStore
from ._provider import RedisProvider

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # 開発モードのフォールバック

__all__ = [
    "RedisChatMessageStore",
    "RedisProvider",
    "__version__",
]
