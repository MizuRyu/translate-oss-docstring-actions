# Copyright (c) Microsoft. All rights reserved.

import logging

from .exceptions import AgentFrameworkException

logging.basicConfig(
    format="[%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

__all__ = ["get_logger"]


def get_logger(name: str = "agent_framework") -> logging.Logger:
    """指定された名前のロガーを取得します。デフォルトは 'agent_framework' です。

    Args:
        name (str): ロガーの名前。デフォルトは 'agent_framework'。

    Returns:
        logging.Logger: 設定されたロガーのインスタンス。

    """
    if not name.startswith("agent_framework"):
        raise AgentFrameworkException("Logger name must start with 'agent_framework'.")
    return logging.getLogger(name)
