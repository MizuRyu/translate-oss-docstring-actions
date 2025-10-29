# Copyright (c) Microsoft. All rights reserved.
"""Purview固有の例外（最小限のエラー整形）。"""

from __future__ import annotations

from agent_framework.exceptions import ServiceResponseException

__all__ = [
    "PurviewAuthenticationError",
    "PurviewRateLimitError",
    "PurviewRequestError",
    "PurviewServiceError",
]


class PurviewServiceError(ServiceResponseException):
    """Purviewエラーの基本例外クラス。"""


class PurviewAuthenticationError(PurviewServiceError):
    """認証／認可失敗（401/403）。"""


class PurviewRateLimitError(PurviewServiceError):
    """レート制限またはスロットリング（429）。"""


class PurviewRequestError(PurviewServiceError):
    """その他の非成功HTTPエラー。"""
