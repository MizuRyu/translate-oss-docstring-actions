# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai import BadRequestError

from ..exceptions import ServiceContentFilterException

__all__ = ["ContentFilterResultSeverity", "OpenAIContentFilterException"]


class ContentFilterResultSeverity(Enum):
    """コンテンツフィルター結果の重大度。"""

    HIGH = "high"
    MEDIUM = "medium"
    SAFE = "safe"
    LOW = "low"


@dataclass
class ContentFilterResult:
    """コンテンツフィルターのチェック結果。"""

    filtered: bool = False
    detected: bool = False
    severity: ContentFilterResultSeverity = ContentFilterResultSeverity.SAFE

    @classmethod
    def from_inner_error_result(cls, inner_error_results: dict[str, Any]) -> "ContentFilterResult":
        """内部のエラー結果からContentFilterResultを作成します。

        引数:
            inner_error_results: 内部のエラー結果。

        戻り値:
            ContentFilterResult: ContentFilterResultオブジェクト。

        """
        return cls(
            filtered=inner_error_results.get("filtered", False),
            detected=inner_error_results.get("detected", False),
            severity=ContentFilterResultSeverity(
                inner_error_results.get("severity", ContentFilterResultSeverity.SAFE.value)
            ),
        )


class ContentFilterCodes(Enum):
    """コンテンツフィルターコード。"""

    RESPONSIBLE_AI_POLICY_VIOLATION = "ResponsibleAIPolicyViolation"


@dataclass
class OpenAIContentFilterException(ServiceContentFilterException):
    """Azure OpenAIのコンテンツフィルターからのエラーに対するAI例外。"""

    # エラーを引き起こしたパラメータ。
    param: str | None

    # コンテンツフィルター固有のエラーコード。
    content_filter_code: ContentFilterCodes

    # 異なるコンテンツフィルターチェックの結果。
    content_filter_result: dict[str, ContentFilterResult]

    def __init__(
        self,
        message: str,
        inner_exception: BadRequestError,
    ) -> None:
        """ContentFilterAIExceptionクラスの新しいインスタンスを初期化します。

        引数:
            message: エラーメッセージ。
            inner_exception: 内部例外。

        """
        super().__init__(message)

        self.param = inner_exception.param
        if inner_exception.body is not None and isinstance(inner_exception.body, dict):
            inner_error = inner_exception.body.get("innererror", {})  # type: ignore
            self.content_filter_code = ContentFilterCodes(
                inner_error.get("code", ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION.value)  # type: ignore
            )
            self.content_filter_result = {
                key: ContentFilterResult.from_inner_error_result(values)  # type: ignore
                for key, values in inner_error.get("content_filter_result", {}).items()  # type: ignore
            }
