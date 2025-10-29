# Copyright (c) Microsoft. All rights reserved.

"""Agent Framework拡張のためのカスタムOpenAI互換イベントタイプ。

これらは標準のOpenAI Responses APIを超えて拡張されたカスタムイベントタイプで、
workflowやtraceなどAgent Framework固有の機能をサポートします。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# 構造化データのためのカスタムAgent Framework OpenAIイベントタイプ エージェントのライフサイクルイベント - シンプルで明確
class AgentStartedEvent:
    """エージェントが実行を開始したときに発行されるイベント。"""

    pass


class AgentCompletedEvent:
    """エージェントが正常に実行を完了したときに発行されるイベント。"""

    pass


@dataclass
class AgentFailedEvent:
    """エージェントが実行中に失敗したときに発行されるイベント。"""

    error: Exception | None = None


class ExecutorActionItem(BaseModel):
    """workflow executorアクションのためのカスタムアイテムタイプ。

    これはDevUI固有の拡張で、workflow executorを出力アイテムとして表現します。
    OpenAIのResponseOutputItemAddedEventは特定のアイテムタイプのみを受け入れるため、
    executorアクションは標準の一部ではないため、このカスタムタイプが必要です。

    """

    type: Literal["executor_action"] = "executor_action"
    id: str
    executor_id: str
    status: Literal["in_progress", "completed", "failed", "cancelled"] = "in_progress"
    metadata: dict[str, Any] | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


class CustomResponseOutputItemAddedEvent(BaseModel):
    """任意のアイテムタイプを受け入れるResponseOutputItemAddedEventのカスタムバージョン。

    これにより、OpenAIの標準と同じイベント構造を維持しつつ、executorアクションアイテムを発行できます。

    """

    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    sequence_number: int
    item: dict[str, Any] | ExecutorActionItem | Any  # 柔軟なアイテムタイプ


class CustomResponseOutputItemDoneEvent(BaseModel):
    """任意のアイテムタイプを受け入れるResponseOutputItemDoneEventのカスタムバージョン。

    これにより、OpenAIの標準と同じイベント構造を維持しながら、executorのアクションアイテムを発行できます。

    """

    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    sequence_number: int
    item: dict[str, Any] | ExecutorActionItem | Any  # 柔軟なアイテムタイプ


class ResponseWorkflowEventComplete(BaseModel):
    """完全なワークフローイベントデータ。"""

    type: Literal["response.workflow_event.complete"] = "response.workflow_event.complete"
    data: dict[str, Any]  # デルタではない完全なイベントデータ
    executor_id: str | None = None
    item_id: str
    output_index: int = 0
    sequence_number: int


class ResponseTraceEventComplete(BaseModel):
    """完全なトレースイベントデータ。"""

    type: Literal["response.trace.complete"] = "response.trace.complete"
    data: dict[str, Any]  # デルタではない完全なトレースデータ
    span_id: str | None = None
    item_id: str
    output_index: int = 0
    sequence_number: int


class ResponseFunctionResultComplete(BaseModel):
    """DevUI拡張：関数実行結果のストリーム。

    これはDevUI拡張です。理由は以下の通りです：
    - OpenAI Responses APIは関数結果をストリームしません（クライアントが関数を実行します）
    - Agent Frameworkはサーバー側で関数を実行するため、デバッグの可視性のために結果をストリームします
    - ResponseFunctionToolCallOutputItemはOpenAI SDKに存在しますが、ResponseOutputItemのユニオンには含まれていません
      （Conversations APIの入力用であり、Responses APIのストリーミング出力用ではありません）

    このイベントはOpenAIの関数出力アイテムと同じ構造を提供しますが、標準イベントが関数結果のストリーミングをサポートしないため、カスタムイベントタイプでラップされています。

    """

    type: Literal["response.function_result.complete"] = "response.function_result.complete"
    call_id: str
    output: str
    status: Literal["in_progress", "completed", "incomplete"]
    item_id: str
    output_index: int = 0
    sequence_number: int
    timestamp: str | None = None  # UI表示用のオプショナルなタイムスタンプ


# Agent Framework拡張フィールド
class AgentFrameworkExtraBody(BaseModel):
    """OpenAIリクエスト用のAgent Framework固有のルーティングフィールド。"""

    entity_id: str
    # input_dataは削除され、すべてのデータに標準のinputフィールドを使用するようになりました

    model_config = ConfigDict(extra="allow")


# Agent Frameworkリクエストモデル - 実際のOpenAIタイプを拡張
class AgentFrameworkRequest(BaseModel):
    """Agent Frameworkルーティングを含むOpenAI ResponseCreateParams。

    これは実際のOpenAI APIリクエスト形式を適切に拡張しています。
    - 'model'フィールドをentity_id（agent/workflow名）として使用
    - 'conversation'フィールドを会話コンテキストとして使用（OpenAI標準）

    """

    # ResponseCreateParamsのすべてのOpenAIフィールド
    model: str  # DevUIでentity_idとして使用！
    input: str | list[Any] | dict[str, Any]  # ResponseInputParam + ワークフロー構造化入力用のdict
    stream: bool | None = False

    # OpenAI会話パラメータ（標準！）
    conversation: str | dict[str, Any] | None = None  # Union[str, {"id": str}]

    # 共通のOpenAIオプショナルフィールド
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None

    # 高度なユースケース用のオプショナルなextra_body
    extra_body: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")

    def get_entity_id(self) -> str:
        """modelフィールドからentity_idを取得。

        DevUIでは、modelがentity_id（agent/workflow名）です。
        シンプルでクリーン！

        """
        return self.model

    def get_conversation_id(self) -> str | None:
        """conversationパラメータからconversation_idを抽出。

        文字列形式とオブジェクト形式の両方をサポート：
        - conversation: "conv_123"
        - conversation: {"id": "conv_123"}

        """
        if isinstance(self.conversation, str):
            return self.conversation
        if isinstance(self.conversation, dict):
            return self.conversation.get("id")
        return None

    def to_openai_params(self) -> dict[str, Any]:
        """OpenAIクライアント互換のためにdictに変換。"""
        return self.model_dump(exclude_none=True)


# エラー処理
class ResponseTraceEvent(BaseModel):
    """実行トレース用のトレースイベント。"""

    type: Literal["trace_event"] = "trace_event"
    data: dict[str, Any]
    timestamp: str


class OpenAIError(BaseModel):
    """OpenAI標準のエラーレスポンスモデル。"""

    error: dict[str, Any]

    @classmethod
    def create(cls, message: str, type: str = "invalid_request_error", code: str | None = None) -> OpenAIError:
        """標準のOpenAIエラーレスポンスを作成。"""
        error_data = {"message": message, "type": type, "code": code}
        return cls(error=error_data)

    def to_dict(self) -> dict[str, Any]:
        """エラーペイロードをプレーンなマッピングとして返す。"""
        return {"error": dict(self.error)}

    def to_json(self) -> str:
        """エラーペイロードをJSONにシリアライズして返す。"""
        return self.model_dump_json()


# すべてのカスタムタイプをエクスポート
__all__ = [
    "AgentFrameworkRequest",
    "OpenAIError",
    "ResponseFunctionResultComplete",
    "ResponseTraceEvent",
    "ResponseTraceEventComplete",
    "ResponseWorkflowEventComplete",
]
