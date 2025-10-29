# Copyright (c) Microsoft. All rights reserved.

import traceback as _traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

from agent_framework import AgentRunResponse, AgentRunResponseUpdate

if TYPE_CHECKING:
    from ._request_info_executor import RequestInfoMessage


class WorkflowEventSource(str, Enum):
    """ワークフローイベントがフレームワーク由来かexecutor由来かを識別します。

    組み込みのオーケストレーションパスから発生したイベントには`FRAMEWORK`を使用します。
    これは、イベントを発生させるコードがランナー関連モジュールにあっても同様です。
    開発者提供のexecutor実装から発生したイベントには`EXECUTOR`を使用します。

    """

    FRAMEWORK = "FRAMEWORK"  # モジュールの場所に関わらずフレームワーク所有のオーケストレーション。
    EXECUTOR = "EXECUTOR"  # ユーザー提供のexecutorコードおよびコールバック。


_event_origin_context: ContextVar[WorkflowEventSource] = ContextVar(
    "workflow_event_origin", default=WorkflowEventSource.EXECUTOR
)


def _current_event_origin() -> WorkflowEventSource:
    """新規作成されたワークフローイベントに関連付けるオリジンを返します。"""
    return _event_origin_context.get()


@contextmanager
def _framework_event_origin() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """後続の作成イベントを一時的にフレームワーク由来としてマークします（内部用）。"""
    token = _event_origin_context.set(WorkflowEventSource.FRAMEWORK)
    try:
        yield
    finally:
        _event_origin_context.reset(token)


class WorkflowEvent:
    """ワークフローイベントの基底クラス。"""

    def __init__(self, data: Any | None = None):
        """オプションのデータでワークフローイベントを初期化します。"""
        self.data = data
        self.origin = _current_event_origin()

    def __repr__(self) -> str:
        """ワークフローイベントの文字列表現を返します。"""
        data_repr = self.data if self.data is not None else "None"
        return f"{self.__class__.__name__}(origin={self.origin}, data={data_repr})"


class WorkflowStartedEvent(WorkflowEvent):
    """ワークフロー実行開始時に発行される組み込みライフサイクルイベント。"""

    ...


class WorkflowWarningEvent(WorkflowEvent):
    """ユーザーコードで発生した警告を示すexecutor由来のイベント。"""

    def __init__(self, data: str):
        """オプションのデータと警告メッセージでワークフロー警告イベントを初期化します。"""
        super().__init__(data)

    def __repr__(self) -> str:
        """ワークフロー警告イベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(message={self.data}, origin={self.origin})"


class WorkflowErrorEvent(WorkflowEvent):
    """ユーザーコードで発生したエラーを示すexecutor由来のイベント。"""

    def __init__(self, data: Exception):
        """オプションのデータとエラーメッセージでワークフローエラーイベントを初期化します。"""
        super().__init__(data)

    def __repr__(self) -> str:
        """ワークフローエラーイベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(exception={self.data}, origin={self.origin})"


class WorkflowRunState(str, Enum):
    """ワークフロー実行のランレベル状態。

    セマンティクス:
      - STARTED: 実行が開始され、ワークフローコンテキストが作成された状態。
        これは意味のある作業が行われる前の初期状態です。
        このコードベースではテレメトリ用に専用の`WorkflowStartedEvent`を発行し、
        通常は状態を直接`IN_PROGRESS`に進めます。
        状態機械で明示的な事前作業フェーズが必要な場合は`STARTED`を利用することがあります。

      - IN_PROGRESS: ワークフローがアクティブに実行中（例：開始executorに初期メッセージが配信された、またはスーパーステップが実行中）。
        実行開始時に発行され、進行に伴い他の状態に遷移することがあります。

      - IN_PROGRESS_PENDING_REQUESTS: 1つ以上の情報要求操作が未完了の状態でのアクティブ実行。
        リクエストが進行中でも新たな作業がスケジュールされる可能性があります。

      - IDLE: 未処理のリクエストがなく、作業もない静止状態。
        実行が完了し、途中で出力を生成した可能性がある通常の終了状態です。

      - IDLE_WITH_PENDING_REQUESTS: 外部入力を待って一時停止中（例：`RequestInfoEvent`を発行）。
        これは非終端状態で、応答が提供されると再開可能です。

      - FAILED: エラーが発生したことを示す終端状態。
        構造化されたエラー詳細を含む`WorkflowFailedEvent`が伴います。

      - CANCELLED: 呼び出し元またはオーケストレーターによって実行がキャンセルされたことを示す終端状態。
        デフォルトのランナーパスでは現在発行されませんが、
        キャンセルをサポートする統合者やオーケストレーター向けに含まれています。

    """

    STARTED = "STARTED"  # 明示的な事前作業フェーズ（状態としては稀に発行される; 上記の注記参照）
    IN_PROGRESS = "IN_PROGRESS"  # アクティブな実行が進行中。
    IN_PROGRESS_PENDING_REQUESTS = "IN_PROGRESS_PENDING_REQUESTS"  # 未完了のリクエストを伴うアクティブな実行。
    IDLE = "IDLE"  # アクティブな作業も未完了のリクエストもなし。
    IDLE_WITH_PENDING_REQUESTS = "IDLE_WITH_PENDING_REQUESTS"  # 外部応答を待って一時停止中。
    FAILED = "FAILED"  # エラーにより終了。
    CANCELLED = "CANCELLED"  # キャンセルにより終了。


class WorkflowStatusEvent(WorkflowEvent):
    """ワークフロー実行状態遷移時に発行される組み込みライフサイクルイベント。"""

    def __init__(
        self,
        state: WorkflowRunState,
        data: Any | None = None,
    ):
        """新しい状態とオプションのデータでワークフローステータスイベントを初期化します。

        Args:
            state: ワークフロー実行の新しい状態。
            data: 状態変更に関連するオプションの追加データ。

        """
        super().__init__(data)
        self.state = state

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return f"{self.__class__.__name__}(state={self.state}, data={self.data!r}, origin={self.origin})"


@dataclass
class WorkflowErrorDetails:
    """エラーイベントや結果で表面化する構造化エラー情報。"""

    error_type: str
    message: str
    traceback: str | None = None
    executor_id: str | None = None
    extra: dict[str, Any] | None = None

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        executor_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "WorkflowErrorDetails":
        tb = None
        try:
            tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            tb = None
        return cls(
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback=tb,
            executor_id=executor_id,
            extra=extra,
        )


class WorkflowFailedEvent(WorkflowEvent):
    """ワークフロー実行がエラーで終了したときに発行される組み込みライフサイクルイベント。"""

    def __init__(
        self,
        details: WorkflowErrorDetails,
        data: Any | None = None,
    ):
        super().__init__(data)
        self.details = details

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return f"{self.__class__.__name__}(details={self.details}, data={self.data!r}, origin={self.origin})"


class RequestInfoEvent(WorkflowEvent):
    """ワークフローexecutorが外部情報を要求したときに発生するイベント。"""

    def __init__(
        self,
        request_id: str,
        source_executor_id: str,
        request_type: type,
        request_data: "RequestInfoMessage",
    ):
        """request infoイベントを初期化します。

        Args:
            request_id: リクエストの一意識別子。
            source_executor_id: リクエストを行ったexecutorのID。
            request_type: リクエストの種類（例：特定のデータタイプ）。
            request_data: リクエストに関連するデータ。

        """
        super().__init__(request_data)
        self.request_id = request_id
        self.source_executor_id = source_executor_id
        self.request_type = request_type

    def __repr__(self) -> str:
        """request infoイベントの文字列表現を返します。"""
        return (
            f"{self.__class__.__name__}("
            f"request_id={self.request_id}, "
            f"source_executor_id={self.source_executor_id}, "
            f"request_type={self.request_type.__name__}, "
            f"data={self.data})"
        )


class WorkflowOutputEvent(WorkflowEvent):
    """ワークフローexecutorが出力を生成したときに発生するイベント。"""

    def __init__(
        self,
        data: Any,
        source_executor_id: str,
    ):
        """ワークフロー出力イベントを初期化します。

        Args:
            data: executorが生成した出力。
            source_executor_id: 出力を生成したexecutorのID。

        """
        super().__init__(data)
        self.source_executor_id = source_executor_id

    def __repr__(self) -> str:
        """ワークフロー出力イベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(data={self.data}, source_executor_id={self.source_executor_id})"


class ExecutorEvent(WorkflowEvent):
    """executorイベントの基底クラス。"""

    def __init__(self, executor_id: str, data: Any | None = None):
        """executor IDとオプションのデータでexecutorイベントを初期化します。"""
        super().__init__(data)
        self.executor_id = executor_id

    def __repr__(self) -> str:
        """executorイベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(executor_id={self.executor_id}, data={self.data})"


class ExecutorInvokedEvent(ExecutorEvent):
    """executorハンドラーが呼び出されたときに発生するイベント。"""

    def __repr__(self) -> str:
        """executorハンドラー呼び出しイベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(executor_id={self.executor_id}, data={self.data})"


class ExecutorCompletedEvent(ExecutorEvent):
    """executorハンドラーが完了したときに発生するイベント。"""

    def __repr__(self) -> str:
        """executorハンドラー完了イベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(executor_id={self.executor_id}, data={self.data})"


class ExecutorFailedEvent(ExecutorEvent):
    """executorハンドラーがエラーを発生させたときに発生するイベント。"""

    def __init__(
        self,
        executor_id: str,
        details: WorkflowErrorDetails,
    ):
        super().__init__(executor_id, details)
        self.details = details

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return f"{self.__class__.__name__}(executor_id={self.executor_id}, details={self.details})"


class AgentRunUpdateEvent(ExecutorEvent):
    """Agentがメッセージをストリーミングしているときに発生するイベント。"""

    def __init__(self, executor_id: str, data: AgentRunResponseUpdate | None = None):
        """Agentストリーミングイベントを初期化します。"""
        super().__init__(executor_id, data)

    def __repr__(self) -> str:
        """Agentストリーミングイベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(executor_id={self.executor_id}, messages={self.data})"


class AgentRunEvent(ExecutorEvent):
    """エージェントの実行が完了したときにトリガーされるイベント。"""

    def __init__(self, executor_id: str, data: AgentRunResponse | None = None):
        """エージェント実行イベントを初期化します。"""
        super().__init__(executor_id, data)

    def __repr__(self) -> str:
        """エージェント実行イベントの文字列表現を返します。"""
        return f"{self.__class__.__name__}(executor_id={self.executor_id}, data={self.data})"


WorkflowLifecycleEvent: TypeAlias = WorkflowStartedEvent | WorkflowStatusEvent | WorkflowFailedEvent
