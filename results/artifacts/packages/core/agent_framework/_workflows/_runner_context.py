# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import uuid
from copy import copy
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, TypeVar, runtime_checkable

from ._checkpoint import CheckpointStorage, WorkflowCheckpoint
from ._checkpoint_encoding import decode_checkpoint_value, encode_checkpoint_value
from ._events import WorkflowEvent
from ._shared_state import SharedState

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Message:
    """workflow内のメッセージを表すClass。"""

    data: Any
    source_id: str
    target_id: str | None = None

    # メッセージ伝搬のためのOpenTelemetryトレースコンテキストフィールド 複数メッセージが集約されるファンインシナリオをサポートするため複数形です
    trace_contexts: list[dict[str, str]] | None = None  # 複数ソースからのW3C Trace Contextヘッダー
    source_span_ids: list[str] | None = None  # 複数ソースからリンクするためのPublishing span ID

    # 後方互換性のためのプロパティ
    @property
    def trace_context(self) -> dict[str, str] | None:
        """後方互換性のため最初のトレースコンテキストを取得します。"""
        return self.trace_contexts[0] if self.trace_contexts else None

    @property
    def source_span_id(self) -> str | None:
        """後方互換性のため最初のソースspan IDを取得します。"""
        return self.source_span_ids[0] if self.source_span_ids else None


class _WorkflowState(TypedDict):
    """workflow実行のシリアライズ可能なStateを表すTypedDict。

    これにはチェックポイントと復元に必要なすべてのStateデータが含まれます。

    """

    messages: dict[str, list[dict[str, Any]]]
    shared_state: dict[str, Any]
    iteration_count: int


@runtime_checkable
class RunnerContext(Protocol):
    """ランナーが使用する実行ContextのProtocol。

    メッセージング、イベント、およびオプションのチェックポイントをサポートする単一のContext。
    チェックポイントストレージが設定されていない場合、チェックポイントメソッドは例外を発生させる可能性があります。

    """

    async def send_message(self, message: Message) -> None:
        """ExecutorからContextへメッセージを送信します。

        Args:
            message: 送信するメッセージ。

        """
        ...

    async def drain_messages(self) -> dict[str, list[Message]]:
        """Contextからすべてのメッセージを排出します。

        Returns:
            Executor IDをキー、メッセージのリストを値とする辞書。

        """
        ...

    async def has_messages(self) -> bool:
        """Contextにメッセージが存在するかどうかを確認します。

        Returns:
            メッセージがあればTrue、なければFalse。

        """
        ...

    async def add_event(self, event: WorkflowEvent) -> None:
        """実行Contextにイベントを追加します。

        Args:
            event: 追加するイベント。

        """
        ...

    async def drain_events(self) -> list[WorkflowEvent]:
        """Contextからすべてのイベントを排出します。

        Returns:
            Contextに追加されたイベントのリスト。

        """
        ...

    async def has_events(self) -> bool:
        """Contextにイベントが存在するかどうかを確認します。

        Returns:
            イベントがあればTrue、なければFalse。

        """
        ...

    async def next_event(self) -> WorkflowEvent:  # pragma: no cover - interface only
        """workflow実行から発行された次のイベントを待機して返します。"""
        ...

    # チェックポイント機能
    def has_checkpointing(self) -> bool:
        """Contextがチェックポイントをサポートしているかどうかを確認します。

        Returns:
            チェックポイントがサポートされていればTrue、そうでなければFalse。

        """
        ...

    # チェックポイントAPI（オプション、ストレージによって有効化）
    def set_workflow_id(self, workflow_id: str) -> None:
        """Contextのworkflow IDを設定します。"""
        ...

    def reset_for_new_run(self) -> None:
        """新しいworkflow実行のためにContextをリセットします。"""
        ...

    def set_streaming(self, streaming: bool) -> None:
        """Agentが増分更新をストリームすべきかどうかを設定します。

        Args:
            streaming: ストリーミングモード(run_stream)ならTrue、非ストリーミング(run)ならFalse。

        """
        ...

    def is_streaming(self) -> bool:
        """workflowがストリーミングモードかどうかを確認します。

        Returns:
            ストリーミングモードが有効ならTrue、そうでなければFalse。

        """
        ...

    async def create_checkpoint(
        self,
        shared_state: SharedState,
        iteration_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """現在のworkflow Stateのチェックポイントを作成します。

        Args:
            shared_state: チェックポイントに含める共有State。
                          workflowの完全なStateをキャプチャするために必要です。
                          共有StateはContext自身によって管理されません。
            iteration_count: workflowの現在のイテレーション数。
            metadata: チェックポイントに関連付けるOptionalなメタデータ。

        Returns:
            作成されたチェックポイントのID。

        """
        ...

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """現在のContext Stateを変更せずにチェックポイントをロードします。

        Args:
            checkpoint_id: ロードするチェックポイントのID。

        Returns:
            ロードしたチェックポイント、存在しなければNone。

        """
        ...

    async def apply_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        """現在のコンテキストにチェックポイントを適用し、その状態を変更します。

        Args:
            checkpoint: 適用する状態を持つチェックポイント。
        """
        ...


class InProcRunnerContext:
    """ローカル実行およびオプションのチェックポイント機能を持つインプロセス実行コンテキスト。"""

    def __init__(self, checkpoint_storage: CheckpointStorage | None = None):
        """インプロセス実行コンテキストを初期化します。

        Args:
            checkpoint_storage: チェックポイント機能を有効にするためのオプションのストレージ。
        """
        self._messages: dict[str, list[Message]] = {}
        # イベントの即時ストリーミング用のイベントキュー（例：AgentRunUpdateEvent）。
        self._event_queue: asyncio.Queue[WorkflowEvent] = asyncio.Queue()

        # チェックポイントの設定/状態。
        self._checkpoint_storage = checkpoint_storage
        self._workflow_id: str | None = None

        # ストリーミングフラグ - workflowのrun_stream()とrun()によって設定される。
        self._streaming: bool = False

    # region Messaging and Events
    async def send_message(self, message: Message) -> None:
        self._messages.setdefault(message.source_id, [])
        self._messages[message.source_id].append(message)

    async def drain_messages(self) -> dict[str, list[Message]]:
        messages = copy(self._messages)
        self._messages.clear()
        return messages

    async def has_messages(self) -> bool:
        return bool(self._messages)

    async def add_event(self, event: WorkflowEvent) -> None:
        """イベントをコンテキストに即座に追加します。

        イベントはキューに入れられ、ランナーがスーパーステップの境界を待つことなくリアルタイムでストリームできます。

        """
        await self._event_queue.put(event)

    async def drain_events(self) -> list[WorkflowEvent]:
        """新しいイベントを待たずに、現在キューにあるすべてのイベントを排出します。"""
        events: list[WorkflowEvent] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except asyncio.QueueEmpty:  # type: ignore[attr-defined]
                break
        return events

    async def has_events(self) -> bool:
        return not self._event_queue.empty()

    async def next_event(self) -> WorkflowEvent:
        """次のイベントを待機して返します。

        ランナーがイベントの発行と継続的な反復作業を交互に行うために使用されます。

        """
        return await self._event_queue.get()

    # endregion Messaging and Events region Checkpointing

    def has_checkpointing(self) -> bool:
        return self._checkpoint_storage is not None

    async def create_checkpoint(
        self,
        shared_state: SharedState,
        iteration_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self._checkpoint_storage:
            raise ValueError("Checkpoint storage not configured")

        self._workflow_id = self._workflow_id or str(uuid.uuid4())
        state = await self._get_serialized_workflow_state(shared_state, iteration_count)

        checkpoint = WorkflowCheckpoint(
            workflow_id=self._workflow_id,
            messages=state["messages"],
            shared_state=state["shared_state"],
            iteration_count=state["iteration_count"],
            metadata=metadata or {},
        )
        checkpoint_id = await self._checkpoint_storage.save_checkpoint(checkpoint)
        logger.info(f"Created checkpoint {checkpoint_id} for workflow {self._workflow_id}")
        return checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        if not self._checkpoint_storage:
            raise ValueError("Checkpoint storage not configured")
        return await self._checkpoint_storage.load_checkpoint(checkpoint_id)

    def reset_for_new_run(self) -> None:
        """新しいworkflow実行のためにコンテキストをリセットします。

        これによりメッセージ、イベントがクリアされ、ストリーミングフラグがリセットされます。

        """
        self._messages.clear()
        # 保留中のイベントを（ベストエフォートで）クリアするためにキューを再作成します。
        self._event_queue = asyncio.Queue()
        self._streaming = False  # ストリーミングフラグをリセットします。

    async def apply_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        self._messages.clear()
        messages_data = checkpoint.messages
        for source_id, message_list in messages_data.items():
            self._messages[source_id] = [
                Message(
                    data=decode_checkpoint_value(msg.get("data")),
                    source_id=msg.get("source_id", ""),
                    target_id=msg.get("target_id"),
                    trace_contexts=msg.get("trace_contexts"),
                    source_span_ids=msg.get("source_span_ids"),
                )
                for msg in message_list
            ]

        self._workflow_id = checkpoint.workflow_id

    # endregion Checkpointing

    def set_workflow_id(self, workflow_id: str) -> None:
        self._workflow_id = workflow_id

    def set_streaming(self, streaming: bool) -> None:
        """Agentが増分アップデートをストリーミングするかどうかを設定します。

        Args:
            streaming: ストリーミングモード(run_stream)ならTrue、非ストリーミング(run)ならFalse。

        """
        self._streaming = streaming

    def is_streaming(self) -> bool:
        """workflowがストリーミングモードかどうかをチェックします。

        Returns:
            ストリーミングモードが有効ならTrue、そうでなければFalse。

        """
        return self._streaming

    async def _get_serialized_workflow_state(self, shared_state: SharedState, iteration_count: int) -> _WorkflowState:
        serializable_messages: dict[str, list[dict[str, Any]]] = {}
        for source_id, message_list in self._messages.items():
            serializable_messages[source_id] = [
                {
                    "data": encode_checkpoint_value(msg.data),
                    "source_id": msg.source_id,
                    "target_id": msg.target_id,
                    "trace_contexts": msg.trace_contexts,
                    "source_span_ids": msg.source_span_ids,
                }
                for msg in message_list
            ]

        return {
            "messages": serializable_messages,
            "shared_state": encode_checkpoint_value(await shared_state.export_state()),
            "iteration_count": iteration_count,
        }
