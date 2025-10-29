# Copyright (c) Microsoft. All rights reserved.

"""会話の流れと参加者選択を管理するグループチャットオーケストレーターの基底クラス。"""

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .._types import ChatMessage
from ._executor import Executor
from ._orchestrator_helpers import ParticipantRegistry
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class BaseGroupChatOrchestrator(Executor, ABC):
    """グループチャットオーケストレーターの抽象基底クラス。

    参加者登録、ルーティング、ラウンド制限チェックなど、全てのグループチャットパターンで共通の機能を提供する。

    サブクラスはパターン固有のオーケストレーションロジックを実装しつつ、共通の参加者管理基盤を継承する必要がある。

    """

    def __init__(self, executor_id: str) -> None:
        """基底オーケストレーターを初期化する。

        Args:
            executor_id: このオーケストレーターExecutorの一意識別子

        """
        super().__init__(executor_id)
        self._registry = ParticipantRegistry()
        # 共有の会話状態管理
        self._conversation: list[ChatMessage] = []
        self._round_index: int = 0
        self._max_rounds: int | None = None
        self._termination_condition: Callable[[list[ChatMessage]], bool | Awaitable[bool]] | None = None

    def register_participant_entry(self, name: str, *, entry_id: str, is_agent: bool) -> None:
        """参加者のエントリーExecutorのルーティング詳細を記録する。

        このメソッドは全てのオーケストレーターパターンで参加者（AgentまたはカスタムExecutor）を登録する統一インターフェースを提供する。

        Args:
            name: 参加者名（選択と追跡に使用）
            entry_id: この参加者のエントリーポイントのExecutor ID
            is_agent: AgentExecutorならTrue、カスタムExecutorならFalse

        """
        self._registry.register(name, entry_id=entry_id, is_agent=is_agent)

    # 会話状態管理（全パターン共通）

    def _append_messages(self, messages: Sequence[ChatMessage]) -> None:
        """会話履歴にメッセージを追加する。

        Args:
            messages: 追加するメッセージ

        """
        self._conversation.extend(messages)

    def _get_conversation(self) -> list[ChatMessage]:
        """現在の会話のコピーを取得する。

        Returns:
            複製された会話リスト

        """
        return list(self._conversation)

    def _clear_conversation(self) -> None:
        """会話履歴をクリアする。"""
        self._conversation.clear()

    def _increment_round(self) -> None:
        """ラウンドカウンターをインクリメントする。"""
        self._round_index += 1

    async def _check_termination(self) -> bool:
        """終了条件に基づいて会話を終了すべきかチェックする。

        同期および非同期の終了条件の両方をサポート。

        Returns:
            終了条件を満たす場合はTrue、そうでなければFalse

        """
        if self._termination_condition is None:
            return False

        result = self._termination_condition(self._get_conversation())
        if inspect.iscoroutine(result) or inspect.isawaitable(result):
            result = await result
        return bool(result)

    @abstractmethod
    def _get_author_name(self) -> str:
        """オーケストレーター生成メッセージの著者名を取得する。

        サブクラスは完了メッセージやその他オーケストレーター生成コンテンツのために安定した著者名を提供するためにこれを実装する必要がある。

        Returns:
            このオーケストレーターが生成するメッセージに使用する著者名

        """
        ...

    def _create_completion_message(
        self,
        text: str | None = None,
        reason: str = "completed",
    ) -> ChatMessage:
        """標準化された完了メッセージを作成する。

        Args:
            text: 任意のメッセージテキスト（Noneの場合は自動生成）
            reason: デフォルトテキストの完了理由

        Returns:
            完了内容を含むChatMessage

        """
        from .._types import Role

        message_text = text or f"Conversation {reason}."
        return ChatMessage(
            role=Role.ASSISTANT,
            text=message_text,
            author_name=self._get_author_name(),
        )

    # 参加者ルーティング（全パターン共通）

    async def _route_to_participant(
        self,
        participant_name: str,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[Any, Any],
        *,
        instruction: str | None = None,
        task: ChatMessage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """会話を参加者にルーティングする。

        このメソッドはデュアルエンベロープパターンを処理する:
        - AgentExecutorsはAgentExecutorRequest（メッセージのみ）を受け取る
        - カスタムExecutorはGroupChatRequestMessage（完全なコンテキスト）を受け取る

        Args:
            participant_name: ルーティング先の参加者名
            conversation: 送信する会話履歴
            ctx: メッセージルーティングのためのワークフローコンテキスト
            instruction: マネージャー/オーケストレーターからの任意の指示
            task: 任意のタスクコンテキスト
            metadata: 任意のメタデータ辞書

        Raises:
            ValueError: 参加者が登録されていない場合

        """
        from ._agent_executor import AgentExecutorRequest
        from ._orchestrator_helpers import prepare_participant_request

        entry_id = self._registry.get_entry_id(participant_name)
        if entry_id is None:
            raise ValueError(f"No registered entry executor for participant '{participant_name}'.")

        if self._registry.is_agent(participant_name):
            # AgentExecutorsは単純なメッセージリストを受け取る
            await ctx.send_message(
                AgentExecutorRequest(messages=conversation, should_respond=True),
                target_id=entry_id,
            )
        else:
            # カスタムExecutorは完全なコンテキストエンベロープを受け取る
            request = prepare_participant_request(
                participant_name=participant_name,
                conversation=conversation,
                instruction=instruction or "",
                task=task,
                metadata=metadata,
            )
            await ctx.send_message(request, target_id=entry_id)

    # ラウンド制限の強制（全パターン共通）

    def _check_round_limit(self) -> bool:
        """ラウンド制限に達したかチェックする。

        インスタンス変数_round_indexと_max_roundsを使用。

        Returns:
            制限に達した場合はTrue、そうでなければFalse

        """
        if self._max_rounds is None:
            return False

        if self._round_index >= self._max_rounds:
            logger.warning(
                "%s reached max_rounds=%s; forcing completion.",
                self.__class__.__name__,
                self._max_rounds,
            )
            return True

        return False

    # 状態永続化（全パターン共通） 状態永続化（全パターン共通）

    def snapshot_state(self) -> dict[str, Any]:
        """チェックポイント用に現在のオーケストレーター状態をキャプチャする。

        デフォルト実装はOrchestrationStateを使って共通状態をシリアライズする。
        サブクラスはパターン固有のデータを追加するために_snapshot_pattern_metadata()をオーバーライドすべき。

        Returns:
            シリアライズされた状態の辞書

        """
        from ._orchestration_state import OrchestrationState

        state = OrchestrationState(
            conversation=list(self._conversation),
            round_index=self._round_index,
            metadata=self._snapshot_pattern_metadata(),
        )
        return state.to_dict()

    def _snapshot_pattern_metadata(self) -> dict[str, Any]:
        """パターン固有の状態をシリアライズする。

        このメソッドをオーバーライドしてパターン固有のチェックポイントデータを追加する。

        Returns:
            パターン固有の状態を含む辞書（デフォルトは空）

        """
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        """チェックポイントからオーケストレーター状態を復元する。

        デフォルト実装はOrchestrationStateを使って共通状態をデシリアライズする。
        サブクラスはパターン固有のデータを復元するために_restore_pattern_metadata()をオーバーライドすべき。

        Args:
            state: シリアライズされた状態の辞書

        """
        from ._orchestration_state import OrchestrationState

        orch_state = OrchestrationState.from_dict(state)
        self._conversation = list(orch_state.conversation)
        self._round_index = orch_state.round_index
        self._restore_pattern_metadata(orch_state.metadata)

    def _restore_pattern_metadata(self, metadata: dict[str, Any]) -> None:
        """パターン固有の状態を復元する。

        このメソッドをオーバーライドしてパターン固有のチェックポイントデータを復元する。

        Args:
            metadata: パターン固有の状態辞書

        """
        pass
