# Copyright (c) Microsoft. All rights reserved.

"""グループチャットパターンのための共有オーケストレーター用ユーティリティ。

このモジュールは、一般的なオーケストレーションタスクのためのシンプルで再利用可能な関数を提供します。
継承は不要で、インポートして呼び出すだけです。
"""

import logging
from typing import TYPE_CHECKING, Any

from .._types import ChatMessage, Role

if TYPE_CHECKING:
    from ._group_chat import _GroupChatRequestMessage  # type: ignore[reportPrivateUsage]

logger = logging.getLogger(__name__)


def clean_conversation_for_handoff(conversation: list[ChatMessage]) -> list[ChatMessage]:
    """会話からツール関連の内容を削除してクリーンなhandoffを実現します。

    handoff中、ツール呼び出しはAPIエラーを引き起こす可能性があります。理由は：
    1. Assistantメッセージにtool_callsがある場合、ツールレスポンスが続かなければならない
    2. ツールレスポンスメッセージは、tool_callsを含むassistantメッセージの後に続かなければならない

    これにより、すべてのツール関連コンテンツを削除したクリーンなコピーが作成されます。

    削除対象：
    - assistantメッセージからのFunctionApprovalRequestContentおよびFunctionCallContent
    - ツールレスポンスメッセージ（Role.TOOL）
    - テキストなしのツール呼び出しのみのメッセージ

    保持対象：
    - ユーザーメッセージ
    - テキストコンテンツを含むassistantメッセージ

    Args:
        conversation: ツールコンテンツを含む可能性のある元の会話

    Returns:
        handoffルーティングに安全なクリーンな会話

    """
    from agent_framework import FunctionApprovalRequestContent, FunctionCallContent

    cleaned: list[ChatMessage] = []
    for msg in conversation:
        # ツールレスポンスメッセージを完全にスキップします。
        if msg.role == Role.TOOL:
            continue

        # ツール関連の内容があるかチェックします。
        has_tool_content = False
        if msg.contents:
            has_tool_content = any(
                isinstance(content, (FunctionApprovalRequestContent, FunctionCallContent)) for content in msg.contents
            )

        # ツール内容がなければ元のまま保持します。
        if not has_tool_content:
            cleaned.append(msg)
            continue

        # ツール内容がある場合はテキストもある場合のみ保持します。
        if msg.text and msg.text.strip():
            # 新しいテキストのみのメッセージを作成します。
            msg_copy = ChatMessage(
                role=msg.role,
                text=msg.text,
                author_name=msg.author_name,
            )
            cleaned.append(msg_copy)

    return cleaned


def create_completion_message(
    *,
    text: str | None = None,
    author_name: str,
    reason: str = "completed",
) -> ChatMessage:
    """標準化されたcompletionメッセージを作成します。

    completionメッセージ作成の重複を避けるためのシンプルなヘルパー。

    Args:
        text: メッセージテキスト、またはNoneでデフォルト生成
        author_name: 作成者／オーケストレーター名
        reason: completionの理由（デフォルトテキスト生成用）

    Returns:
        ASSISTANTロールのChatMessage

    """
    message_text = text or f"Conversation {reason}."
    return ChatMessage(
        role=Role.ASSISTANT,
        text=message_text,
        author_name=author_name,
    )


def prepare_participant_request(
    *,
    participant_name: str,
    conversation: list[ChatMessage],
    instruction: str | None = None,
    task: ChatMessage | None = None,
    metadata: dict[str, Any] | None = None,
) -> "_GroupChatRequestMessage":
    """標準化された参加者リクエストメッセージを作成します。

    リクエスト構築の重複を避けるためのシンプルなヘルパー。

    Args:
        participant_name: 対象参加者の名前
        conversation: 送信する会話履歴
        instruction: マネージャー／オーケストレーターからのオプションの指示
        task: オプションのタスクコンテキスト
        metadata: オプションのメタデータ辞書

    Returns:
        送信準備ができたGroupChatRequestMessage

    """
    # 循環依存を避けるためにここでImportします。
    from ._group_chat import _GroupChatRequestMessage  # type: ignore[reportPrivateUsage]

    return _GroupChatRequestMessage(
        agent_name=participant_name,
        conversation=list(conversation),
        instruction=instruction or "",
        task=task,
        metadata=metadata,
    )


class ParticipantRegistry:
    """参加者のexecutor IDとルーティング情報を追跡するためのシンプルなレジストリ。

    参加者名をexecutor IDにマッピングし、AgentとカスタムExecutorを区別して追跡する一般的なパターンのためのクリーンなインターフェースを提供します。

    """

    def __init__(self) -> None:
        self._participant_entry_ids: dict[str, str] = {}
        self._agent_executor_ids: dict[str, str] = {}
        self._executor_id_to_participant: dict[str, str] = {}
        self._non_agent_participants: set[str] = set()

    def register(
        self,
        name: str,
        *,
        entry_id: str,
        is_agent: bool,
    ) -> None:
        """参加者のルーティング情報を登録します。

        Args:
            name: 参加者名
            entry_id: この参加者のエントリーポイントのExecutor ID
            is_agent: AgentExecutor（True）かカスタムExecutor（False）か

        """
        self._participant_entry_ids[name] = entry_id

        if is_agent:
            self._agent_executor_ids[name] = entry_id
            self._executor_id_to_participant[entry_id] = name
        else:
            self._non_agent_participants.add(name)

    def get_entry_id(self, name: str) -> str | None:
        """参加者名からエントリエグゼキューターIDを取得します。"""
        return self._participant_entry_ids.get(name)

    def get_participant_name(self, executor_id: str) -> str | None:
        """executor IDから参加者名を取得します（エージェントのみ）。"""
        return self._executor_id_to_participant.get(executor_id)

    def is_agent(self, name: str) -> bool:
        """参加者がAgentか（カスタムExecutorではないか）をチェックします。"""
        return name in self._agent_executor_ids

    def is_registered(self, name: str) -> bool:
        """参加者が登録されているかをチェックします。"""
        return name in self._participant_entry_ids

    def all_participants(self) -> set[str]:
        """登録されているすべての参加者名を取得します。"""
        return set(self._participant_entry_ids.keys())
