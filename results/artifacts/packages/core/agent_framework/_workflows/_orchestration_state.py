# Copyright (c) Microsoft. All rights reserved.

"""グループチャットオーケストレーターのための統一された状態管理。

GroupChat、Handoff、Magenticパターン間で標準化されたチェックポイントシリアライズのためのOrchestrationStateデータクラスを提供します。
"""

from dataclasses import dataclass, field
from typing import Any

from .._types import ChatMessage


def _new_chat_message_list() -> list[ChatMessage]:
    """型付き空のChatMessageリストのファクトリー関数。

    型チェッカーを満たします。

    """
    return []


def _new_metadata_dict() -> dict[str, Any]:
    """型付き空のメタデータ辞書のファクトリー関数。

    型チェッカーを満たします。

    """
    return {}


@dataclass
class OrchestrationState:
    """オーケストレーターのチェックポイント用の統一された状態コンテナ。

    このデータクラスは、3つのグループチャットパターン全体でチェックポイントのシリアライズを標準化し、
    メタデータを通じてパターン固有の拡張を可能にします。

    共通属性は共有のオーケストレーションの関心事（タスク、会話、ラウンド追跡）をカバーし、
    パターン固有の状態はメタデータ辞書に格納されます。

    Attributes:
        conversation: 完全な会話履歴（すべてのメッセージ）
        round_index: 完了した調整ラウンドの数（追跡されていない場合は0）
        metadata: パターン固有の状態のための拡張可能な辞書
        task: オプショナルな主要タスク／質問

    """

    conversation: list[ChatMessage] = field(default_factory=_new_chat_message_list)
    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=_new_metadata_dict)
    task: ChatMessage | None = None

    def to_dict(self) -> dict[str, Any]:
        """チェックポイント用に辞書にシリアライズします。

        Returns:
            永続化のためにエンコードされたconversationとmetadataを含む辞書

        """
        from ._conversation_state import encode_chat_messages

        result: dict[str, Any] = {
            "conversation": encode_chat_messages(self.conversation),
            "round_index": self.round_index,
            "metadata": dict(self.metadata),
        }
        if self.task is not None:
            result["task"] = encode_chat_messages([self.task])[0]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationState":
        """チェックポイント化された辞書からデシリアライズします。

        Args:
            data: エンコードされたconversationを含むチェックポイントデータ

        Returns:
            復元されたOrchestrationStateインスタンス

        """
        from ._conversation_state import decode_chat_messages

        task = None
        if "task" in data:
            decoded_tasks = decode_chat_messages([data["task"]])
            task = decoded_tasks[0] if decoded_tasks else None

        return cls(
            conversation=decode_chat_messages(data.get("conversation", [])),
            round_index=data.get("round_index", 0),
            metadata=dict(data.get("metadata", {})),
            task=task,
        )
