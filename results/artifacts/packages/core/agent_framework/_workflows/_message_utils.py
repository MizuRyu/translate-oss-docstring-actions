# Copyright (c) Microsoft. All rights reserved.

"""ワークフローメッセージ入力の正規化のための共有ヘルパー。"""

from collections.abc import Sequence

from agent_framework import ChatMessage, Role


def normalize_messages_input(
    messages: str | ChatMessage | Sequence[str | ChatMessage] | None = None,
) -> list[ChatMessage]:
    """異種のメッセージ入力をChatMessageオブジェクトのリストに正規化します。

    Args:
        messages: 文字列、ChatMessage、またはそのいずれかのシーケンス。Noneの場合は空リストを返します。

    Returns:
        ワークフローで使用可能なChatMessageインスタンスのリスト。

    """
    if messages is None:
        return []

    if isinstance(messages, str):
        return [ChatMessage(role=Role.USER, text=messages)]

    if isinstance(messages, ChatMessage):
        return [messages]

    normalized: list[ChatMessage] = []
    for item in messages:
        if isinstance(item, str):
            normalized.append(ChatMessage(role=Role.USER, text=item))
        elif isinstance(item, ChatMessage):
            normalized.append(item)
        else:
            raise TypeError(
                f"Messages sequence must contain only str or ChatMessage instances; found {type(item).__name__}."
            )
    return normalized


__all__ = ["normalize_messages_input"]
