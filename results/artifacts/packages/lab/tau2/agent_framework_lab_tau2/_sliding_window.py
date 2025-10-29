# Copyright (c) Microsoft. All rights reserved.

import json
from collections.abc import Sequence
from typing import Any

import tiktoken
from agent_framework import ChatMessage, ChatMessageStore, Role
from loguru import logger


class SlidingWindowChatMessageStore(ChatMessageStore):
    """ChatMessageStoreのトークン認識スライディングウィンドウ実装。

    完全な履歴と切り詰められたウィンドウの2つのメッセージリストを維持。
    トークン制限を超えた場合は自動的に最も古いメッセージを削除。
    また、有効な会話フローを確保するために先頭のツールメッセージも削除。

    """

    def __init__(
        self,
        messages: Sequence[ChatMessage] | None = None,
        max_tokens: int = 3800,
        system_message: str | None = None,
        tool_definitions: Any | None = None,
    ):
        super().__init__(messages=messages)
        self.truncated_messages = self.messages.copy()
        self.max_tokens = max_tokens
        self.system_message = system_message  # トークン数に含まれる
        self.tool_definitions = tool_definitions
        # 一般的に使用される語彙表に基づく推定値
        self.encoding = tiktoken.get_encoding("o200k_base")

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        await super().add_messages(messages)

        self.truncated_messages = self.messages.copy()
        self.truncate_messages()

    async def list_messages(self) -> list[ChatMessage]:
        """現在のメッセージリストを取得（切り詰められている可能性あり）。"""
        return self.truncated_messages

    async def list_all_messages(self) -> list[ChatMessage]:
        """切り詰められたものも含めてストア内のすべてのメッセージを取得する。"""
        return self.messages

    def truncate_messages(self) -> None:
        while len(self.truncated_messages) > 0 and self.get_token_count() > self.max_tokens:
            logger.warning("Messages exceed max tokens. Truncating oldest message.")
            self.truncated_messages.pop(0)
        # 先頭のツールメッセージを削除する
        while len(self.truncated_messages) > 0 and self.truncated_messages[0].role == Role.TOOL:
            logger.warning("Removing leading tool message because tool result cannot be the first message.")
            self.truncated_messages.pop(0)

    def get_token_count(self) -> int:
        """tiktokenを使ってメッセージリストのトークン数を推定する。

        Returns:
            推定トークン数

        """
        total_tokens = 0

        # システムメッセージのトークンを追加（提供されていれば）
        if self.system_message:
            total_tokens += len(self.encoding.encode(self.system_message))
            total_tokens += 4  # システムメッセージのフォーマット用の追加トークン

        for msg in self.truncated_messages:
            # 役割やフォーマットなどでメッセージごとに4トークンを追加
            total_tokens += 4

            # 異なるコンテンツタイプを処理する
            if hasattr(msg, "contents") and msg.contents:
                for content in msg.contents:
                    if hasattr(content, "type"):
                        if content.type == "text":
                            total_tokens += len(self.encoding.encode(content.text))
                        elif content.type == "function_call":
                            total_tokens += 4
                            # 関数呼び出しをシリアライズし、トークン数をカウントする
                            func_call_data = {
                                "name": content.name,
                                "arguments": content.arguments,
                            }
                            total_tokens += self.estimate_any_object_token_count(func_call_data)
                        elif content.type == "function_result":
                            total_tokens += 4
                            # 関数結果をシリアライズし、トークン数をカウントする
                            func_result_data = {
                                "call_id": content.call_id,
                                "result": content.result,
                            }
                            total_tokens += self.estimate_any_object_token_count(func_result_data)
                        else:
                            # その他のコンテンツタイプの場合は、コンテンツ全体をシリアライズする
                            total_tokens += self.estimate_any_object_token_count(content)
                    else:
                        # タイプなしのコンテンツはテキストとして扱う
                        total_tokens += self.estimate_any_object_token_count(content)
            elif hasattr(msg, "text") and msg.text:
                # シンプルなテキストメッセージ
                total_tokens += self.estimate_any_object_token_count(msg.text)
            else:
                # スキップする
                pass

        if total_tokens > self.max_tokens / 2:
            logger.opt(colors=True).warning(
                f"<yellow>Total tokens {total_tokens} is "
                f"{total_tokens / self.max_tokens * 100:.0f}% "
                f"of max tokens {self.max_tokens}</yellow>"
            )
        elif total_tokens > self.max_tokens:
            logger.opt(colors=True).warning(
                f"<red>Total tokens {total_tokens} is over max tokens {self.max_tokens}. Will truncate messages.</red>"
            )

        return total_tokens

    def estimate_any_object_token_count(self, obj: Any) -> int:
        try:
            serialized = json.dumps(obj)
        except Exception:
            serialized = str(obj)
        return len(self.encoding.encode(serialized))
