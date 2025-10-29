# Copyright (c) Microsoft. All rights reserved.

import sys
from collections.abc import MutableSequence, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from agent_framework import ChatMessage, Context, ContextProvider
from agent_framework.exceptions import ServiceInitializationError
from mem0 import AsyncMemory, AsyncMemoryClient

if sys.version_info >= (3, 11):
    from typing import NotRequired, Self, TypedDict  # pragma: no cover
else:
    from typing_extensions import NotRequired, Self, TypedDict  # pragma: no cover

if sys.version_info >= (3, 12):
    from typing import override  # type: ignore # pragma: no cover
else:
    from typing_extensions import override  # type: ignore[import] # pragma: no cover


# Mem0検索レスポンスフォーマットの型エイリアス（v1.1およびv2；v1は非推奨ですがv2の型定義に一致します）
class MemorySearchResponse_v1_1(TypedDict):
    results: list[dict[str, Any]]
    relations: NotRequired[list[dict[str, Any]]]


MemorySearchResponse_v2 = list[dict[str, Any]]


class Mem0Provider(ContextProvider):
    """Mem0 Context Provider。"""

    def __init__(
        self,
        mem0_client: AsyncMemory | AsyncMemoryClient | None = None,
        api_key: str | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        scope_to_per_operation_thread_id: bool = False,
        context_prompt: str = ContextProvider.DEFAULT_CONTEXT_PROMPT,
    ) -> None:
        """Mem0Providerクラスの新しいインスタンスを初期化します。

        Args:
            mem0_client: 事前に作成されたMem0 MemoryClient、またはデフォルトクライアントを作成するためのNone。
            api_key: Mem0 API認証用のAPIキー。指定しない場合はMEM0_API_KEY環境変数を使用しようとします。
            application_id: メモリのスコープ用のアプリケーションID、またはNone。
            agent_id: メモリのスコープ用のエージェントID、またはNone。
            thread_id: メモリのスコープ用のスレッドID、またはNone。
            user_id: メモリのスコープ用のユーザーID、またはNone。
            scope_to_per_operation_thread_id: メモリを操作ごとのスレッドIDにスコープするかどうか。
            context_prompt: 取得したメモリに先行して付加するプロンプト。

        """
        should_close_client = False
        if mem0_client is None:
            mem0_client = AsyncMemoryClient(api_key=api_key)
            should_close_client = True

        self.api_key = api_key
        self.application_id = application_id
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.user_id = user_id
        self.scope_to_per_operation_thread_id = scope_to_per_operation_thread_id
        self.context_prompt = context_prompt
        self.mem0_client = mem0_client
        self._per_operation_thread_id: str | None = None
        self._should_close_client = should_close_client

    async def __aenter__(self) -> "Self":
        """非同期コンテキストマネージャのエントリ。"""
        if self.mem0_client and isinstance(self.mem0_client, AbstractAsyncContextManager):
            await self.mem0_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """非同期コンテキストマネージャの終了。"""
        if self._should_close_client and self.mem0_client and isinstance(self.mem0_client, AbstractAsyncContextManager):
            await self.mem0_client.__aexit__(exc_type, exc_val, exc_tb)

    async def thread_created(self, thread_id: str | None = None) -> None:
        """新しいスレッドが作成されたときに呼び出されます。

        Args:
            thread_id: スレッドのID、またはNone。

        """
        self._validate_per_operation_thread_id(thread_id)
        self._per_operation_thread_id = self._per_operation_thread_id or thread_id

    @override
    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        self._validate_filters()

        request_messages_list = (
            [request_messages] if isinstance(request_messages, ChatMessage) else list(request_messages)
        )
        response_messages_list = (
            [response_messages]
            if isinstance(response_messages, ChatMessage)
            else list(response_messages)
            if response_messages
            else []
        )
        messages_list = [*request_messages_list, *response_messages_list]

        messages: list[dict[str, str]] = [
            {"role": message.role.value, "content": message.text}
            for message in messages_list
            if message.role.value in {"user", "assistant", "system"} and message.text and message.text.strip()
        ]

        if messages:
            await self.mem0_client.add(  # type: ignore[misc]
                messages=messages,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self._per_operation_thread_id if self.scope_to_per_operation_thread_id else self.thread_id,
                metadata={"application_id": self.application_id},
            )

    @override
    async def invoking(self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any) -> Context:
        """AIモデルを呼び出す前にコンテキストを提供するために呼び出されます。

        Args:
            messages: スレッド内の新しいメッセージのリスト。

        Keyword Args:
            **kwargs: 現在は使用されていません。

        Returns:
            Context: メモリを含む指示を持つContextオブジェクト。

        """
        self._validate_filters()
        messages_list = [messages] if isinstance(messages, ChatMessage) else list(messages)
        input_text = "\n".join(msg.text for msg in messages_list if msg and msg.text and msg.text.strip())

        search_response: MemorySearchResponse_v1_1 | MemorySearchResponse_v2 = await self.mem0_client.search(  # type: ignore[misc]
            query=input_text,
            user_id=self.user_id,
            agent_id=self.agent_id,
            run_id=self._per_operation_thread_id if self.scope_to_per_operation_thread_id else self.thread_id,
        )

        # APIバージョンに応じてレスポンススキーマがわずかに異なります
        if isinstance(search_response, list):
            memories = search_response
        elif isinstance(search_response, dict) and "results" in search_response:
            memories = search_response["results"]
        else:
            # 予期しないスキーマに対するフォールバック - レスポンスをテキストとしてそのまま返します
            memories = [search_response]

        line_separated_memories = "\n".join(memory.get("memory", "") for memory in memories)

        return Context(
            messages=[ChatMessage(role="user", text=f"{self.context_prompt}\n{line_separated_memories}")]
            if line_separated_memories
            else None
        )

    def _validate_filters(self) -> None:
        """少なくとも1つのフィルターが提供されていることを検証します。

        Raises:
            ServiceInitializationError: フィルターが提供されていない場合。

        """
        if not self.agent_id and not self.user_id and not self.application_id and not self.thread_id:
            raise ServiceInitializationError(
                "At least one of the filters: agent_id, user_id, application_id, or thread_id is required."
            )

    def _validate_per_operation_thread_id(self, thread_id: str | None) -> None:
        """スコープされている場合に新しいスレッドIDが既存のものと競合しないことを検証します。

        Args:
            thread_id: 新しいスレッドID、またはNone。

        Raises:
            ValueError: 既にスレッドIDが存在する場合に新しいスレッドIDが提供された場合。

        """
        if (
            self.scope_to_per_operation_thread_id
            and thread_id
            and self._per_operation_thread_id
            and thread_id != self._per_operation_thread_id
        ):
            raise ValueError(
                "Mem0Provider can only be used with one thread at a time when scope_to_per_operation_thread_id is True."
            )
