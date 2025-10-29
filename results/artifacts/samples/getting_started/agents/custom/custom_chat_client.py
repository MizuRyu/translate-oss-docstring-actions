# Copyright (c) Microsoft. All rights reserved.

import asyncio
import random
from collections.abc import AsyncIterable, MutableSequence
from typing import Any, ClassVar

from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Role,
    TextContent,
    use_chat_middleware,
    use_function_invocation,
)

"""
Custom Chat Client Implementation Example

This sample demonstrates implementing a custom chat client by extending BaseChatClient class,
showing integration with ChatAgent and both streaming and non-streaming responses.
"""


@use_function_invocation
@use_chat_middleware
class EchoingChatClient(BaseChatClient):
    """メッセージを修正してエコーバックするカスタムチャットクライアント。

    これはBaseChatClientを拡張し、必要な_inner_get_response()と_inner_get_streaming_response()メソッドを実装することで
    カスタムチャットクライアントを実装する方法を示す。
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "EchoingChatClient"

    def __init__(self, *, prefix: str = "Echo:", **kwargs: Any) -> None:
        """EchoingChatClientを初期化する。

        Args:
            prefix: エコーバックするメッセージに追加するプレフィックス。
            **kwargs: BaseChatClientに渡される追加のキーワード引数。

        """
        super().__init__(**kwargs)
        self.prefix = prefix

    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        """ユーザーのメッセージをプレフィックス付きでエコーバックする。"""
        if not messages:
            response_text = "No messages to echo!"
        else:
            # 最後のユーザーメッセージをエコーする
            last_user_message = None
            for message in reversed(messages):
                if message.role == Role.USER:
                    last_user_message = message
                    break

            if last_user_message and last_user_message.text:
                response_text = f"{self.prefix} {last_user_message.text}"
            else:
                response_text = f"{self.prefix} [No text message found]"

        response_message = ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=response_text)])

        return ChatResponse(
            messages=[response_message],
            model_id="echo-model-v1",
            response_id=f"echo-resp-{random.randint(1000, 9999)}",
        )

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """エコーしたメッセージを文字ごとにストリーミングで返す。"""
        # 最初に完全なレスポンスを取得する
        response = await self._inner_get_response(messages=messages, chat_options=chat_options, **kwargs)

        if response.messages:
            response_text = response.messages[0].text or ""

            # 文字ごとにストリーミングする
            for char in response_text:
                yield ChatResponseUpdate(
                    contents=[TextContent(text=char)],
                    role=Role.ASSISTANT,
                    response_id=f"echo-stream-resp-{random.randint(1000, 9999)}",
                    model_id="echo-model-v1",
                )
                await asyncio.sleep(0.05)


async def main() -> None:
    """ChatAgentを使ったカスタムチャットクライアントの実装と使用例を示す。"""
    print("=== Custom Chat Client Example ===\n")

    # カスタムチャットクライアントを作成する
    print("--- EchoingChatClient Example ---")

    echo_client = EchoingChatClient(prefix="🔊 Echo:")

    # チャットクライアントを直接使用する
    print("Using chat client directly:")
    direct_response = await echo_client.get_response("Hello, custom chat client!")
    print(f"Direct response: {direct_response.messages[0].text}")

    # カスタムチャットクライアントを使ってAgentを作成する
    echo_agent = echo_client.create_agent(
        name="EchoAgent",
        instructions="You are a helpful assistant that echoes back what users say.",
    )

    print(f"\nAgent Name: {echo_agent.name}")
    print(f"Agent Display Name: {echo_agent.display_name}")

    # Agentで非ストリーミングをテストする
    query = "This is a test message"
    print(f"\nUser: {query}")
    result = await echo_agent.run(query)
    print(f"Agent: {result.messages[0].text}")

    # Agentでストリーミングをテストする
    query2 = "Stream this message back to me"
    print(f"\nUser: {query2}")
    print("Agent: ", end="", flush=True)
    async for chunk in echo_agent.run_stream(query2):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()

    # Threadと会話履歴を使った例
    print("\n--- Using Custom Chat Client with Thread ---")

    thread = echo_agent.get_new_thread()

    # 会話内の複数メッセージ
    messages = [
        "Hello, I'm starting a conversation",
        "How are you doing?",
        "Thanks for chatting!",
    ]

    for msg in messages:
        result = await echo_agent.run(msg, thread=thread)
        print(f"User: {msg}")
        print(f"Agent: {result.messages[0].text}\n")

    # 会話履歴を確認する
    if thread.message_store:
        thread_messages = await thread.message_store.list_messages()
        print(f"Thread contains {len(thread_messages)} messages")
    else:
        print("Thread has no message store configured")


if __name__ == "__main__":
    asyncio.run(main())
