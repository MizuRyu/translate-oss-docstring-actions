# 著作権 (c) Microsoft。無断転載を禁じます。

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Role,
    TextContent,
)

"""
Custom Agent Implementation Example

This sample demonstrates implementing a custom agent by extending BaseAgent class,
showing the minimal requirements for both streaming and non-streaming responses.
"""


class EchoAgent(BaseAgent):
    """ユーザーメッセージをプレフィックス付きでエコーするシンプルなカスタムAgentです。

    これはBaseAgentを拡張し、必要なrun()とrun_stream()メソッドを実装することで完全なカスタムAgentを作成する方法を示します。

    """

    echo_prefix: str = "Echo: "

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        echo_prefix: str = "Echo: ",
        **kwargs: Any,
    ) -> None:
        """EchoAgentを初期化します。

        Args:
            name: Agentの名前。
            description: Agentの説明。
            echo_prefix: エコーメッセージに追加するプレフィックス。
            **kwargs: BaseAgentに渡される追加のキーワード引数。

        """
        super().__init__(
            name=name,
            description=description,
            echo_prefix=echo_prefix,  # type: ignore
            **kwargs,
        )

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Agentを実行し、完全なレスポンスを返します。

        Args:
            messages: 処理するメッセージ。
            thread: 会話スレッド（Optional）。
            **kwargs: 追加のキーワード引数。

        Returns:
            AgentRunResponseでAgentの返信を含みます。

        """
        # 入力メッセージをリストに正規化します
        normalized_messages = self._normalize_messages(messages)

        if not normalized_messages:
            response_message = ChatMessage(
                role=Role.ASSISTANT,
                contents=[TextContent(text="Hello! I'm a custom echo agent. Send me a message and I'll echo it back.")],
            )
        else:
            # 簡単のため、最後のユーザーメッセージをエコーします
            last_message = normalized_messages[-1]
            if last_message.text:
                echo_text = f"{self.echo_prefix}{last_message.text}"
            else:
                echo_text = f"{self.echo_prefix}[Non-text message received]"

            response_message = ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=echo_text)])

        # スレッドが提供されていれば新しいメッセージを通知します
        if thread is not None:
            await self._notify_thread_of_new_messages(thread, normalized_messages, response_message)

        return AgentRunResponse(messages=[response_message])

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Agentを実行し、ストリーミングレスポンスの更新をyieldします。

        Args:
            messages: 処理するメッセージ。
            thread: 会話スレッド（Optional）。
            **kwargs: 追加のキーワード引数。

        Yields:
            AgentRunResponseUpdateオブジェクトを含み、レスポンスのチャンクを返します。

        """
        # 入力メッセージをリストに正規化します
        normalized_messages = self._normalize_messages(messages)

        if not normalized_messages:
            response_text = "Hello! I'm a custom echo agent. Send me a message and I'll echo it back."
        else:
            # 簡単のため、最後のユーザーメッセージをエコーします
            last_message = normalized_messages[-1]
            if last_message.text:
                response_text = f"{self.echo_prefix}{last_message.text}"
            else:
                response_text = f"{self.echo_prefix}[Non-text message received]"

        # レスポンスを単語ごとにyieldしてストリーミングをシミュレートする
        words = response_text.split()
        for i, word in enumerate(words):
            # 最初の単語を除き、単語の前にスペースを追加する
            chunk_text = f" {word}" if i > 0 else word

            yield AgentRunResponseUpdate(
                contents=[TextContent(text=chunk_text)],
                role=Role.ASSISTANT,
            )

            # ストリーミングをシミュレートするための小さな遅延
            await asyncio.sleep(0.1)

        # 提供されていれば、完全なレスポンスをThreadに通知する
        if thread is not None:
            complete_response = ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=response_text)])
            await self._notify_thread_of_new_messages(thread, normalized_messages, complete_response)


async def main() -> None:
    """カスタムEchoAgentの使い方を示す。"""
    print("=== Custom Agent Example ===\n")

    # EchoAgentを作成する
    print("--- EchoAgent Example ---")
    echo_agent = EchoAgent(
        name="EchoBot", description="A simple agent that echoes messages with a prefix", echo_prefix="🔊 Echo: "
    )

    # 非ストリーミングのテスト
    print(f"Agent Name: {echo_agent.name}")
    print(f"Agent ID: {echo_agent.id}")
    print(f"Display Name: {echo_agent.display_name}")

    query = "Hello, custom agent!"
    print(f"\nUser: {query}")
    result = await echo_agent.run(query)
    print(f"Agent: {result.messages[0].text}")

    # ストリーミングのテスト
    query2 = "This is a streaming test"
    print(f"\nUser: {query2}")
    print("Agent: ", end="", flush=True)
    async for chunk in echo_agent.run_stream(query2):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()

    # Threadを使った例
    print("\n--- Using Custom Agent with Thread ---")
    thread = echo_agent.get_new_thread()

    # 最初のメッセージ
    result1 = await echo_agent.run("First message", thread=thread)
    print("User: First message")
    print(f"Agent: {result1.messages[0].text}")

    # 同じThread内の2番目のメッセージ
    result2 = await echo_agent.run("Second message", thread=thread)
    print("User: Second message")
    print(f"Agent: {result2.messages[0].text}")

    # 会話履歴を確認する
    if thread.message_store:
        messages = await thread.message_store.list_messages()
        print(f"\nThread contains {len(messages)} messages in history")
    else:
        print("\nThread has no message store configured")


if __name__ == "__main__":
    asyncio.run(main())
