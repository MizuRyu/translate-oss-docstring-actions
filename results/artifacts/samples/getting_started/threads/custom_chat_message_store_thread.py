# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Collection
from typing import Any

from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel


class CustomStoreState(BaseModel):
    """カスタムchat message store stateの実装。"""

    messages: list[ChatMessage]


class CustomChatMessageStore(ChatMessageStoreProtocol):
    """カスタムchat message storeの実装。
    実際のアプリケーションでは、リレーショナルデータベースやベクターストアの実装となる場合があります。"""

    def __init__(self, messages: Collection[ChatMessage] | None = None) -> None:
        self._messages: list[ChatMessage] = []
        if messages:
            self._messages.extend(messages)

    async def add_messages(self, messages: Collection[ChatMessage]) -> None:
        self._messages.extend(messages)

    async def list_messages(self) -> list[ChatMessage]:
        return self._messages

    async def deserialize_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        if serialized_store_state:
            state = CustomStoreState.model_validate(serialized_store_state, **kwargs)
            if state.messages:
                self._messages.extend(state.messages)

    async def serialize_state(self, **kwargs: Any) -> Any:
        state = CustomStoreState(messages=self._messages)
        return state.model_dump(**kwargs)


async def main() -> None:
    """スレッド用にサードパーティまたはカスタムchat message storeを使用する方法を示します。"""
    print("=== Thread with 3rd party or custom chat message store ===")

    # ここではOpenAI Chat Clientを例として使用していますが、他のchat clientも使用可能です。
    agent = OpenAIChatClient().create_agent(
        name="Joker",
        instructions="You are good at telling jokes.",
        # カスタムchat message storeを使用します。 指定しない場合はデフォルトのインメモリストアが使用されます。
        chat_message_store_factory=CustomChatMessageStore,
    )

    # agent会話のために新しいスレッドを開始します。
    thread = agent.get_new_thread()

    # ユーザー入力に応答します。
    query = "Tell me a joke about a pirate."
    print(f"User: {query}")
    print(f"Agent: {await agent.run(query, thread=thread)}\n")

    # 後で使用できるようにスレッドのstateをシリアライズします。
    serialized_thread = await thread.serialize()

    # スレッドはデータベース、ファイル、その他のストレージメカニズムに保存され、後で再度読み込むことができます。
    print(f"Serialized thread: {serialized_thread}\n")

    # ストレージから読み込んだ後にスレッドstateをデシリアライズします。
    resumed_thread = await agent.deserialize_thread(serialized_thread)

    # ユーザー入力に応答します。
    query = "Now tell the same joke in the voice of a pirate, and add some emojis to the joke."
    print(f"User: {query}")
    print(f"Agent: {await agent.run(query, thread=resumed_thread)}\n")


if __name__ == "__main__":
    asyncio.run(main())
