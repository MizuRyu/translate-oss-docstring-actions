# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client with Thread Management Example

This sample demonstrates thread management with OpenAI Responses Client, showing
persistent conversation context and simplified response handling.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """自動スレッド作成を示す例。"""
    print("=== Automatic Thread Creation Example ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 最初の会話 - スレッドが提供されていないため、自動的に作成されます
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1.text}")

    # 2番目の会話 - まだスレッドが提供されていないため、別の新しいスレッドを作成します
    query2 = "What was the last city I asked about?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2.text}")
    print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence_in_memory() -> None:
    """
    複数の会話にわたるスレッドの永続性を示す例。
    この例では、メッセージはメモリ内に保存されます。

    """
    print("=== Thread Persistence Example (In-Memory) ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 再利用される新しいスレッドを作成する
    thread = agent.get_new_thread()

    # 最初の会話
    query1 = "What's the weather like in Tokyo?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # 同じスレッドを使用した2番目の会話 - コンテキストを維持する
    query2 = "How about London?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")

    # 3番目の会話 - エージェントは前の2つの都市を覚えているはずです
    query3 = "Which of the cities I asked about has better weather?"
    print(f"\nUser: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3.text}")
    print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_id() -> None:
    """
    サービスからの既存のスレッドIDを使用する方法を示す例。
    この例では、メッセージはOpenAI conversation stateを使用してサーバー上に保存されます。

    """
    print("=== Existing Thread ID Example ===")

    # 最初に会話を作成し、スレッドIDを取得する
    existing_thread_id = None

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 会話を開始してスレッドIDを取得する
    thread = agent.get_new_thread()

    query1 = "What's the weather in Paris?"
    print(f"User: {query1}")
    # `store`パラメータをTrueに設定してOpenAI conversation stateを有効にする
    result1 = await agent.run(query1, thread=thread, store=True)
    print(f"Agent: {result1.text}")

    # 最初のレスポンス後にスレッドIDが設定される
    existing_thread_id = thread.service_thread_id
    print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        agent = ChatAgent(
            chat_client=OpenAIResponsesClient(),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        )

        # 既存のIDでスレッドを作成する
        thread = AgentThread(service_thread_id=existing_thread_id)

        query2 = "What was the last city I asked about?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, thread=thread, store=True)
        print(f"Agent: {result2.text}")
        print("Note: The agent continues the conversation from the previous thread by using thread ID.\n")


async def main() -> None:
    print("=== OpenAI Response Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence_in_memory()
    await example_with_existing_thread_id()


if __name__ == "__main__":
    asyncio.run(main())
