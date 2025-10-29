# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent
from agent_framework.openai import OpenAIAssistantsClient
from pydantic import Field

"""
OpenAI Assistants with Thread Management Example

This sample demonstrates thread management with OpenAI Assistants, showing
persistent conversation threads and context preservation across interactions.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """自動Thread作成（サービス管理Thread）の例。"""
    print("=== Automatic Thread Creation Example ===")

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        # 最初の会話 - Threadが提供されていないため自動作成される
        query1 = "What's the weather like in Seattle?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"Agent: {result1.text}")

        # 2番目の会話 - まだThreadが提供されていないため別の新しいThreadを作成する
        query2 = "What was the last city I asked about?"
        print(f"\nUser: {query2}")
        result2 = await agent.run(query2)
        print(f"Agent: {result2.text}")
        print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence() -> None:
    """複数会話にわたるThreadの永続性を示す例。"""
    print("=== Thread Persistence Example ===")
    print("Using the same thread across multiple conversations to maintain context.\n")

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        # 再利用される新しいThreadを作成する
        thread = agent.get_new_thread()

        # 最初の会話
        query1 = "What's the weather like in Tokyo?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, thread=thread)
        print(f"Agent: {result1.text}")

        # 同じThreadを使った2番目の会話 - コンテキストを維持する
        query2 = "How about London?"
        print(f"\nUser: {query2}")
        result2 = await agent.run(query2, thread=thread)
        print(f"Agent: {result2.text}")

        # 3番目の会話 - Agentは前の両方の都市を覚えているはず
        query3 = "Which of the cities I asked about has better weather?"
        print(f"\nUser: {query3}")
        result3 = await agent.run(query3, thread=thread)
        print(f"Agent: {result3.text}")
        print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_id() -> None:
    """サービスからの既存Thread IDを使う方法を示す例。"""
    print("=== Existing Thread ID Example ===")
    print("Using a specific thread ID to continue an existing conversation.\n")

    # まず会話を作成しThread IDを取得する
    existing_thread_id = None

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    ) as agent:
        # 会話を開始しThread IDを取得する
        thread = agent.get_new_thread()
        query1 = "What's the weather in Paris?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, thread=thread)
        print(f"Agent: {result1.text}")

        # 最初のレスポンス後にThread IDが設定される
        existing_thread_id = thread.service_thread_id
        print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        # 新しいAgentインスタンスを作成するが既存のThread IDを使う
        async with ChatAgent(
            chat_client=OpenAIAssistantsClient(thread_id=existing_thread_id),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent:
            # 既存IDでThreadを作成する
            thread = AgentThread(service_thread_id=existing_thread_id)

            query2 = "What was the last city I asked about?"
            print(f"User: {query2}")
            result2 = await agent.run(query2, thread=thread)
            print(f"Agent: {result2.text}")
            print("Note: The agent continues the conversation from the previous thread.\n")


async def main() -> None:
    print("=== OpenAI Assistants Chat Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence()
    await example_with_existing_thread_id()


if __name__ == "__main__":
    asyncio.run(main())
