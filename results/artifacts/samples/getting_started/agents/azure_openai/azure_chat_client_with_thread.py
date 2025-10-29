# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent, ChatMessageStore
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from pydantic import Field

"""
Azure OpenAI Chat Client with Thread Management Example

This sample demonstrates thread management with Azure OpenAI Chat Client, comparing
automatic thread creation with explicit thread management for persistent context.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """自動スレッド作成（サービス管理スレッド）を示す例です。"""
    print("=== Automatic Thread Creation Example ===")

    # 認証のために、ターミナルで `az login` コマンドを実行するか、AzureCliCredential を好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
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


async def example_with_thread_persistence() -> None:
    """複数の会話にわたるスレッドの持続性を示す例です。"""
    print("=== Thread Persistence Example ===")
    print("Using the same thread across multiple conversations to maintain context.\n")

    # 認証のために、ターミナルで `az login` コマンドを実行するか、AzureCliCredential を好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 再利用される新しいスレッドを作成します
    thread = agent.get_new_thread()

    # 最初の会話
    query1 = "What's the weather like in Tokyo?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # 同じスレッドを使った2番目の会話 - コンテキストを維持します
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


async def example_with_existing_thread_messages() -> None:
    """Azureの既存のスレッドメッセージを扱う方法を示す例です。"""
    print("=== Existing Thread Messages Example ===")

    # 認証のために、ターミナルで `az login` コマンドを実行するか、AzureCliCredential を好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 会話を開始してメッセージ履歴を構築します
    thread = agent.get_new_thread()

    query1 = "What's the weather in Paris?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # スレッドには現在、メモリ内に会話履歴が含まれています
    if thread.message_store:
        messages = await thread.message_store.list_messages()
        print(f"Thread contains {len(messages or [])} messages")

    print("\n--- Continuing with the same thread in a new agent instance ---")

    # 新しいエージェントインスタンスを作成しますが、既存のスレッドとそのメッセージ履歴を使用します
    new_agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 会話履歴を含む同じスレッドオブジェクトを使用します
    query2 = "What was the last city I asked about?"
    print(f"User: {query2}")
    result2 = await new_agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")
    print("Note: The agent continues the conversation using the local message history.\n")

    print("\n--- Alternative: Creating a new thread from existing messages ---")

    # 既存のメッセージから新しいスレッドを作成することもできます
    messages = await thread.message_store.list_messages() if thread.message_store else []
    new_thread = AgentThread(message_store=ChatMessageStore(messages))

    query3 = "How does the Paris weather compare to London?"
    print(f"User: {query3}")
    result3 = await new_agent.run(query3, thread=new_thread)
    print(f"Agent: {result3.text}")
    print("Note: This creates a new thread with the same conversation history.\n")


async def main() -> None:
    print("=== Azure Chat Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence()
    await example_with_existing_thread_messages()


if __name__ == "__main__":
    asyncio.run(main())
