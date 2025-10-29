# 著作権 (c) Microsoft。無断転載を禁じます。

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from pydantic import Field

"""
Azure OpenAI Responses Client with Thread Management Example

This sample demonstrates thread management with Azure OpenAI Responses Client, comparing
automatic thread creation with explicit thread management for persistent context.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_automatic_thread_creation() -> None:
    """自動スレッド作成の例です。"""
    print("=== Automatic Thread Creation Example ===")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 最初の会話 - スレッドが提供されていないため自動的に作成されます
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1.text}")

    # 2番目の会話 - まだスレッドが提供されていないため別の新しいスレッドを作成します
    query2 = "What was the last city I asked about?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2.text}")
    print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence_in_memory() -> None:
    """
    複数の会話にわたるスレッドの永続性を示す例です。
    この例では、メッセージはメモリ内に保存されます。

    """
    print("=== Thread Persistence Example (In-Memory) ===")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
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

    # 3番目の会話 - Agentは前の両方の都市を覚えているはずです
    query3 = "Which of the cities I asked about has better weather?"
    print(f"\nUser: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3.text}")
    print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_id() -> None:
    """
    サービスからの既存のスレッドIDを使う方法の例です。
    この例では、メッセージはAzure OpenAI conversation stateを使ってサーバー上に保存されます。

    """
    print("=== Existing Thread ID Example ===")

    # まず会話を作成し、スレッドIDを取得します
    existing_thread_id = None

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # 会話を開始しスレッドIDを取得します
    thread = agent.get_new_thread()

    query1 = "What's the weather in Paris?"
    print(f"User: {query1}")
    # `store`パラメーターをTrueに設定してAzure OpenAI conversation stateを有効にします
    result1 = await agent.run(query1, thread=thread, store=True)
    print(f"Agent: {result1.text}")

    # 最初のレスポンス後にスレッドIDが設定されます
    existing_thread_id = thread.service_thread_id
    print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        agent = ChatAgent(
            chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        )

        # 既存のIDでスレッドを作成します
        thread = AgentThread(service_thread_id=existing_thread_id)

        query2 = "What was the last city I asked about?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, thread=thread, store=True)
        print(f"Agent: {result2.text}")
        print("Note: The agent continues the conversation from the previous thread by using thread ID.\n")


async def main() -> None:
    print("=== Azure OpenAI Response Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence_in_memory()
    await example_with_existing_thread_id()


if __name__ == "__main__":
    asyncio.run(main())
