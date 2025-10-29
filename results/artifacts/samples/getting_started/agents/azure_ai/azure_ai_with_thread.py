# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import AgentThread, ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Azure AI Agent with Thread Management Example

This sample demonstrates thread management with Azure AI Agents, comparing
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

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent,
    ):
        # 最初の会話 - スレッドが提供されていないため自動的に作成されます。
        first_query = "What's the weather like in Seattle?"
        print(f"User: {first_query}")
        first_result = await agent.run(first_query)
        print(f"Agent: {first_result.text}")

        # 2番目の会話 - まだスレッドが提供されていないため別の新しいスレッドが作成されます。
        second_query = "What was the last city I asked about?"
        print(f"\nUser: {second_query}")
        second_result = await agent.run(second_query)
        print(f"Agent: {second_result.text}")
        print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence() -> None:
    """複数の会話にわたるスレッドの永続性を示す例です。"""
    print("=== Thread Persistence Example ===")
    print("Using the same thread across multiple conversations to maintain context.\n")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent,
    ):
        # 再利用される新しいスレッドを作成します。
        thread = agent.get_new_thread()

        # 最初の会話。
        first_query = "What's the weather like in Tokyo?"
        print(f"User: {first_query}")
        first_result = await agent.run(first_query, thread=thread)
        print(f"Agent: {first_result.text}")

        # 同じスレッドを使った2番目の会話 - コンテキストを維持します。
        second_query = "How about London?"
        print(f"\nUser: {second_query}")
        second_result = await agent.run(second_query, thread=thread)
        print(f"Agent: {second_result.text}")

        # 3番目の会話 - Agentは前の両方の都市を覚えているはずです。
        third_query = "Which of the cities I asked about has better weather?"
        print(f"\nUser: {third_query}")
        third_result = await agent.run(third_query, thread=thread)
        print(f"Agent: {third_result.text}")
        print("Note: The agent remembers context from previous messages in the same thread.\n")


async def example_with_existing_thread_id() -> None:
    """サービスからの既存のスレッドIDを使う方法を示す例です。"""
    print("=== Existing Thread ID Example ===")
    print("Using a specific thread ID to continue an existing conversation.\n")

    # まず会話を作成し、スレッドIDを取得します。
    existing_thread_id = None

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent,
    ):
        # 会話を開始しスレッドIDを取得します。
        thread = agent.get_new_thread()
        first_query = "What's the weather in Paris?"
        print(f"User: {first_query}")
        first_result = await agent.run(first_query, thread=thread)
        print(f"Agent: {first_result.text}")

        # スレッドIDは最初のレスポンス後に設定されます。
        existing_thread_id = thread.service_thread_id
        print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        # 新しいAgentインスタンスを作成しますが既存のスレッドIDを使用します。
        async with (
            AzureCliCredential() as credential,
            ChatAgent(
                chat_client=AzureAIAgentClient(thread_id=existing_thread_id, async_credential=credential),
                instructions="You are a helpful weather agent.",
                tools=get_weather,
            ) as agent,
        ):
            # 既存のIDでスレッドを作成します。
            thread = AgentThread(service_thread_id=existing_thread_id)

            second_query = "What was the last city I asked about?"
            print(f"User: {second_query}")
            second_result = await agent.run(second_query, thread=thread)
            print(f"Agent: {second_result.text}")
            print("Note: The agent continues the conversation from the previous thread.\n")


async def main() -> None:
    print("=== Azure AI Chat Client Agent Thread Management Examples ===\n")

    await example_with_automatic_thread_creation()
    await example_with_thread_persistence()
    await example_with_existing_thread_id()


if __name__ == "__main__":
    asyncio.run(main())
