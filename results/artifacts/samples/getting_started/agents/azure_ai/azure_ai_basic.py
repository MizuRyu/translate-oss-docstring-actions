# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Azure AI Agent Basic Example

This sample demonstrates basic usage of AzureAIAgentClient to create agents with automatic
lifecycle management. Shows both streaming and non-streaming responses with function tools.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def non_streaming_example() -> None:
    """非ストリーミングレスポンスの例（一度に完全な結果を取得）。"""
    print("=== Non-streaming Response Example ===")

    # Agent ID が指定されていないため、エージェントは自動的に作成され、レスポンス取得後に削除されます。 認証には、ターミナルで `az login`
    # コマンドを実行するか、AzureCliCredential を好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent,
    ):
        query = "What's the weather like in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}\n")


async def streaming_example() -> None:
    """ストリーミングレスポンスの例（生成される結果を逐次取得）。"""
    print("=== Streaming Response Example ===")

    # Agent ID が指定されていないため、エージェントは自動的に作成され、レスポンス取得後に削除されます。 認証には、ターミナルで `az login`
    # コマンドを実行するか、AzureCliCredential を好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        ) as agent,
    ):
        query = "What's the weather like in Portland?"
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n")


async def main() -> None:
    print("=== Basic Azure AI Chat Client Agent Example ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
