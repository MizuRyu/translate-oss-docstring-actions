# 著作権 (c) Microsoft。無断転載を禁じます。

import asyncio
from datetime import datetime, timezone
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from pydantic import Field

"""
Azure OpenAI Responses Client with Function Tools Example

This sample demonstrates function tool integration with Azure OpenAI Responses Client,
showing both agent-level and query-level tool configuration patterns.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


def get_time() -> str:
    """現在のUTC時間を取得します。"""
    current_time = datetime.now(timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."


async def tools_on_agent_level() -> None:
    """Agent作成時に定義されたツールの例です。"""
    print("=== Tools Defined on Agent Level ===")

    # Agent作成時にツールが提供されます Agentはその生涯の間に任意のクエリでこれらのツールを使用できます 認証には、ターミナルで`az
    # login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant that can provide weather and time information.",
        tools=[get_weather, get_time],  # Tools defined at agent creation
    )

    # 最初のクエリ - Agentはweatherツールを使用できます
    query1 = "What's the weather like in New York?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1}\n")

    # 2番目のクエリ - Agentはtimeツールを使用できます
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2}\n")

    # 3番目のクエリ - 必要に応じて両方のツールを使用できます
    query3 = "What's the weather in London and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3)
    print(f"Agent: {result3}\n")


async def tools_on_run_level() -> None:
    """runメソッドに渡されたツールの例です。"""
    print("=== Tools Passed to Run Method ===")

    # ツールなしで作成されたAgent 認証には、ターミナルで`az
    # login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant.",
        # ここではツールは定義されていません
    )

    # weatherツールを使った最初のクエリ
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, tools=[get_weather])  # runメソッドに渡されたツール
    print(f"Agent: {result1}\n")

    # timeツールを使った2番目のクエリ
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, tools=[get_time])  # このクエリには異なるツール
    print(f"Agent: {result2}\n")

    # 複数のツールを使った3番目のクエリ
    query3 = "What's the weather in Chicago and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3, tools=[get_weather, get_time])  # 複数のツール
    print(f"Agent: {result3}\n")


async def mixed_tools_example() -> None:
    """Agentレベルのツールとrunメソッドのツールの両方を示す例です。"""
    print("=== Mixed Tools Example (Agent + Run Method) ===")

    # いくつかの基本ツールで作成されたAgent 認証には、ターミナルで`az
    # login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a comprehensive assistant that can help with various information requests.",
        tools=[get_weather],  # Base tool available for all queries
    )

    # Agentツールと追加のrunメソッドツールの両方を使ったクエリ
    query = "What's the weather in Denver and what's the current UTC time?"
    print(f"User: {query}")

    # Agentは作成時のget_weatherとrunメソッドからの追加ツールにアクセスできます
    result = await agent.run(
        query,
        tools=[get_time],  # Additional tools for this specific query
    )
    print(f"Agent: {result}\n")


async def main() -> None:
    print("=== Azure OpenAI Responses Client Agent with Function Tools Examples ===\n")

    await tools_on_agent_level()
    await tools_on_run_level()
    await mixed_tools_example()


if __name__ == "__main__":
    asyncio.run(main())
