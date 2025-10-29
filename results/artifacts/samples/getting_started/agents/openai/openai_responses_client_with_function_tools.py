# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import datetime, timezone
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client with Function Tools Example

This sample demonstrates function tool integration with OpenAI Responses Client,
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
    """Agent作成時にToolsを定義する例。"""
    print("=== Tools Defined on Agent Level ===")

    # Agent作成時にToolsが提供されます Agentはその生涯の間に任意のクエリでこれらのToolsを使用できます
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant that can provide weather and time information.",
        tools=[get_weather, get_time],  # Tools defined at agent creation
    )

    # 最初のクエリ - Agentはweather toolを使用できます
    query1 = "What's the weather like in New York?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1}\n")

    # 2番目のクエリ - Agentはtime toolを使用できます
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2}\n")

    # 3番目のクエリ - Agentは必要に応じて両方のToolsを使用できます
    query3 = "What's the weather in London and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3)
    print(f"Agent: {result3}\n")


async def tools_on_run_level() -> None:
    """runメソッドに渡されたToolsを示す例。"""
    print("=== Tools Passed to Run Method ===")

    # Toolsなしで作成されたAgent
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful assistant.",
        # ここではToolsは定義されていません
    )

    # weather toolを使った最初のクエリ
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, tools=[get_weather])  # runメソッドに渡されたTool
    print(f"Agent: {result1}\n")

    # time toolを使った2番目のクエリ
    query2 = "What's the current UTC time?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, tools=[get_time])  # このクエリには異なるTool
    print(f"Agent: {result2}\n")

    # 複数のToolsを使った3番目のクエリ
    query3 = "What's the weather in Chicago and what's the current UTC time?"
    print(f"User: {query3}")
    result3 = await agent.run(query3, tools=[get_weather, get_time])  # 複数のTools
    print(f"Agent: {result3}\n")


async def mixed_tools_example() -> None:
    """AgentレベルのToolsとrunメソッドのToolsの両方を示す例。"""
    print("=== Mixed Tools Example (Agent + Run Method) ===")

    # いくつかの基本的なToolsで作成されたAgent
    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a comprehensive assistant that can help with various information requests.",
        tools=[get_weather],  # Base tool available for all queries
    )

    # Agentツールと追加のrunメソッドツールの両方を使ったクエリ
    query = "What's the weather in Denver and what's the current UTC time?"
    print(f"User: {query}")

    # Agentは作成時からget_weatherにアクセスでき、runメソッドから追加のToolsも利用可能
    result = await agent.run(
        query,
        tools=[get_time],  # Additional tools for this specific query
    )
    print(f"Agent: {result}\n")


async def main() -> None:
    print("=== OpenAI Responses Client Agent with Function Tools Examples ===\n")

    await tools_on_agent_level()
    await tools_on_run_level()
    await mixed_tools_example()


if __name__ == "__main__":
    asyncio.run(main())
