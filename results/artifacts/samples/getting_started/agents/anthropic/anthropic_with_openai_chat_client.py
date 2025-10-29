# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIChatClient

"""
Anthropic with OpenAI Chat Client Example

This sample demonstrates using Anthropic models through OpenAI Chat Client by
configuring the base URL to point to Anthropic's API for cross-provider compatibility.
"""


def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def non_streaming_example() -> None:
    """非ストリーミングレスポンスの例（一度に完全な結果を取得）。"""
    print("=== Non-streaming Response Example ===")

    agent = OpenAIChatClient(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
        model_id=os.getenv("ANTHROPIC_MODEL"),
    ).create_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Seattle?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Result: {result}\n")


async def streaming_example() -> None:
    """ストリーミングレスポンスの例（生成される結果を逐次取得）。"""
    print("=== Streaming Response Example ===")

    agent = OpenAIChatClient(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
        model_id=os.getenv("ANTHROPIC_MODEL"),
    ).create_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Portland?"
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream(query):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


async def main() -> None:
    print("=== Anthropic with OpenAI Chat Client Agent Example ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
