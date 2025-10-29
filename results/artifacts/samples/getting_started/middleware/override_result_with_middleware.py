# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from random import randint
from typing import Annotated

from agent_framework import (
    AgentRunContext,
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatMessage,
    Role,
    TextContent,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Result Override with Middleware (Regular and Streaming)

This sample demonstrates how to use middleware to intercept and modify function results
after execution, supporting both regular and streaming agent responses. The example shows:

- How to execute the original function first and then modify its result
- Replacing function outputs with custom messages or transformed data
- Using middleware for result filtering, formatting, or enhancement
- Detecting streaming vs non-streaming execution using context.is_streaming
- Overriding streaming results with custom async generators

The weather override middleware lets the original weather function execute normally,
then replaces its result with a custom "perfect weather" message. For streaming responses,
it creates a custom async generator that yields the override message in chunks.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def weather_override_middleware(
    context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
) -> None:
    """ストリーミングおよび非ストリーミングの両方のケースで天気の結果を上書きするミドルウェア。"""

    # 元のAgentの実行を最初に完了させる
    await next(context)

    # 上書きする結果があるか確認する（Agentが天気関数を呼び出したか）
    if context.result is not None:
        # カスタムの天気メッセージを作成する
        chunks = [
            "Weather Advisory - ",
            "due to special atmospheric conditions, ",
            "all locations are experiencing perfect weather today! ",
            "Temperature is a comfortable 22°C with gentle breezes. ",
            "Perfect day for outdoor activities!",
        ]

        if context.is_streaming:
            # ストリーミングの場合：チャンクをyieldする非同期ジェネレーターを作成する
            async def override_stream() -> AsyncIterable[AgentRunResponseUpdate]:
                for chunk in chunks:
                    yield AgentRunResponseUpdate(contents=[TextContent(text=chunk)])

            context.result = override_stream()
        else:
            # 非ストリーミングの場合：文字列メッセージに置き換えるだけ
            custom_message = "".join(chunks)
            context.result = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=custom_message)])


async def main() -> None:
    """ストリーミングおよび非ストリーミングの両方に対するミドルウェアによる結果上書きを示す例。"""
    print("=== Result Override Middleware Example ===")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather assistant. Use the weather tool to get current conditions.",
            tools=get_weather,
            middleware=weather_override_middleware,
        ) as agent,
    ):
        # 非ストリーミングの例
        print("\n--- Non-streaming Example ---")
        query = "What's the weather like in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}")

        # ストリーミングの例
        print("\n--- Streaming Example ---")
        query = "What's the weather like in Portland?"
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        async for chunk in agent.run_stream(query):
            if chunk.text:
                print(chunk.text, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
