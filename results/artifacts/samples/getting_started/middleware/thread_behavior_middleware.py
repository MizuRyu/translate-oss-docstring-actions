# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

from agent_framework import (
    AgentRunContext,
    ChatMessageStore,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from pydantic import Field

"""
Thread Behavior Middleware Example

This sample demonstrates how middleware can access and track thread state across multiple agent runs.
The example shows:

- How AgentRunContext.thread property behaves across multiple runs
- How middleware can access conversation history through the thread
- The timing of when thread messages are populated (before vs after next() call)
- How to track thread state changes across runs

Key behaviors demonstrated:
1. First run: context.messages is populated, context.thread is initially empty (before next())
2. After next(): thread contains input message + response from agent
3. Second run: context.messages contains only current input, thread contains previous history
4. After next(): thread contains full conversation history (all previous + current messages)
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    from random import randint

    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def thread_tracking_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    """複数回の実行にわたるスレッドの動作を追跡・ログするミドルウェア。"""
    thread_messages = []
    if context.thread and context.thread.message_store:
        thread_messages = await context.thread.message_store.list_messages()

    print(f"[Middleware pre-execution] Current input messages: {len(context.messages)}")
    print(f"[Middleware pre-execution] Thread history messages: {len(thread_messages)}")

    # Agentを実行するために次を呼び出す
    await next(context)

    # Agent実行後のスレッド状態を確認する
    updated_thread_messages = []
    if context.thread and context.thread.message_store:
        updated_thread_messages = await context.thread.message_store.list_messages()

    print(f"[Middleware post-execution] Updated thread messages: {len(updated_thread_messages)}")


async def main() -> None:
    """複数回の実行にわたるミドルウェア内のスレッド動作を示す例。"""
    print("=== Thread Behavior Middleware Example ===")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather assistant.",
        tools=get_weather,
        middleware=thread_tracking_middleware,
        # 会話履歴を永続化するためのメッセージストアファクトリでAgentを構成する
        chat_message_store_factory=ChatMessageStore,
    )

    # 実行間でメッセージを永続化するスレッドを作成する
    thread = agent.get_new_thread()

    print("\nFirst Run:")
    query1 = "What's the weather like in Tokyo?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    print("\nSecond Run:")
    query2 = "How about in London?"
    print(f"User: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")


if __name__ == "__main__":
    asyncio.run(main())
