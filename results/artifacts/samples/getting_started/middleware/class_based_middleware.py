# Copyright (c) Microsoft. All rights reserved.

import asyncio
import time
from collections.abc import Awaitable, Callable
from random import randint
from typing import Annotated

from agent_framework import (
    AgentMiddleware,
    AgentRunContext,
    AgentRunResponse,
    ChatMessage,
    FunctionInvocationContext,
    FunctionMiddleware,
    Role,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Class-based Middleware Example

This sample demonstrates how to implement middleware using class-based approach by inheriting
from AgentMiddleware and FunctionMiddleware base classes. The example includes:

- SecurityAgentMiddleware: Checks for security violations in user queries and blocks requests
  containing sensitive information like passwords or secrets
- LoggingFunctionMiddleware: Logs function execution details including timing and parameters

This approach is useful when you need stateful middleware or complex logic that benefits
from object-oriented design patterns.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


class SecurityAgentMiddleware(AgentMiddleware):
    """セキュリティ違反をチェックするAgent Middleware。"""

    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Awaitable[None]],
    ) -> None:
        # クエリの潜在的なセキュリティ違反をチェックする 最後のユーザーメッセージを見る
        last_message = context.messages[-1] if context.messages else None
        if last_message and last_message.text:
            query = last_message.text
            if "password" in query.lower() or "secret" in query.lower():
                print("[SecurityAgentMiddleware] Security Warning: Detected sensitive information, blocking request.")
                # 警告メッセージで結果を上書きする
                context.result = AgentRunResponse(
                    messages=[
                        ChatMessage(role=Role.ASSISTANT, text="Detected sensitive information, the request is blocked.")
                    ]
                )
                # 単にnext()を呼ばずに実行を防ぐ
                return

        print("[SecurityAgentMiddleware] Security check passed.")
        await next(context)


class LoggingFunctionMiddleware(FunctionMiddleware):
    """関数呼び出しをログに記録するFunction Middleware。"""

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        function_name = context.function.name
        print(f"[LoggingFunctionMiddleware] About to call function: {function_name}.")

        start_time = time.time()

        await next(context)

        end_time = time.time()
        duration = end_time - start_time

        print(f"[LoggingFunctionMiddleware] Function {function_name} completed in {duration:.5f}s.")


async def main() -> None:
    """クラスベースMiddlewareのデモ。"""
    print("=== Class-based Middleware Example ===")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather assistant.",
            tools=get_weather,
            middleware=[SecurityAgentMiddleware(), LoggingFunctionMiddleware()],
        ) as agent,
    ):
        # 通常クエリでテストする
        print("\n--- Normal Query ---")
        query = "What's the weather like in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text}\n")

        # セキュリティ関連クエリでテストする
        print("--- Security Test ---")
        query = "What's the password for the weather service?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
