# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Awaitable, Callable
from random import randint
from typing import Annotated

from agent_framework import (
    AgentMiddleware,
    AgentRunContext,
    AgentRunResponse,
    ChatMessage,
    Role,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Middleware Termination Example

This sample demonstrates how middleware can terminate execution using the `context.terminate` flag.
The example includes:

- PreTerminationMiddleware: Terminates execution before calling next() to prevent agent processing
- PostTerminationMiddleware: Allows processing to complete but terminates further execution

This is useful for implementing security checks, rate limiting, or early exit conditions.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


class PreTerminationMiddleware(AgentMiddleware):
    """Agentを呼び出す前に実行を終了するMiddleware。"""

    def __init__(self, blocked_words: list[str]):
        self.blocked_words = [word.lower() for word in blocked_words]

    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Awaitable[None]],
    ) -> None:
        # ユーザーメッセージにブロックされた単語が含まれているかチェックする
        last_message = context.messages[-1] if context.messages else None
        if last_message and last_message.text:
            query = last_message.text.lower()
            for blocked_word in self.blocked_words:
                if blocked_word in query:
                    print(f"[PreTerminationMiddleware] Blocked word '{blocked_word}' detected. Terminating request.")

                    # カスタムレスポンスを設定する
                    context.result = AgentRunResponse(
                        messages=[
                            ChatMessage(
                                role=Role.ASSISTANT,
                                text=(
                                    f"Sorry, I cannot process requests containing '{blocked_word}'. "
                                    "Please rephrase your question."
                                ),
                            )
                        ]
                    )

                    # さらなる処理を防ぐためにterminateフラグを設定する
                    context.terminate = True
                    break

        await next(context)


class PostTerminationMiddleware(AgentMiddleware):
    """処理を許可するが複数の実行で最大レスポンス数に達したら終了するMiddleware。"""

    def __init__(self, max_responses: int = 1):
        self.max_responses = max_responses
        self.response_count = 0

    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Awaitable[None]],
    ) -> None:
        print(f"[PostTerminationMiddleware] Processing request (response count: {self.response_count})")

        # 処理前に終了すべきかチェックする
        if self.response_count >= self.max_responses:
            print(
                f"[PostTerminationMiddleware] Maximum responses ({self.max_responses}) reached. "
                "Terminating further processing."
            )
            context.terminate = True

        # Agentに通常通り処理させる
        await next(context)

        # 処理後にレスポンス数をインクリメントする
        self.response_count += 1


async def pre_termination_middleware() -> None:
    """特定の単語を含むリクエストをブロックする事前終了Middlewareのデモ。"""
    print("\n--- Example 1: Pre-termination Middleware ---")
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather assistant.",
            tools=get_weather,
            middleware=PreTerminationMiddleware(blocked_words=["bad", "inappropriate"]),
        ) as agent,
    ):
        # 通常クエリでテストする
        print("\n1. Normal query:")
        query = "What's the weather like in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text}")

        # ブロックされた単語でテストする
        print("\n2. Query with blocked word:")
        query = "What's the bad weather in New York?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text}")


async def post_termination_middleware() -> None:
    """複数の実行でレスポンスを制限する事後終了Middlewareのデモ。"""
    print("\n--- Example 2: Post-termination Middleware ---")
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather assistant.",
            tools=get_weather,
            middleware=PostTerminationMiddleware(max_responses=1),
        ) as agent,
    ):
        # 最初の実行（動作するはず）
        print("\n1. First run:")
        query = "What's the weather in Paris?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text}")

        # 2回目の実行（Middlewareによって終了されるはず）
        print("\n2. Second run (should be terminated):")
        query = "What about the weather in London?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text if result.text else 'No response (terminated)'}")

        # 3回目の実行（これも終了するはずです）
        print("\n3. Third run (should also be terminated):")
        query = "And New York?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text if result.text else 'No response (terminated)'}")


async def main() -> None:
    """ミドルウェアの終了機能を示す例です。"""
    print("=== Middleware Termination Example ===")
    await pre_termination_middleware()
    await post_termination_middleware()


if __name__ == "__main__":
    asyncio.run(main())
