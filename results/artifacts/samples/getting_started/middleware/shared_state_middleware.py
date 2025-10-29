# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Awaitable, Callable
from random import randint
from typing import Annotated

from agent_framework import (
    FunctionInvocationContext,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Shared State Function-based Middleware Example

This sample demonstrates how to implement function-based middleware within a class to share state.
The example includes:

- A MiddlewareContainer class with two simple function middleware methods
- First middleware: Counts function calls and stores the count in shared state
- Second middleware: Uses the shared count to add call numbers to function results

This approach shows how middleware can work together by sharing state within the same class instance.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


def get_time(
    timezone: Annotated[str, Field(description="The timezone to get the time for.")] = "UTC",
) -> str:
    """指定されたタイムゾーンの現在時刻を取得します。"""
    import datetime

    return f"The current time in {timezone} is {datetime.datetime.now().strftime('%H:%M:%S')}"


class MiddlewareContainer:
    """共有状態を持つミドルウェア関数を保持するコンテナクラス。"""

    def __init__(self) -> None:
        # シンプルな共有状態：関数呼び出し回数をカウントする
        self.call_count: int = 0

    async def call_counter_middleware(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        """最初のミドルウェア：共有状態の呼び出し回数をインクリメントする。"""
        # 共有の呼び出し回数をインクリメントする
        self.call_count += 1

        print(f"[CallCounter] This is function call #{self.call_count}")

        # 次のミドルウェア/関数を呼び出す
        await next(context)

    async def result_enhancer_middleware(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        """2番目のミドルウェア：共有の呼び出し回数を使って関数の結果を強化する。"""
        print(f"[ResultEnhancer] Current total calls so far: {self.call_count}")

        # 次のミドルウェア/関数を呼び出す
        await next(context)

        # 関数実行後、共有状態を使って結果を強化する
        if context.result:
            enhanced_result = f"[Call #{self.call_count}] {context.result}"
            context.result = enhanced_result
            print("[ResultEnhancer] Enhanced result with call number")


async def main() -> None:
    """共有状態を用いた関数ベースのミドルウェアを示す例。"""
    print("=== Shared State Function-based Middleware Example ===")

    # 共有状態を持つミドルウェアコンテナを作成する
    middleware_container = MiddlewareContainer()

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="UtilityAgent",
            instructions="You are a helpful assistant that can provide weather information and current time.",
            tools=[get_weather, get_time],
            # 同じコンテナインスタンスから両方のミドルウェア関数を渡す 順序が重要：カウンターが最初に実行されてカウントを増やし、
            # その後、結果強化が更新されたカウントを使用する
            middleware=[
                middleware_container.call_counter_middleware,
                middleware_container.result_enhancer_middleware,
            ],
        ) as agent,
    ):
        # 複数のリクエストをテストして共有状態の動作を確認する
        queries = [
            "What's the weather like in New York?",
            "What time is it in London?",
            "What's the weather in Tokyo?",
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i} ---")
            print(f"User: {query}")
            result = await agent.run(query)
            print(f"Agent: {result.text if result.text else 'No response'}")

        # 最終統計を表示する
        print("\n=== Final Statistics ===")
        print(f"Total function calls made: {middleware_container.call_count}")


if __name__ == "__main__":
    asyncio.run(main())
