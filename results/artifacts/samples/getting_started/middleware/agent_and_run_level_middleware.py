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
    FunctionInvocationContext,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Agent-Level and Run-Level Middleware Example

This sample demonstrates the difference between agent-level and run-level middleware:

- Agent-level middleware: Applied to ALL runs of the agent (persistent across runs)
- Run-level middleware: Applied to specific runs only (isolated per run)

The example shows:
1. Agent-level security middleware that validates all requests
2. Agent-level performance monitoring across all runs
3. Run-level context middleware for specific use cases (high priority, debugging)
4. Run-level caching middleware for expensive operations

Execution order: Agent middleware (outermost) -> Run middleware (innermost) -> Agent execution
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


# AgentレベルのMiddleware（すべての実行に適用）
class SecurityAgentMiddleware(AgentMiddleware):
    """すべてのRequestを検証するAgentレベルのセキュリティMiddleware。"""

    async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
        print("[SecurityMiddleware] Checking security for all requests...")

        # 最後のユーザーメッセージにセキュリティ違反がないかチェックする
        last_message = context.messages[-1] if context.messages else None
        if last_message and last_message.text:
            query = last_message.text.lower()
            if any(word in query for word in ["password", "secret", "credentials"]):
                print("[SecurityMiddleware] Security violation detected! Blocking request.")
                return  # 実行を防ぐためにnext()を呼び出さない

        print("[SecurityMiddleware] Security check passed.")
        context.metadata["security_validated"] = True
        await next(context)


async def performance_monitor_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    """すべての実行に対するAgentレベルのパフォーマンス監視。"""
    print("[PerformanceMonitor] Starting performance monitoring...")
    start_time = time.time()

    await next(context)

    end_time = time.time()
    duration = end_time - start_time
    print(f"[PerformanceMonitor] Total execution time: {duration:.3f}s")
    context.metadata["execution_time"] = duration


# RunレベルMiddleware（特定の実行にのみ適用）
class HighPriorityMiddleware(AgentMiddleware):
    """高優先度RequestのためのRunレベルMiddleware。"""

    async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
        print("[HighPriority] Processing high priority request with expedited handling...")

        # AgentレベルMiddlewareによって設定されたメタデータを読み取る
        if context.metadata.get("security_validated"):
            print("[HighPriority] Security validation confirmed from agent middleware")

        # 高優先度フラグを設定する
        context.metadata["priority"] = "high"
        context.metadata["expedited"] = True

        await next(context)
        print("[HighPriority] High priority processing completed")


async def debugging_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    """特定の実行のトラブルシューティングのためのRunレベルデバッグMiddleware。"""
    print("[Debug] Debug mode enabled for this run")
    print(f"[Debug] Messages count: {len(context.messages)}")
    print(f"[Debug] Is streaming: {context.is_streaming}")

    # Agent Middlewareから既存のメタデータをログに記録する
    if context.metadata:
        print(f"[Debug] Existing metadata: {context.metadata}")

    context.metadata["debug_enabled"] = True

    await next(context)

    print("[Debug] Debug information collected")


class CachingMiddleware(AgentMiddleware):
    """高コスト操作のためのRunレベルキャッシュMiddleware。"""

    def __init__(self) -> None:
        self.cache: dict[str, AgentRunResponse] = {}

    async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
        # 最後のメッセージからシンプルなキャッシュキーを作成する
        last_message = context.messages[-1] if context.messages else None
        cache_key: str = last_message.text if last_message and last_message.text else "no_message"

        if cache_key in self.cache:
            print(f"[Cache] Cache HIT for: '{cache_key[:30]}...'")
            context.result = self.cache[cache_key]  # type: ignore
            return  # next()を呼ばずにキャッシュされた結果を返す

        print(f"[Cache] Cache MISS for: '{cache_key[:30]}...'")
        context.metadata["cache_key"] = cache_key

        await next(context)

        # 結果があればキャッシュする
        if context.result:
            self.cache[cache_key] = context.result  # type: ignore
            print("[Cache] Result cached for future use")


async def function_logging_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    """すべての関数呼び出しをログに記録するFunction Middleware。"""
    function_name = context.function.name
    args = context.arguments
    print(f"[FunctionLog] Calling function: {function_name} with args: {args}")

    await next(context)

    print(f"[FunctionLog] Function {function_name} completed")


async def main() -> None:
    """AgentレベルとRunレベルMiddlewareを示す例。"""
    print("=== Agent-Level and Run-Level Middleware Example ===\n")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="WeatherAgent",
            instructions="You are a helpful weather assistant.",
            tools=get_weather,
            # AgentレベルMiddleware：すべての実行に適用
            middleware=[
                SecurityAgentMiddleware(),
                performance_monitor_middleware,
                function_logging_middleware,
            ],
        ) as agent,
    ):
        print("Agent created with agent-level middleware:")
        print("   - SecurityMiddleware (blocks sensitive requests)")
        print("   - PerformanceMonitor (tracks execution time)")
        print("   - FunctionLogging (logs all function calls)")
        print()

        # Run 1: RunレベルMiddlewareなしの通常クエリ
        print("=" * 60)
        print("RUN 1: Normal query (agent-level middleware only)")
        print("=" * 60)
        query = "What's the weather like in Paris?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text if result.text else 'No response'}")
        print()

        # Run 2: RunレベルMiddleware付きの高優先度Request
        print("=" * 60)
        print("RUN 2: High priority request (agent + run-level middleware)")
        print("=" * 60)
        query = "What's the weather in Tokyo? This is urgent!"
        print(f"User: {query}")
        result = await agent.run(
            query,
            middleware=HighPriorityMiddleware(),  # Run-level middleware
        )
        print(f"Agent: {result.text if result.text else 'No response'}")
        print()

        # Run 3: RunレベルデバッグMiddleware付きのデバッグモード
        print("=" * 60)
        print("RUN 3: Debug mode (agent + run-level debugging)")
        print("=" * 60)
        query = "What's the weather in London?"
        print(f"User: {query}")
        result = await agent.run(
            query,
            middleware=[debugging_middleware],  # Run-level middleware
        )
        print(f"Agent: {result.text if result.text else 'No response'}")
        print()

        # Run 4: 複数のRunレベルMiddleware
        print("=" * 60)
        print("RUN 4: Multiple run-level middleware (caching + debug)")
        print("=" * 60)
        caching = CachingMiddleware()
        query = "What's the weather in New York?"
        print(f"User: {query}")
        result = await agent.run(
            query,
            middleware=[caching, debugging_middleware],  # Multiple run-level middleware
        )
        print(f"Agent: {result.text if result.text else 'No response'}")
        print()

        # Run 5: 同じクエリでのキャッシュヒットテスト
        print("=" * 60)
        print("RUN 5: Test cache hit (same query as Run 4)")
        print("=" * 60)
        print(f"User: {query}")  # Run 4と同じクエリ
        result = await agent.run(
            query,
            middleware=[caching],  # Same caching middleware instance
        )
        print(f"Agent: {result.text if result.text else 'No response'}")
        print()

        # Run 6: セキュリティ違反テスト
        print("=" * 60)
        print("RUN 6: Security test (should be blocked by agent middleware)")
        print("=" * 60)
        query = "What's the secret weather password for Berlin?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text if result.text else 'Request was blocked by security middleware'}")
        print()

        # Run 7: 再び通常クエリ（RunレベルMiddlewareの干渉なし）
        print("=" * 60)
        print("RUN 7: Normal query again (agent-level middleware only)")
        print("=" * 60)
        query = "What's the weather in Sydney?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result.text if result.text else 'No response'}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
