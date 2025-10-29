# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

from agent_framework import FunctionInvocationContext
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Exception Handling with Middleware

This sample demonstrates how to use middleware for centralized exception handling in function calls.
The example shows:

- How to catch exceptions thrown by functions and provide graceful error responses
- Overriding function results when errors occur to provide user-friendly messages
- Using middleware to implement retry logic, fallback mechanisms, or error reporting

The middleware catches TimeoutError from an unstable data service and replaces it with
a helpful message for the user, preventing raw exceptions from reaching the end user.
"""


def unstable_data_service(
    query: Annotated[str, Field(description="The data query to execute.")],
) -> str:
    """時々例外をスローするシミュレートされたデータサービス。"""
    # 失敗をシミュレートする
    raise TimeoutError("Data service request timed out")


async def exception_handling_middleware(
    context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
) -> None:
    function_name = context.function.name

    try:
        print(f"[ExceptionHandlingMiddleware] Executing function: {function_name}")
        await next(context)
        print(f"[ExceptionHandlingMiddleware] Function {function_name} completed successfully.")
    except TimeoutError as e:
        print(f"[ExceptionHandlingMiddleware] Caught TimeoutError: {e}")
        # レスポンスにカスタムメッセージを提供するために関数結果を上書きする。
        context.result = (
            "Request Timeout: The data service is taking longer than expected to respond.",
            "Respond with message - 'Sorry for the inconvenience, please try again later.'",
        )


async def main() -> None:
    """Middlewareによる例外処理のデモ。"""
    print("=== Exception Handling Middleware Example ===")

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="DataAgent",
            instructions="You are a helpful data assistant. Use the data service tool to fetch information for users.",
            tools=unstable_data_service,
            middleware=exception_handling_middleware,
        ) as agent,
    ):
        query = "Get user statistics"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
