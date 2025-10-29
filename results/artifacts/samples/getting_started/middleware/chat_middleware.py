# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Awaitable, Callable
from random import randint
from typing import Annotated

from agent_framework import (
    ChatContext,
    ChatMessage,
    ChatMiddleware,
    ChatResponse,
    Role,
    chat_middleware,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Chat Middleware Example

This sample demonstrates how to use chat middleware to observe and override
inputs sent to AI models. Chat middleware intercepts chat requests before they reach
the underlying AI service, allowing you to:

1. Observe and log input messages
2. Modify input messages before sending to AI
3. Override the entire response

The example covers:
- Class-based chat middleware inheriting from ChatMiddleware
- Function-based chat middleware with @chat_middleware decorator
- Middleware registration at agent level (applies to all runs)
- Middleware registration at run level (applies to specific run only)
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


class InputObserverMiddleware(ChatMiddleware):
    """入力メッセージを観察し修正するクラスベースのMiddleware。"""

    def __init__(self, replacement: str | None = None):
        """ユーザーメッセージの置き換えで初期化する。"""
        self.replacement = replacement

    async def process(
        self,
        context: ChatContext,
        next: Callable[[ChatContext], Awaitable[None]],
    ) -> None:
        """AIに送信される前の入力メッセージを観察し修正する。"""
        print("[InputObserverMiddleware] Observing input messages:")

        for i, message in enumerate(context.messages):
            content = message.text if message.text else str(message.contents)
            print(f"  Message {i + 1} ({message.role.value}): {content}")

        print(f"[InputObserverMiddleware] Total messages: {len(context.messages)}")

        # 強化されたテキストで新しいメッセージを作成してユーザーメッセージを修正する
        modified_messages: list[ChatMessage] = []
        modified_count = 0

        for message in context.messages:
            if message.role == Role.USER and message.text:
                original_text = message.text
                updated_text = original_text

                if self.replacement:
                    updated_text = self.replacement
                    print(f"[InputObserverMiddleware] Updated: '{original_text}' -> '{updated_text}'")

                modified_message = ChatMessage(role=message.role, text=updated_text)
                modified_messages.append(modified_message)
                modified_count += 1
            else:
                modified_messages.append(message)

        # コンテキスト内のメッセージを置き換える
        context.messages[:] = modified_messages

        # 次のMiddlewareまたはAI実行に進む
        await next(context)

        # 処理が完了したことを観察する
        print("[InputObserverMiddleware] Processing completed")


@chat_middleware
async def security_and_override_middleware(
    context: ChatContext,
    next: Callable[[ChatContext], Awaitable[None]],
) -> None:
    """セキュリティフィルタリングとレスポンス上書きを実装する関数ベースのMiddleware。"""
    print("[SecurityMiddleware] Processing input...")

    # セキュリティチェック - 機密情報をブロックする
    blocked_terms = ["password", "secret", "api_key", "token"]

    for message in context.messages:
        if message.text:
            message_lower = message.text.lower()
            for term in blocked_terms:
                if term in message_lower:
                    print(f"[SecurityMiddleware] BLOCKED: Found '{term}' in message")

                    # AIを呼び出す代わりにレスポンスを上書きする
                    context.result = ChatResponse(
                        messages=[
                            ChatMessage(
                                role=Role.ASSISTANT,
                                text="I cannot process requests containing sensitive information. "
                                "Please rephrase your question without including passwords, secrets, or other "
                                "sensitive data.",
                            )
                        ]
                    )

                    # 実行停止のためにterminateフラグを設定する
                    context.terminate = True
                    return

    # 次のMiddlewareまたはAI実行に進む
    await next(context)


async def class_based_chat_middleware() -> None:
    """AgentレベルでのクラスベースMiddlewareのデモ。"""
    print("\n" + "=" * 60)
    print("Class-based Chat Middleware (Agent Level)")
    print("=" * 60)

    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="EnhancedChatAgent",
            instructions="You are a helpful AI assistant.",
            # AgentレベルでクラスベースMiddlewareを登録（すべての実行に適用）
            middleware=InputObserverMiddleware(),
            tools=get_weather,
        ) as agent,
    ):
        query = "What's the weather in Seattle?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Final Response: {result.text if result.text else 'No response'}")


async def function_based_chat_middleware() -> None:
    """Agentレベルでの関数ベースMiddlewareのデモ。"""
    print("\n" + "=" * 60)
    print("Function-based Chat Middleware (Agent Level)")
    print("=" * 60)

    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="FunctionMiddlewareAgent",
            instructions="You are a helpful AI assistant.",
            # Agentレベルで関数ベースMiddlewareを登録する
            middleware=security_and_override_middleware,
        ) as agent,
    ):
        # 通常クエリのシナリオ
        print("\n--- Scenario 1: Normal Query ---")
        query = "Hello, how are you?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Final Response: {result.text if result.text else 'No response'}")

        # セキュリティ違反のシナリオ
        print("\n--- Scenario 2: Security Violation ---")
        query = "What is my password for this account?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Final Response: {result.text if result.text else 'No response'}")


async def run_level_middleware() -> None:
    """RunレベルでのMiddleware登録のデモ。"""
    print("\n" + "=" * 60)
    print("Run-level Chat Middleware")
    print("=" * 60)

    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="RunLevelAgent",
            instructions="You are a helpful AI assistant.",
            tools=get_weather,
            # AgentレベルにMiddlewareなし
        ) as agent,
    ):
        # シナリオ1: Middlewareなしで実行
        print("\n--- Scenario 1: No Middleware ---")
        query = "What's the weather in Tokyo?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Response: {result.text if result.text else 'No response'}")

        # シナリオ2: この呼び出しのみに特定Middlewareを適用（強化とセキュリティの両方）
        print("\n--- Scenario 2: With Run-level Middleware ---")
        print(f"User: {query}")
        result = await agent.run(
            query,
            middleware=[
                InputObserverMiddleware(replacement="What's the weather in Madrid?"),
                security_and_override_middleware,
            ],
        )
        print(f"Response: {result.text if result.text else 'No response'}")

        # シナリオ3: RunレベルMiddlewareによるセキュリティテスト
        print("\n--- Scenario 3: Security Test with Run-level Middleware ---")
        query = "Can you help me with my secret API key?"
        print(f"User: {query}")
        result = await agent.run(
            query,
            middleware=security_and_override_middleware,
        )
        print(f"Response: {result.text if result.text else 'No response'}")


async def main() -> None:
    """すべてのChat Middleware例を実行する。"""
    print("Chat Middleware Examples")
    print("========================")

    await class_based_chat_middleware()
    await function_based_chat_middleware()
    await run_level_middleware()


if __name__ == "__main__":
    asyncio.run(main())
