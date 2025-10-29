# 著作権 (c) Microsoft。無断転載を禁じます。

import asyncio

from agent_framework.microsoft import CopilotStudioAgent

"""
Copilot Studio Agent Basic Example

This sample demonstrates basic usage of CopilotStudioAgent with automatic configuration
from environment variables, showing both streaming and non-streaming responses.
"""

# 必要な環境変数: COPILOTSTUDIOAGENT__ENVIRONMENTID - Copilotがデプロイされている環境ID
# COPILOTSTUDIOAGENT__SCHEMANAME - CopilotのAgent識別子/スキーマ名 COPILOTSTUDIOAGENT__AGENTAPPID
# - 認証用クライアントID COPILOTSTUDIOAGENT__TENANTID - 認証用テナントID


async def non_streaming_example() -> None:
    """非ストリーミングレスポンスの例（一度に完全な結果を取得）。"""
    print("=== Non-streaming Response Example ===")

    agent = CopilotStudioAgent()

    query = "What is the capital of France?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Agent: {result}\n")


async def streaming_example() -> None:
    """ストリーミングレスポンスの例（生成される結果を逐次取得）。"""
    print("=== Streaming Response Example ===")

    agent = CopilotStudioAgent()

    query = "What is the capital of Spain?"
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.run_stream(query):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


async def main() -> None:
    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
