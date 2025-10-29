# Copyright (c) Microsoft. All rights reserved.
"""基本的なSKのChatCompletionAgentとAgent FrameworkのChatAgentの比較。

両方のサンプルはOpenAI互換の環境変数（OPENAI_API_KEYまたはAzure OpenAIの設定）を期待します。実行前にプロンプトやクライアントの接続を使用するモデルに合わせて更新してください。
"""

import asyncio


async def run_semantic_kernel() -> None:
    """簡単な質問に対してSKのChatCompletionAgentを呼び出します。"""
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    # SKのAgentはChatCompletionAgentを通じてスレッド状態を内部で保持します。
    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(),
        name="Support",
        instructions="Answer in one sentence.",
    )
    response = await agent.get_response(messages="How do I reset my bike tire?")
    print("[SK]", response.message.content)


async def run_agent_framework() -> None:
    """Agent FrameworkのOpenAIChatClientから作成されたChatAgentを呼び出します。"""
    from agent_framework.openai import OpenAIChatClient

    # AFはOpenAIChatClientをバックエンドにした軽量なChatAgentを構築します。
    chat_agent = OpenAIChatClient().create_agent(
        name="Support",
        instructions="Answer in one sentence.",
    )
    reply = await chat_agent.run("How do I reset my bike tire?")
    print("[AF]", reply.text)


async def main() -> None:
    await run_semantic_kernel()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
