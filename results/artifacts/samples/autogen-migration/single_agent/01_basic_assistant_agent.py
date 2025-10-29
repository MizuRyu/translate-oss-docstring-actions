# Copyright (c) Microsoft. All rights reserved.
"""Basic AutoGen AssistantAgent と Agent Framework ChatAgent の比較。

両方のサンプルは OpenAI 互換の環境変数（OPENAI_API_KEY または Azure OpenAI の設定）を期待しています。実行前にプロンプトやクライアントの配線を選択したモデルに合わせて更新してください。
"""

import asyncio


async def run_autogen() -> None:
    """簡単な質問のために AutoGen の AssistantAgent を呼び出す。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # OpenAI モデルクライアントを使用した AutoGen エージェント
    client = OpenAIChatCompletionClient(model="gpt-4.1-mini")
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        system_message="You are a helpful assistant. Answer in one sentence.",
    )

    # エージェントを実行する（AutoGen は会話の状態を内部で管理します）
    result = await agent.run(task="What is the capital of France?")
    print("[AutoGen]", result.messages[-1].to_text())


async def run_agent_framework() -> None:
    """OpenAIChatClient から作成された Agent Framework の ChatAgent を呼び出す。"""
    from agent_framework.openai import OpenAIChatClient

    # AF は OpenAIChatClient をバックエンドにした軽量の ChatAgent を構築します
    client = OpenAIChatClient(model_id="gpt-4.1-mini")
    agent = client.create_agent(
        name="assistant",
        instructions="You are a helpful assistant. Answer in one sentence.",
    )

    # エージェントを実行する（AF エージェントはデフォルトでステートレスです）
    result = await agent.run("What is the capital of France?")
    print("[Agent Framework]", result.text)


async def main() -> None:
    print("=" * 60)
    print("Basic Assistant Agent Comparison")
    print("=" * 60)
    await run_autogen()
    print()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
