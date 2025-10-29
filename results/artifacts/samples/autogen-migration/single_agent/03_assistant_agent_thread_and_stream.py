# Copyright (c) Microsoft. All rights reserved.
"""AutoGen と Agent Framework: Thread 管理とストリーミングレスポンス。

両フレームワークでの会話状態管理とストリーミングを示します。
"""

import asyncio


async def run_autogen() -> None:
    """会話履歴とストリーミングを持つ AutoGen エージェント。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(model="gpt-4.1-mini")
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        system_message="You are a helpful math tutor.",
        model_client_stream=True,
    )

    print("[AutoGen] Conversation with history:")
    # 最初のターン - AutoGen は内部で状態を管理し、Console でストリーミングを行う
    result = await agent.run(task="What is 15 + 27?")
    print(f"  Q1: {result.messages[-1].to_text()}")

    # 2回目のターン - エージェントはコンテキストを記憶する
    result = await agent.run(task="What about that number times 2?")
    print(f"  Q2: {result.messages[-1].to_text()}")

    print("\n[AutoGen] Streaming response:")
    # トークンストリーミングのために Console でレスポンスをストリームする
    await Console(agent.run_stream(task="Count from 1 to 5"))


async def run_agent_framework() -> None:
    """明示的なスレッドとストリーミングを持つ Agent Framework エージェント。"""
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4.1-mini")
    agent = client.create_agent(
        name="assistant",
        instructions="You are a helpful math tutor.",
    )

    print("[Agent Framework] Conversation with thread:")
    # 状態を維持するためにスレッドを作成する
    thread = agent.get_new_thread()

    # 最初のターン - 履歴を維持するためにスレッドを渡す
    result1 = await agent.run("What is 15 + 27?", thread=thread)
    print(f"  Q1: {result1.text}")

    # 2回目のターン - スレッドを介してエージェントはコンテキストを記憶する
    result2 = await agent.run("What about that number times 2?", thread=thread)
    print(f"  Q2: {result2.text}")

    print("\n[Agent Framework] Streaming response:")
    # レスポンスをストリームする
    print("  ", end="")
    async for chunk in agent.run_stream("Count from 1 to 5"):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


async def main() -> None:
    print("=" * 60)
    print("Thread Management and Streaming Comparison")
    print("=" * 60)
    await run_autogen()
    print()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
