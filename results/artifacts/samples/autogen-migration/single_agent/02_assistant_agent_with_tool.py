# Copyright (c) Microsoft. All rights reserved.
"""AutoGen AssistantAgent と Agent Framework ChatAgent の関数ツール付き比較。

両フレームワークでエージェントにツールを作成してアタッチする方法を示します。
"""

import asyncio


async def run_autogen() -> None:
    """FunctionTool を持つ AutoGen エージェント。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # シンプルなツール関数を定義する
    def get_weather(location: str) -> str:
        """場所の天気を取得する。

        Args:
            location: 都市名または場所。

        Returns:
            天気の説明。

        """
        return f"The weather in {location} is sunny and 72°F."

    # 関数を FunctionTool でラップする
    weather_tool = FunctionTool(
        func=get_weather,
        description="Get weather information for a location",
    )

    # ツール付きエージェントを作成する
    client = OpenAIChatCompletionClient(model="gpt-4.1-mini")
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=[weather_tool],
        system_message="You are a helpful assistant. Use available tools to answer questions.",
    )

    # ツール使用で実行する
    result = await agent.run(task="What's the weather in Seattle?")
    print("[AutoGen]", result.messages[-1].to_text())


async def run_agent_framework() -> None:
    """@ai_function デコレータを使った Agent Framework エージェント。"""
    from agent_framework import ai_function
    from agent_framework.openai import OpenAIChatClient

    # @ai_function デコレータでツールを定義する（自動スキーマ推論）
    @ai_function
    def get_weather(location: str) -> str:
        """場所の天気を取得する。

        Args:
            location: 都市名または場所。

        Returns:
            天気の説明。

        """
        return f"The weather in {location} is sunny and 72°F."

    # ツール付きエージェントを作成する
    client = OpenAIChatClient(model_id="gpt-4.1-mini")
    agent = client.create_agent(
        name="assistant",
        instructions="You are a helpful assistant. Use available tools to answer questions.",
        tools=[get_weather],
    )

    # ツール使用で実行する
    result = await agent.run("What's the weather in Seattle?")
    print("[Agent Framework]", result.text)


async def main() -> None:
    print("=" * 60)
    print("Assistant Agent with Tools Comparison")
    print("=" * 60)
    await run_autogen()
    print()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
