# Copyright (c) Microsoft. All rights reserved.

"""Agent Framework DevUIを使用したインメモリエンティティ登録の例。

これはAgentやワークフローをOpenAI互換APIエンドポイントとして提供する最も簡単な方法を示します。
異なるエンティティタイプを示すためにAgentと基本的なワークフローの両方を含みます。
"""

import logging
import os
from typing import Annotated

from agent_framework import ChatAgent, Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.devui import serve
from typing_extensions import Never


# Agentのためのツール関数
def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    temperature = 53
    return f"The weather in {location} is {conditions[0]} with a high of {temperature}°C."


def get_time(
    timezone: Annotated[str, "The timezone to get time for."] = "UTC",
) -> str:
    """タイムゾーンの現在時刻を取得します。"""
    from datetime import datetime

    # 例のために簡略化しています
    return f"Current time in {timezone}: {datetime.now().strftime('%H:%M:%S')}"


# 基本的なワークフローのExecutor
class UpperCase(Executor):
    """テキストを大文字に変換します。"""

    @handler
    async def to_upper(self, text: str, ctx: WorkflowContext[str]) -> None:
        """入力を大文字に変換し、次のExecutorに渡します。"""
        result = text.upper()
        await ctx.send_message(result)


class AddExclamation(Executor):
    """テキストに感嘆符を追加します。"""

    @handler
    async def add_exclamation(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        """感嘆符を追加し、ワークフローの出力としてyieldします。"""
        result = f"{text}!"
        await ctx.yield_output(result)


def main():
    """インメモリのエンティティ登録を示すメイン関数。"""
    # ログのセットアップ
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Azure OpenAIのChat Clientを作成します。
    chat_client = AzureOpenAIChatClient(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        model_id=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    )

    # Agentを作成します。
    weather_agent = ChatAgent(
        name="weather-assistant",
        description="Provides weather information and time",
        instructions=(
            "You are a helpful weather and time assistant. Use the available tools to "
            "provide accurate weather information and current time for any location."
        ),
        chat_client=chat_client,
        tools=[get_weather, get_time],
    )

    simple_agent = ChatAgent(
        name="general-assistant",
        description="A simple conversational agent",
        instructions="You are a helpful assistant.",
        chat_client=chat_client,
    )

    # 基本的なワークフローを作成します: Input -> UpperCase -> AddExclamation -> Output
    upper_executor = UpperCase(id="upper_case")
    exclaim_executor = AddExclamation(id="add_exclamation")

    basic_workflow = (
        WorkflowBuilder(
            name="Text Transformer",
            description="Simple 2-step workflow that converts text to uppercase and adds exclamation",
        )
        .set_start_executor(upper_executor)
        .add_edge(upper_executor, exclaim_executor)
        .build()
    )

    # サービング用のエンティティを収集します。
    entities = [weather_agent, simple_agent, basic_workflow]

    logger.info("Starting DevUI on http://localhost:8090")
    logger.info("Entities available:")
    logger.info("  - Agents: weather-assistant, general-assistant")
    logger.info("  - Workflow: basic text transformer (uppercase + exclamation)")

    # 自動生成されたエンティティIDでサーバーを起動します。
    serve(entities=entities, port=8090, auto_open=True)


if __name__ == "__main__":
    main()
