# Copyright (c) Microsoft. All rights reserved.
"""Agent Framework Debug UI用のFoundryベースのWeather Agent。

このAgentはAzure AI FoundryとAzure CLI認証を使用します。
devui開始前に必ず 'az login' を実行してください。
"""

import os
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    temperature = 22
    return f"The weather in {location} is {conditions[0]} with a high of {temperature}°C."


def get_forecast(
    location: Annotated[str, Field(description="The location to get the forecast for.")],
    days: Annotated[int, Field(description="Number of days for forecast")] = 3,
) -> str:
    """複数日の天気予報を取得。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    forecast: list[str] = []

    for day in range(1, days + 1):
        condition = conditions[day % len(conditions)]
        temp = 18 + day
        forecast.append(f"Day {day}: {condition}, {temp}°C")

    return f"Weather forecast for {location}:\n" + "\n".join(forecast)


# Agent Frameworkの規約に従ったAgentインスタンス
agent = ChatAgent(
    name="FoundryWeatherAgent",
    chat_client=AzureAIAgentClient(
        project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
        model_deployment_name=os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME"),
        async_credential=AzureCliCredential(),
    ),
    instructions="""
    You are a weather assistant using Azure AI Foundry models. You can provide
    current weather information and forecasts for any location. Always be helpful
    and provide detailed weather information when asked.
    """,
    tools=[get_weather, get_forecast],
)


def main():
    """DevUIでFoundry Weather Agentを起動。"""
    import logging

    from agent_framework.devui import serve

    # ログ設定
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Foundry Weather Agent")
    logger.info("Available at: http://localhost:8090")
    logger.info("Entity ID: agent_FoundryWeatherAgent")
    logger.info("Note: Make sure 'az login' has been run for authentication")

    # Agentでサーバーを起動
    serve(entities=[agent], port=8090, auto_open=True)


if __name__ == "__main__":
    main()
