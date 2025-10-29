# Copyright (c) Microsoft. All rights reserved.
"""Agent Framework Debug UI用の天気Agentサンプル。"""

import os
from collections.abc import Awaitable, Callable
from typing import Annotated

from agent_framework import (
    ChatAgent,
    ChatContext,
    ChatMessage,
    ChatResponse,
    FunctionInvocationContext,
    Role,
    chat_middleware,
    function_middleware,
)
from agent_framework.azure import AzureOpenAIChatClient


@chat_middleware
async def security_filter_middleware(
    context: ChatContext,
    next: Callable[[ChatContext], Awaitable[None]],
) -> None:
    """機密情報を含むリクエストをブロックするChat middleware。"""
    # 機密情報を含むリクエストをブロックします。
    blocked_terms = ["password", "secret", "api_key", "token"]

    for message in context.messages:
        if message.text:
            message_lower = message.text.lower()
            for term in blocked_terms:
                if term in message_lower:
                    # LLMを呼び出さずにレスポンスを上書きします。
                    context.result = ChatResponse(
                        messages=[
                            ChatMessage(
                                role=Role.ASSISTANT,
                                text=(
                                    "I cannot process requests containing sensitive information. "
                                    "Please rephrase your question without including passwords, secrets, "
                                    "or other sensitive data."
                                ),
                            )
                        ]
                    )
                    return

    await next(context)


@function_middleware
async def atlantis_location_filter_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    """Atlantis向けの天気リクエストをブロックするFunction middleware。"""
    # locationパラメータが"atlantis"かどうかをチェックします。
    location = getattr(context.arguments, "location", None)
    if location and location.lower() == "atlantis":
        context.result = (
            "Blocked! Hold up right there!! Tell the user that "
            "'Atlantis is a special place, we must never ask about the weather there!!'"
        )
        context.terminate = True
        return

    await next(context)


def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """指定された場所の天気を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    temperature = 53
    return f"The weather in {location} is {conditions[0]} with a high of {temperature}°C."


def get_forecast(
    location: Annotated[str, "The location to get the forecast for."],
    days: Annotated[int, "Number of days for forecast"] = 3,
) -> str:
    """複数日の天気予報を取得します。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    forecast: list[str] = []

    for day in range(1, days + 1):
        condition = conditions[0]
        temp = 53
        forecast.append(f"Day {day}: {condition}, {temp}°C")

    return f"Weather forecast for {location}:\n" + "\n".join(forecast)


# Agent Frameworkの規約に従ったAgentインスタンス。
agent = ChatAgent(
    name="AzureWeatherAgent",
    description="A helpful agent that provides weather information and forecasts",
    instructions="""
    You are a weather assistant. You can provide current weather information
    and forecasts for any location. Always be helpful and provide detailed
    weather information when asked.
    """,
    chat_client=AzureOpenAIChatClient(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    ),
    tools=[get_weather, get_forecast],
    middleware=[security_filter_middleware, atlantis_location_filter_middleware],
)


def main():
    """DevUIでAzure天気Agentを起動します。"""
    import logging

    from agent_framework.devui import serve

    # ログのセットアップ
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Azure Weather Agent")
    logger.info("Available at: http://localhost:8090")
    logger.info("Entity ID: agent_AzureWeatherAgent")

    # Agentでサーバーを起動します。
    serve(entities=[agent], port=8090, auto_open=True)


if __name__ == "__main__":
    main()
