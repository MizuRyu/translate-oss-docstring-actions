# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.observability import get_tracer, setup_observability
from agent_framework.openai import OpenAIChatClient
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import format_trace_id
from pydantic import Field

"""
This sample shows how you can observe an agent in Agent Framework by using the
same observability setup function.
"""


async def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    await asyncio.sleep(randint(0, 10) / 10.0)  # ネットワークコールをシミュレートする
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main():
    # トレーシングを有効にし、環境変数に基づいて必要なトレーシング、ロギング、メトリクスプロバイダーを作成します。
    # 利用可能な設定オプションについては.env.exampleファイルを参照してください。
    setup_observability()

    questions = ["What's the weather in Amsterdam?", "and in Paris, and which is better?", "Why is the sky blue?"]

    with get_tracer().start_as_current_span("Scenario: Agent Chat", kind=SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")

        agent = ChatAgent(
            chat_client=OpenAIChatClient(),
            tools=get_weather,
            name="WeatherAgent",
            instructions="You are a weather assistant.",
        )
        thread = agent.get_new_thread()
        for question in questions:
            print(f"User: {question}")
            print(f"{agent.display_name}: ", end="")
            async for update in agent.run_stream(
                question,
                thread=thread,
            ):
                if update.text:
                    print(update.text, end="")


if __name__ == "__main__":
    asyncio.run(main())
