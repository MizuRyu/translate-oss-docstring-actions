# Copyright (c) Microsoft. All rights reserved.

import argparse
import asyncio
from contextlib import suppress
from random import randint
from typing import TYPE_CHECKING, Annotated, Literal

from agent_framework import ai_function
from agent_framework.observability import get_tracer, setup_observability
from agent_framework.openai import OpenAIResponsesClient
from opentelemetry import trace
from opentelemetry.trace.span import format_trace_id
from pydantic import Field

if TYPE_CHECKING:
    from agent_framework import ChatClientProtocol

"""
This sample, show how you can configure observability of an application via the
`setup_observability` function with environment variables.

When you run this sample with an OTLP endpoint or an Application Insights connection string,
you should see traces, logs, and metrics in the configured backend.

If no OTLP endpoint or Application Insights connection string is configured, the sample will
output traces, logs, and metrics to the console.
"""

# SDKで収集されたテレメトリデータを表示するために実行可能なシナリオを定義します
SCENARIOS = ["chat_client", "chat_client_stream", "ai_function", "all"]


async def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    await asyncio.sleep(randint(0, 10) / 10.0)  # ネットワークコールをシミュレートします
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def run_chat_client(client: "ChatClientProtocol", stream: bool = False) -> None:
    """AIサービスを実行します。

    この関数はAIサービスを実行し、出力を表示します。
    バックグラウンドでサービス実行のテレメトリが収集され、
    トレースは設定されたテレメトリバックエンドに送信されます。

    テレメトリにはAIサービス実行に関する情報が含まれます。

    Args:
        client: 使用するchat client。
        stream: レスポンスにストリーミングを使用するかどうか

    Remarks:
        以下のシナリオでは、次の内容が確認できるはずです:
        1つのClient spanに4つの子スパンが存在:
            2つの内部スパンはgen_ai.operation.name=chat
                1つ目はfinish_reasonが "tool_calls"
                2つ目はfinish_reasonが "stop"
            2つの内部スパンはgen_ai.operation.name=execute_tool


    """
    scenario_name = "Chat Client Stream" if stream else "Chat Client"
    with get_tracer().start_as_current_span(name=f"Scenario: {scenario_name}", kind=trace.SpanKind.CLIENT):
        print("Running scenario:", scenario_name)
        message = "What's the weather in Amsterdam and in Paris?"
        print(f"User: {message}")
        if stream:
            print("Assistant: ", end="")
            async for chunk in client.get_streaming_response(message, tools=get_weather):
                if str(chunk):
                    print(str(chunk), end="")
            print("")
        else:
            response = await client.get_response(message, tools=get_weather)
            print(f"Assistant: {response}")


async def run_ai_function() -> None:
    """AI関数を実行します。

    この関数はAI関数を実行し、出力を表示します。
    バックグラウンドで関数実行のテレメトリが収集され、
    トレースは設定されたテレメトリバックエンドに送信されます。

    テレメトリにはAI関数実行およびAIサービス実行に関する情報が含まれます。

    """
    with get_tracer().start_as_current_span("Scenario: AI Function", kind=trace.SpanKind.CLIENT):
        print("Running scenario: AI Function")
        func = ai_function(get_weather)
        weather = await func.invoke(location="Amsterdam")
        print(f"Weather in Amsterdam:\n{weather}")


async def main(scenario: Literal["chat_client", "chat_client_stream", "ai_function", "all"] = "all"):
    """選択されたシナリオを実行します。"""

    # トレーシングを有効にし、環境変数に基づいて必要なトレーシング、ロギング、メトリクスプロバイダーを作成します。 利用可能な設定オプションは .env.example
    # ファイルを参照してください。
    setup_observability()

    with get_tracer().start_as_current_span("Sample Scenario's", kind=trace.SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")

        client = OpenAIResponsesClient()

        # SDKでテレメトリが収集されるシナリオ、基本的なものから複雑なものまで。
        if scenario == "ai_function" or scenario == "all":
            with suppress(Exception):
                await run_ai_function()
        if scenario == "chat_client_stream" or scenario == "all":
            with suppress(Exception):
                await run_chat_client(client, stream=True)
        if scenario == "chat_client" or scenario == "all":
            with suppress(Exception):
                await run_chat_client(client, stream=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIOS,
        default="all",
        help="The scenario to run. Default is all.",
    )

    args = arg_parser.parse_args()
    asyncio.run(main(args.scenario))
