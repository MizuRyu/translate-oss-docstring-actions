# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import TYPE_CHECKING, Annotated

from agent_framework.observability import get_tracer
from agent_framework.openai import OpenAIResponsesClient
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import format_trace_id
from pydantic import Field

if TYPE_CHECKING:
    from agent_framework import ChatClientProtocol


"""
This sample shows how you can configure observability of an application with zero code changes.
It relies on the OpenTelemetry auto-instrumentation capabilities, and the observability setup
is done via environment variables.

This sample requires the `APPLICATIONINSIGHTS_CONNECTION_STRING` environment variable to be set.

Run the sample with the following command:
```
uv run --env-file=.env opentelemetry-instrument python advanced_zero_code.py
```
"""


async def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    await asyncio.sleep(randint(0, 10) / 10.0)  # ネットワークコールをシミュレートする
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def run_chat_client(client: "ChatClientProtocol", stream: bool = False) -> None:
    """AIサービスを実行します。

    この関数はAIサービスを実行し、出力を表示します。
    サービス実行のテレメトリは裏で収集され、
    トレースは設定されたテレメトリバックエンドに送信されます。

    テレメトリにはAIサービス実行に関する情報が含まれます。

    Args:
        stream: プラグインでストリーミングを使用するかどうか

    Remarks:
        関数呼び出しがOpenTelemetryループの外にある場合、
        モデルへの各呼び出しは別々のスパンとして処理されます。
        一方、OpenTelemetryが最後に配置される場合、
        1つのスパンが表示され、1回以上の関数呼び出しラウンドを含むことがあります。

        以下のシナリオでは、次のように表示されるはずです：

        gen_ai.operation.name=chat の2つのスパン
            1つ目は finish_reason が "tool_calls"
            2つ目は finish_reason が "stop"
        gen_ai.operation.name=execute_tool の2つのスパン


    """
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


async def main() -> None:
    with get_tracer().start_as_current_span("Zero Code", kind=SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")

        client = OpenAIResponsesClient()

        await run_chat_client(client, stream=True)
        await run_chat_client(client, stream=False)


if __name__ == "__main__":
    asyncio.run(main())
