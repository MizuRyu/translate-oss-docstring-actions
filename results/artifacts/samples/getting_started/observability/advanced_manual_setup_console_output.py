# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from random import randint
from typing import Annotated

from agent_framework.openai import OpenAIChatClient
from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv._incubating.attributes.service_attributes import SERVICE_NAME
from opentelemetry.trace import set_tracer_provider
from pydantic import Field

"""
This sample shows how to manually configure to send traces, logs, and metrics to the console,
without using the `setup_observability` helper function.
"""

resource = Resource.create({SERVICE_NAME: "ManualSetup"})


def setup_logging():
    # アプリケーション用のグローバルロガープロバイダーを作成して設定する。
    logger_provider = LoggerProvider(resource=resource)
    # ログプロセッサはエクスポーターで初期化されます。エクスポーターは責任を持ちます
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))
    # グローバルデフォルトのロガープロバイダーを設定する
    set_logger_provider(logger_provider)
    # エクスポーターにOTLP形式でログレコードを書き込むためのロギングハンドラーを作成する。
    handler = LoggingHandler()
    # ハンドラーをルートロガーにアタッチします。引数なしの`getLogger()`はルートロガーを返します。
    # すべての子ロガーからのイベントはこのハンドラーで処理されます。
    logger = logging.getLogger()
    logger.addHandler(handler)
    # ハンドラーで処理されるすべてのレコードを許可するためにログレベルをNOTSETに設定する。
    logger.setLevel(logging.NOTSET)


def setup_tracing():
    # アプリケーション用のトレースプロバイダーを初期化する。これはトレーサーを作成するためのファクトリーです。
    tracer_provider = TracerProvider(resource=resource)
    # スパンプロセッサはエクスポーターで初期化されます。 エクスポーターはテレメトリデータを特定のバックエンドに送信する責任があります。
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    # グローバルデフォルトのトレースプロバイダーを設定する
    set_tracer_provider(tracer_provider)


def setup_metrics():
    # アプリケーション用のメトリックプロバイダーを初期化する。これはメーターを作成するためのファクトリーです。
    meter_provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5000)],
        resource=resource,
    )
    # グローバルデフォルトのメータープロバイダーを設定する
    set_meter_provider(meter_provider)


async def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    await asyncio.sleep(randint(0, 10) / 10.0)  # ネットワークコールをシミュレートする
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def run_chat_client() -> None:
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
    client = OpenAIChatClient()
    message = "What's the weather in Amsterdam and in Paris?"
    print(f"User: {message}")
    print("Assistant: ", end="")
    async for chunk in client.get_streaming_response(message, tools=get_weather):
        if str(chunk):
            print(str(chunk), end="")
    print("")


async def main():
    """選択したシナリオを実行します。"""
    setup_logging()
    setup_tracing()
    setup_metrics()

    await run_chat_client()


if __name__ == "__main__":
    asyncio.run(main())
