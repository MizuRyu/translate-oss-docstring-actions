# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

import dotenv
from agent_framework import HostedCodeInterpreterTool
from agent_framework.azure import AzureAIAgentClient
from agent_framework.observability import get_tracer
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import format_trace_id
from pydantic import Field

"""
This sample, shows you can leverage the built-in telemetry in Azure AI.
It uses the Azure AI client to setup the telemetry, this calls out to
Azure AI for the connection string of the attached Application Insights
instance.

You must add an Application Insights instance to your Azure AI project
for this sample to work.
"""

# `AZURE_AI_PROJECT_ENDPOINT` 環境変数の読み込み用
dotenv.load_dotenv()

# 青色で印刷し、印刷後にリセットするANSIカラーコード
BLUE = "\x1b[34m"
RESET = "\x1b[0m"


async def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    await asyncio.sleep(randint(0, 10) / 10.0)  # ネットワークコールをシミュレートします
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    """AIサービスを実行します。

    この関数はAIサービスを実行し、出力を表示します。
    バックグラウンドでサービス実行のテレメトリが収集され、
    トレースは設定されたテレメトリバックエンドに送信されます。

    テレメトリにはAIサービス実行に関する情報が含まれます。

    azure_aiでは、Azure AI実装によって呼び出される特定の操作（例: `create_agent`）も確認できます。

    """
    questions = [
        "What's the weather in Amsterdam and in Paris?",
        "Why is the sky blue?",
        "Tell me about AI.",
        "Can you write a python function that adds two numbers? and use it to add 8483 and 5692?",
    ]
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential) as project,
        AzureAIAgentClient(project_client=project) as client,
    ):
        # トレーシングを有効にし、アプリケーションがAzure AIプロジェクトに紐づくApplication
        # Insightsインスタンスにテレメトリデータを送信するように設定します。 既存の設定は上書きされます。
        await client.setup_azure_ai_observability()

        with get_tracer().start_as_current_span(
            name="Foundry Telemetry from Agent Framework", kind=SpanKind.CLIENT
        ) as current_span:
            print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")

            for question in questions:
                print(f"{BLUE}User: {question}{RESET}")
                print(f"{BLUE}Assistant: {RESET}", end="")
                async for chunk in client.get_streaming_response(
                    question, tools=[get_weather, HostedCodeInterpreterTool()]
                ):
                    if str(chunk):
                        print(f"{BLUE}{str(chunk)}{RESET}", end="")
                print(f"{BLUE}{RESET}")

            print(f"{BLUE}Done{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
