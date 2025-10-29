# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
)
from agent_framework.observability import get_tracer, setup_observability
from opentelemetry.trace import SpanKind
from opentelemetry.trace.span import format_trace_id
from typing_extensions import Never

"""
This sample shows the telemetry collected when running a Agent Framework workflow.

Telemetry data that the workflow system emits includes:
- Overall workflow build & execution spans
- Individual executor processing spans
- Message publishing between executors
"""


# シーケンシャルワークフロー用のexecutors
class UpperCaseExecutor(Executor):
    """テキストを大文字に変換するexecutor。"""

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """入力文字列を大文字に変換してタスクを実行します。"""
        print(f"UpperCaseExecutor: Processing '{text}'")
        result = text.upper()
        print(f"UpperCaseExecutor: Result '{result}'")

        # 結果をワークフロー内の次のexecutorに送信します。
        await ctx.send_message(result)


class ReverseTextExecutor(Executor):
    """テキストを逆順にするexecutor。"""

    @handler
    async def reverse_text(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        """入力文字列を逆順にしてタスクを実行します。"""
        print(f"ReverseTextExecutor: Processing '{text}'")
        result = text[::-1]
        print(f"ReverseTextExecutor: Result '{result}'")

        # 出力をyieldします。
        await ctx.yield_output(result)


async def run_sequential_workflow() -> None:
    """テレメトリ収集を示すシンプルなシーケンシャルワークフローを実行します。

    このワークフローは2つのexecutorを順に処理します:
    1. UpperCaseExecutorが入力を大文字に変換
    2. ReverseTextExecutorが文字列を逆順にし、ワークフローを完了

    """
    # ステップ1: executorを作成します。
    upper_case_executor = UpperCaseExecutor(id="upper_case_executor")
    reverse_text_executor = ReverseTextExecutor(id="reverse_text_executor")

    # ステップ2: 定義されたエッジでワークフローを構築します。
    workflow = (
        WorkflowBuilder()
        .add_edge(upper_case_executor, reverse_text_executor)
        .set_start_executor(upper_case_executor)
        .build()
    )

    # ステップ3: 初期メッセージでワークフローを実行します。
    input_text = "hello world"
    print(f"Starting workflow with input: '{input_text}'")

    output_event = None
    async for event in workflow.run_stream("Hello world"):
        if isinstance(event, WorkflowOutputEvent):
            # WorkflowOutputEventには最終結果が含まれます。
            output_event = event

    if output_event:
        print(f"Workflow completed with result: '{output_event.data}'")


async def main():
    """シンプルなシーケンシャルワークフローでテレメトリサンプルを実行します。"""
    # トレーシングを有効にし、環境変数に基づいて必要なトレーシング、ロギング、メトリクスプロバイダーを作成します。 利用可能な設定オプションは .env.example
    # ファイルを参照してください。
    setup_observability()

    with get_tracer().start_as_current_span("Sequential Workflow Scenario", kind=SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")

        # シーケンシャルワークフローシナリオを実行します。
        await run_sequential_workflow()


if __name__ == "__main__":
    asyncio.run(main())
