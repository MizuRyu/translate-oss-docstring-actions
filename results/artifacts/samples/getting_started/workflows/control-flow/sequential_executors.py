# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import cast

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
)
from typing_extensions import Never

"""
Sample: Sequential workflow with streaming.

Two custom executors run in sequence. The first converts text to uppercase,
the second reverses the text and completes the workflow. The run_stream loop prints events as they occur.

Purpose:
Show how to define explicit Executor classes with @handler methods, wire them in order with
WorkflowBuilder, and consume streaming events. Demonstrate typed WorkflowContext[T_Out, T_W_Out] for outputs,
ctx.send_message to pass intermediate values, and ctx.yield_output to provide workflow outputs.

Prerequisites:
- No external services required.
"""


class UpperCaseExecutor(Executor):
    """入力文字列を大文字に変換して転送します。

    コンセプト:
    - @handlerメソッドは呼び出し可能なステップを定義します。
    - WorkflowContext[str]はこのステップが次のノードに文字列を出力することを示します。

    """

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """入力を大文字に変換して下流に送ります。"""
        result = text.upper()
        # 中間結果をチェーン内の次のexecutorに渡します。
        await ctx.send_message(result)


class ReverseTextExecutor(Executor):
    """受け取った文字列を逆順にしてワークフロー出力を生成します。

    コンセプト:
    - ctx.yield_outputを使って端末結果が準備できたときにワークフロー出力を提供します。
    - 端末ノードはメッセージをさらに転送しません。

    """

    @handler
    async def reverse_text(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
        """入力文字列を逆順にしてワークフロー出力を生成します。"""
        result = text[::-1]
        await ctx.yield_output(result)


async def main() -> None:
    """2ステップのシーケンシャルなワークフローを構築し、ストリーミングで実行してイベントを観察します。"""
    # ステップ1: executorインスタンスを作成します。
    upper_case_executor = UpperCaseExecutor(id="upper_case_executor")
    reverse_text_executor = ReverseTextExecutor(id="reverse_text_executor")

    # ステップ2: ワークフローグラフを構築します。 順序が重要です。upper_case_executor ->
    # reverse_text_executorを接続し、開始を設定します。
    workflow = (
        WorkflowBuilder()
        .add_edge(upper_case_executor, reverse_text_executor)
        .set_start_executor(upper_case_executor)
        .build()
    )

    # ステップ3: 単一の入力に対してイベントをストリーミングします。 ストリームにはexecutorの呼び出しと完了イベント、さらにワークフローの出力が含まれます。
    outputs: list[str] = []
    async for event in workflow.run_stream("hello world"):
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            outputs.append(cast(str, event.data))

    if outputs:
        print(f"Workflow outputs: {outputs}")


if __name__ == "__main__":
    asyncio.run(main())
