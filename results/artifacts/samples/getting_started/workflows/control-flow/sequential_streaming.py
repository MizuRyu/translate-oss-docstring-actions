# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import WorkflowBuilder, WorkflowContext, WorkflowOutputEvent, executor
from typing_extensions import Never

"""
Sample: Foundational sequential workflow with streaming using function-style executors.

Two lightweight steps run in order. The first converts text to uppercase.
The second reverses the text and yields the workflow output. Events are printed as they arrive from run_stream.

Purpose:
Show how to declare executors with the @executor decorator, connect them with WorkflowBuilder,
pass intermediate values using ctx.send_message, and yield final output using ctx.yield_output().
Demonstrate how streaming exposes ExecutorInvokedEvent and ExecutorCompletedEvent for observability.

Prerequisites:
- No external services required.
"""


# ステップ1: executorデコレーターを使ってメソッドを定義します。
@executor(id="upper_case_executor")
async def to_upper_case(text: str, ctx: WorkflowContext[str]) -> None:
    """入力を大文字に変換し、次のステップに渡します。

    Concepts:
    - @executorデコレーターはこの関数をワークフローノードとして登録します。
    - WorkflowContext[str]は、このノードが下流に文字列ペイロードを送出することを示します。

    """
    result = text.upper()

    # 中間結果をワークフローグラフ内の次のexecutorに送信します。
    await ctx.send_message(result)


@executor(id="reverse_text_executor")
async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """入力を反転し、ワークフローの出力をyieldします。

    Concepts:
    - 終端ノードはctx.yield_output()を使って出力をyieldします。
    - ワークフローはアイドル状態（作業がなくなる）になると完了します。

    """
    result = text[::-1]

    # このワークフロー実行の最終出力をyieldします。
    await ctx.yield_output(result)


async def main():
    """2ステップのシーケンシャルなワークフローを構築し、ストリーミングで実行してイベントを観察します。"""
    # ステップ2: 定義したエッジでワークフローを構築します。
    # 順序が重要です。upper_case_executorが最初に実行され、その後reverse_text_executorが実行されます。
    workflow = WorkflowBuilder().add_edge(to_upper_case, reverse_text).set_start_executor(to_upper_case).build()

    # ステップ3: ワークフローを実行し、リアルタイムでイベントをストリーミングします。
    async for event in workflow.run_stream("hello world"):
        # ワークフローの進行に伴い、executorの呼び出しと完了イベントが表示されます。
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            print(f"Workflow completed with result: {event.data}")

    """
    Sample Output:

    Event: ExecutorInvokedEvent(executor_id=upper_case_executor)
    Event: ExecutorCompletedEvent(executor_id=upper_case_executor)
    Event: ExecutorInvokedEvent(executor_id=reverse_text_executor)
    Event: ExecutorCompletedEvent(executor_id=reverse_text_executor)
    Event: WorkflowOutputEvent(data='DLROW OLLEH', source_executor_id=reverse_text_executor)
    Workflow completed with result: DLROW OLLEH
    """


if __name__ == "__main__":
    asyncio.run(main())
