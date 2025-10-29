# Copyright (c) Microsoft. All rights reserved.

import asyncio
import random

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, WorkflowOutputEvent, handler
from typing_extensions import Never

"""
Sample: Concurrent fan out and fan in with two different tasks that output results of different types.

Purpose:
Show how to construct a parallel branch pattern in workflows. Demonstrate:
- Fan out by targeting multiple executors from one dispatcher.
- Fan in by collecting a list of results from the executors.
- Simple tracing using AgentRunEvent to observe execution order and progress.

Prerequisites:
- Familiarity with WorkflowBuilder, executors, edges, events, and streaming runs.
"""


class Dispatcher(Executor):
    """
    このデコレータの唯一の目的は、ワークフローの入力を
    他のexecutorにディスパッチすることです。

    """

    @handler
    async def handle(self, numbers: list[int], ctx: WorkflowContext[list[int]]):
        if not numbers:
            raise RuntimeError("Input must be a valid list of integers.")

        await ctx.send_message(numbers)


class Average(Executor):
    """整数のリストの平均を計算します。"""

    @handler
    async def handle(self, numbers: list[int], ctx: WorkflowContext[float]):
        average: float = sum(numbers) / len(numbers)
        await ctx.send_message(average)


class Sum(Executor):
    """整数のリストの合計を計算します。"""

    @handler
    async def handle(self, numbers: list[int], ctx: WorkflowContext[int]):
        total: int = sum(numbers)
        await ctx.send_message(total)


class Aggregator(Executor):
    """異なるタスクからの結果を集約し、最終出力をyieldします。"""

    @handler
    async def handle(self, results: list[int | float], ctx: WorkflowContext[Never, list[int | float]]):
        """ソースexecutorから結果を受け取ります。

        フレームワークはソースexecutorからのメッセージを自動的に収集し、リストとして配信します。

        Args:
            results (list[int | float]): 上流executorからの実行結果。
                型注釈は上流executorが生成するユニオン型のリストでなければなりません。
            ctx (WorkflowContext[Never, list[int | float]]): 最終出力をyieldできるワークフローコンテキスト。

        """
        await ctx.yield_output(results)


async def main() -> None:
    # 1) executorを作成します
    dispatcher = Dispatcher(id="dispatcher")
    average = Average(id="average")
    summation = Sum(id="summation")
    aggregator = Aggregator(id="aggregator")

    # 2) シンプルなfan outとfan inワークフローを構築します
    workflow = (
        WorkflowBuilder()
        .set_start_executor(dispatcher)
        .add_fan_out_edges(dispatcher, [average, summation])
        .add_fan_in_edges([average, summation], aggregator)
        .build()
    )

    # 3) ワークフローを実行します
    output: list[int | float] | None = None
    async for event in workflow.run_stream([random.randint(1, 100) for _ in range(10)]):
        if isinstance(event, WorkflowOutputEvent):
            output = event.data

    if output is not None:
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
