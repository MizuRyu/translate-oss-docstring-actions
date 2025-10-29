# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, get_logger, handler
from agent_framework.observability import setup_observability

"""Basic tracing workflow sample.

Sample: Workflow Tracing basics

A minimal two executor workflow demonstrates built in OpenTelemetry spans when diagnostics are enabled.
The sample raises an error if tracing is not configured.

Purpose:
- Require diagnostics by checking ENABLE_OTEL and wiring a console exporter.
- Show the span categories produced by a simple graph:
  - workflow.build (events: build.started, build.validation_completed, build.completed, edge_group.process)
  - workflow.run (events: workflow.started, workflow.completed or workflow.error)
  - executor.process (for each executor invocation)
  - message.send (for each outbound message)
- Provide a tiny flow that is easy to run and reason about: uppercase then print.

Prerequisites:
- No external services required for the workflow itself.
"""

logger = get_logger()


class StartExecutor(Executor):
    @handler  # type: ignore[misc]
    async def handle_input(self, message: str, ctx: WorkflowContext[str]) -> None:
        # 変換して下流に転送します。これによりexecutor.processとmessage.sendのスパンが生成されます。
        await ctx.send_message(message.upper())


class EndExecutor(Executor):
    @handler  # type: ignore[misc]
    async def handle_final(self, message: str, ctx: WorkflowContext) -> None:
        # Sink executor。ワークフローはアイドル状態で保留作業がないと完了します。
        print(f"Final result: {message}")


async def main() -> None:
    # トレーシングを有効にし、環境変数に基づいて必要なトレーシング、ロギング、メトリクスプロバイダーを作成します。
    setup_observability()

    # 2ノードのグラフを構築します：StartExecutor -> EndExecutor。ビルダーはworkflow.buildスパンを発行します。
    workflow = (
        WorkflowBuilder()
        .add_edge(StartExecutor(id="start"), EndExecutor(id="end"))
        .set_start_executor("start")  # set_start_executor accepts an executor id string or the instance
        .build()
    )  # ここでworkflow.buildスパンが発行されます。

    # 単純なペイロードで一度実行します。workflow.runとexecutorおよびmessageのスパンが表示されるはずです。
    await workflow.run("hello tracing")  # workflow.run + executor.processとmessage.sendのスパン


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
