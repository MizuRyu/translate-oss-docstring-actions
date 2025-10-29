# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    ExecutorFailedEvent,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowFailedEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
    handler,
)
from agent_framework._workflows._events import WorkflowOutputEvent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from typing_extensions import Never

"""
Step 3: Agents in a workflow with streaming

A Writer agent generates content,
then passes the conversation to a Reviewer agent that finalizes the result.
The workflow is invoked with run_stream so you can observe events as they occur.

Purpose:
Show how to wrap chat agents created by AzureOpenAIChatClient inside workflow executors, wire them with WorkflowBuilder,
and consume streaming events from the workflow. Demonstrate the @handler pattern with typed inputs and typed
WorkflowContext[T_Out, T_W_Out] outputs. Agents automatically yield outputs when they complete.
The streaming loop also surfaces WorkflowEvent.origin so you can distinguish runner-generated lifecycle events
from executor-generated data-plane events.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Basic familiarity with WorkflowBuilder, executors, edges, events, and streaming runs.
"""


class Writer(Executor):
    """コンテンツ生成のためのドメイン固有Agentを所有するカスタムexecutor。

    このクラスは以下を示します:
    - ChatAgentをExecutorにアタッチし、ワークフローのノードとして参加させる方法。
    - @handlerメソッドを使い、型付き入力を受け取りctx.send_messageで型付き出力を転送する方法。

    """

    agent: ChatAgent

    def __init__(self, chat_client: AzureOpenAIChatClient, id: str = "writer"):
        # 設定済みのAzureOpenAIChatClientを使ってドメイン固有Agentを作成します。
        self.agent = chat_client.create_agent(
            instructions=(
                "You are an excellent content writer. You create new content and edit contents based on the feedback."
            ),
        )
        # このAgentをexecutorノードに関連付けます。基底Executorはself.agentに保存します。
        super().__init__(id=id)

    @handler
    async def handle(self, message: ChatMessage, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        """コンテンツを生成し、更新された会話を転送します。

        このhandlerの契約:
        - messageは受信したユーザーのChatMessageです。
        - ctxは下流にlist[ChatMessage]を送信することを期待するWorkflowContextです。

        ここで示すパターン:
        1) 受信メッセージで会話を初期化します。
        2) アタッチされたAgentを実行してアシスタントメッセージを生成します。
        3) 累積したメッセージをctx.send_messageで次のexecutorに転送します。

        """
        # 受信したユーザーメッセージで会話を開始します。
        messages: list[ChatMessage] = [message]
        # Agentを実行し、Agentのメッセージで会話を拡張します。
        response = await self.agent.run(messages)
        messages.extend(response.messages)
        # 累積したメッセージをワークフロー内の次のexecutorに転送します。
        await ctx.send_message(messages)


class Reviewer(Executor):
    """レビューAgentを所有しワークフローを完了するカスタムexecutor。"""

    agent: ChatAgent

    def __init__(self, chat_client: AzureOpenAIChatClient, id: str = "reviewer"):
        # コンテンツを評価・改善するドメイン固有Agentを作成します。
        self.agent = chat_client.create_agent(
            instructions=(
                "You are an excellent content reviewer. You review the content and provide feedback to the writer."
            ),
        )
        super().__init__(id=id)

    @handler
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[Never, str]) -> None:
        """会話の全トランスクリプトをレビューし最終出力をyieldします。

        このノードはこれまでのすべてのメッセージを消費します。Agentを使って最終テキストを生成し、
        出力をyieldします。ワークフローはアイドル状態になると完了します。

        """
        response = await self.agent.run(messages)
        await ctx.yield_output(response.text)


async def main():
    """2ノードのワークフローを構築し、ストリーミングでイベントを観察しながら実行します。"""
    # Azureチャットクライアントを作成します。AzureCliCredentialは現在のaz loginを使用します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    # 2つのAgentバックのexecutorをインスタンス化します。
    writer = Writer(chat_client)
    reviewer = Reviewer(chat_client)

    # フルーエントビルダーを使ってワークフローを構築します。 開始ノードを設定し、writerからreviewerへのエッジを接続します。
    workflow = WorkflowBuilder().set_start_executor(writer).add_edge(writer, reviewer).build()

    # ユーザーの初期メッセージでワークフローを実行し、発生するイベントをストリームします。
    # これによりexecutorイベント、ワークフロー出力、run状態の変化、エラーが表面化します。
    async for event in workflow.run_stream(
        ChatMessage(role="user", text="Create a slogan for a new electric SUV that is affordable and fun to drive.")
    ):
        if isinstance(event, WorkflowStatusEvent):
            prefix = f"State ({event.origin.value}): "
            if event.state == WorkflowRunState.IN_PROGRESS:
                print(prefix + "IN_PROGRESS")
            elif event.state == WorkflowRunState.IN_PROGRESS_PENDING_REQUESTS:
                print(prefix + "IN_PROGRESS_PENDING_REQUESTS (requests in flight)")
            elif event.state == WorkflowRunState.IDLE:
                print(prefix + "IDLE (no active work)")
            elif event.state == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
                print(prefix + "IDLE_WITH_PENDING_REQUESTS (prompt user or UI now)")
            else:
                print(prefix + str(event.state))
        elif isinstance(event, WorkflowOutputEvent):
            print(f"Workflow output ({event.origin.value}): {event.data}")
        elif isinstance(event, ExecutorFailedEvent):
            print(
                f"Executor failed ({event.origin.value}): "
                f"{event.executor_id} {event.details.error_type}: {event.details.message}"
            )
        elif isinstance(event, WorkflowFailedEvent):
            details = event.details
            print(f"Workflow failed ({event.origin.value}): {details.error_type}: {details.message}")
        else:
            print(f"{event.__class__.__name__} ({event.origin.value}): {event}")

    """
    Sample Output:

    State (RUNNER): IN_PROGRESS
    ExecutorInvokeEvent (RUNNER): ExecutorInvokeEvent(executor_id=writer)
    ExecutorCompletedEvent (RUNNER): ExecutorCompletedEvent(executor_id=writer)
    ExecutorInvokeEvent (RUNNER): ExecutorInvokeEvent(executor_id=reviewer)
    Workflow output (EXECUTOR): Drive the Future. Affordable Adventure, Electrified.
    ExecutorCompletedEvent (RUNNER): ExecutorCompletedEvent(executor_id=reviewer)
    State (RUNNER): IDLE
    """


if __name__ == "__main__":
    asyncio.run(main())
