# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""
Step 2: Agents in a Workflow non-streaming

This sample uses two custom executors. A Writer agent creates or edits content,
then hands the conversation to a Reviewer agent which evaluates and finalizes the result.

Purpose:
Show how to wrap chat agents created by AzureOpenAIChatClient inside workflow executors. Demonstrate the @handler pattern
with typed inputs and typed WorkflowContext[T] outputs, connect executors with the fluent WorkflowBuilder, and finish
by yielding outputs from the terminal node.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Basic familiarity with WorkflowBuilder, executors, edges, events, and streaming or non streaming runs.
"""


class Writer(Executor):
    """コンテンツ生成を担当するドメイン固有Agentを所有するカスタムexecutor。

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
    async def handle(self, message: ChatMessage, ctx: WorkflowContext[list[ChatMessage], str]) -> None:
        """Agentを使ってコンテンツを生成し、更新された会話を転送します。

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
    """レビューAgentを所有しワークフローを完了するカスタムexecutor。

    このクラスは以下を示します:
    - 上流で生成された型付きペイロードを消費すること。
    - 最終テキスト結果をyieldしてワークフローを完了させること。

    """

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
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage], str]) -> None:
        """会話の全トランスクリプトをレビューし最終文字列で完了します。

        このノードはこれまでのすべてのメッセージを消費します。Agentを使って最終テキストを生成し、
        出力をyieldして完了を通知します。

        """
        response = await self.agent.run(messages)
        await ctx.yield_output(response.text)


async def main():
    """シンプルな2ノードAgentワークフローを構築し実行します: Writer から Reviewer へ。"""
    # Azureチャットクライアントを作成します。AzureCliCredentialは現在のaz loginを使用します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    # 2つのAgentバックのexecutorをインスタンス化します。
    writer = Writer(chat_client)
    reviewer = Reviewer(chat_client)

    # フルーエントビルダーを使ってワークフローを構築します。 開始ノードを設定し、writerからreviewerへのエッジを接続します。
    workflow = WorkflowBuilder().set_start_executor(writer).add_edge(writer, reviewer).build()

    # ユーザーの初期メッセージでワークフローを実行します。 基礎的な明確さのために、run（非ストリーミング）を使用し、ワークフローの出力を表示してください。
    events = await workflow.run(
        ChatMessage(role="user", text="Create a slogan for a new electric SUV that is affordable and fun to drive.")
    )
    # 終端ノードは出力を生成します。その内容を表示してください。
    outputs = events.get_outputs()
    if outputs:
        print(outputs[-1])


if __name__ == "__main__":
    asyncio.run(main())
