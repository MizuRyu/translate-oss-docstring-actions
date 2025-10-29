# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import AgentRunUpdateEvent, WorkflowBuilder, WorkflowOutputEvent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""
Sample: Agents in a workflow with streaming

A Writer agent generates content, then a Reviewer agent critiques it.
The workflow uses streaming so you can observe incremental AgentRunUpdateEvent chunks as each agent produces tokens.

Purpose:
Show how to wire chat agents into a WorkflowBuilder pipeline by adding agents directly as edges.

Demonstrate:
- Automatic streaming of agent deltas via AgentRunUpdateEvent when using run_stream().
- Agents adapt to workflow mode: run_stream() emits incremental updates, run() emits complete responses.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Basic familiarity with WorkflowBuilder, edges, events, and streaming runs.
"""


async def main():
    """シンプルな2ノードAgentワークフローを構築し実行します: Writer から Reviewer へ。"""
    # Azureチャットクライアントを作成します。AzureCliCredentialは現在のaz loginを使用します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    # 2つのドメイン固有チャットAgentを定義します。
    writer_agent = chat_client.create_agent(
        instructions=(
            "You are an excellent content writer. You create new content and edit contents based on the feedback."
        ),
        name="writer_agent",
    )

    reviewer_agent = chat_client.create_agent(
        instructions=(
            "You are an excellent content reviewer."
            "Provide actionable feedback to the writer about the provided content."
            "Provide the feedback in the most concise manner possible."
        ),
        name="reviewer_agent",
    )

    # フルーエントビルダーを使ってワークフローを構築します。 開始ノードを設定し、writerからreviewerへのエッジを接続します。
    # Agentはワークフローモードに適応します: run_stream()は増分更新用、run()は完全なレスポンス用。
    workflow = (
        WorkflowBuilder()
        .set_start_executor(writer_agent)
        .add_edge(writer_agent, reviewer_agent)
        .build()
    )

    # ワークフローからイベントをストリームします。各executorごとに部分的なトークン更新を集約して読みやすい出力にします。
    last_executor_id: str | None = None

    events = workflow.run_stream("Create a slogan for a new electric SUV that is affordable and fun to drive.")
    async for event in events:
        if isinstance(event, AgentRunUpdateEvent):
            # AgentRunUpdateEventは基盤となるAgentからの増分テキストデルタを含みます。
            # Executorが変わるとプレフィックスを表示し、同じ行に更新を追加します。
            eid = event.executor_id
            if eid != last_executor_id:
                if last_executor_id is not None:
                    print()
                print(f"{eid}:", end=" ", flush=True)
                last_executor_id = eid
            print(event.data, end="", flush=True)
        elif isinstance(event, WorkflowOutputEvent):
            print("\n===== Final output =====")
            print(event.data)

    """
    Sample Output:

    writer_agent: Charge Up Your Journey. Fun, Affordable, Electric.
    reviewer_agent: Clear message, but consider highlighting SUV specific benefits (space, versatility) for stronger
        impact. Try more vivid language to evoke excitement. Example: "Big on Space. Big on Fun. Electric for Everyone."
    ===== Final Output =====
    Clear message, but consider highlighting SUV specific benefits (space, versatility) for stronger impact. Try more
        vivid language to evoke excitement. Example: "Big on Space. Big on Fun. Electric for Everyone."
    """


if __name__ == "__main__":
    asyncio.run(main())
