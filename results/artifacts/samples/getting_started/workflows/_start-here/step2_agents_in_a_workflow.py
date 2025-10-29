# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import AgentRunEvent, WorkflowBuilder
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""
Step 2: Agents in a Workflow non-streaming

This sample uses two custom executors. A Writer agent creates or edits content,
then hands the conversation to a Reviewer agent which evaluates and finalizes the result.

Purpose:
Show how to wrap chat agents created by AzureOpenAIChatClient inside workflow executors. Demonstrate how agents
automatically yield outputs when they complete, removing the need for explicit completion events.
The workflow completes when it becomes idle.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Basic familiarity with WorkflowBuilder, executors, edges, events, and streaming or non streaming runs.
"""


async def main():
    """シンプルな2ノードAgentワークフローを構築し実行します: Writer から Reviewer へ。"""
    # Azureチャットクライアントを作成します。AzureCliCredentialは現在のaz loginを使用します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    writer_agent = chat_client.create_agent(
        instructions=(
            "You are an excellent content writer. You create new content and edit contents based on the feedback."
        ),
        name="writer",
    )

    reviewer_agent = chat_client.create_agent(
        instructions=(
            "You are an excellent content reviewer."
            "Provide actionable feedback to the writer about the provided content."
            "Provide the feedback in the most concise manner possible."
        ),
        name="reviewer",
    )

    # フルーエントビルダーを使ってワークフローを構築します。 開始ノードを設定し、writerからreviewerへのエッジを接続します。
    workflow = WorkflowBuilder().set_start_executor(writer_agent).add_edge(writer_agent, reviewer_agent).build()

    # ユーザーの初期メッセージでワークフローを実行します。 基礎的な明確さのために、run（非ストリーミング）を使い、終端イベントを出力します。
    events = await workflow.run("Create a slogan for a new electric SUV that is affordable and fun to drive.")
    # Agentの実行イベントと最終出力を表示します
    for event in events:
        if isinstance(event, AgentRunEvent):
            print(f"{event.executor_id}: {event.data}")

    print(f"{'=' * 60}\nWorkflow Outputs: {events.get_outputs()}")
    # 最終的なrun状態を要約します（例: COMPLETED）
    print("Final state:", events.get_final_state())

    """
    Sample Output:

    writer: "Charge Up Your Adventure—Affordable Fun, Electrified!"
    reviewer: Slogan: "Plug Into Fun—Affordable Adventure, Electrified."

    **Feedback:**
    - Clear focus on affordability and enjoyment.
    - "Plug into fun" connects emotionally and highlights electric nature.
    - Consider specifying "SUV" for clarity in some uses.
    - Strong, upbeat tone suitable for marketing.
    ============================================================
    Workflow Outputs: ['Slogan: "Plug Into Fun—Affordable Adventure, Electrified."

    **Feedback:**
    - Clear focus on affordability and enjoyment.
    - "Plug into fun" connects emotionally and highlights electric nature.
    - Consider specifying "SUV" for clarity in some uses.
    - Strong, upbeat tone suitable for marketing.']
    """


if __name__ == "__main__":
    asyncio.run(main())
