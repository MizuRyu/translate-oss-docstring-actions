# Copyright (c) Microsoft. All rights reserved.
"""AutoGen の Swarm パターンと Agent Framework の HandoffBuilder の比較。

エージェントがタスク要件に基づいて他の専門エージェントに制御を渡す
エージェントハンドオフの調整を示します。
"""

import asyncio


async def run_autogen() -> None:
    """AutoGen の Swarm パターンによる human-in-the-loop ハンドオフ。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
    from autogen_agentchat.messages import HandoffMessage
    from autogen_agentchat.teams import Swarm
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(model="gpt-4.1-mini")

    # 専門家にルーティングするトリアージエージェントを作成します
    triage_agent = AssistantAgent(
        name="triage",
        model_client=client,
        system_message=(
            "You are a triage agent. Analyze the user's request and hand off to the appropriate specialist.\n"
            "If you need information from the user, first send your message, then handoff to user.\n"
            "Use TERMINATE when the issue is fully resolved."
        ),
        handoffs=["billing_agent", "technical_support", "user"],
        model_client_stream=True,
    )

    # 請求専門家を作成します
    billing_agent = AssistantAgent(
        name="billing_agent",
        model_client=client,
        system_message=(
            "You are a billing specialist. Help with payment and billing questions.\n"
            "If you need information from the user, first send your message, then handoff to user.\n"
            "When the issue is resolved, handoff to triage to finalize."
        ),
        handoffs=["triage", "user"],
        model_client_stream=True,
    )

    # 技術サポート専門家を作成します
    tech_support = AssistantAgent(
        name="technical_support",
        model_client=client,
        system_message=(
            "You are technical support. Help with technical issues.\n"
            "If you need information from the user, first send your message, then handoff to user.\n"
            "When the issue is resolved, handoff to triage to finalize."
        ),
        handoffs=["triage", "user"],
        model_client_stream=True,
    )

    # human-in-the-loop 終了を持つ Swarm チームを作成します
    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
    team = Swarm(
        participants=[triage_agent, billing_agent, tech_support],
        termination_condition=termination,
    )

    # デモ用のスクリプト化されたユーザー応答
    scripted_responses = [
        "I was charged twice for my subscription",
        "Yes, the charge of $49.99 appears twice on my credit card statement.",
        "Thank you for your help!",
    ]
    response_index = 0

    # human-in-the-loop パターンで実行します
    print("[AutoGen] Swarm handoff conversation:")
    task_result = await Console(team.run_stream(task=scripted_responses[response_index]))
    last_message = task_result.messages[-1]
    response_index += 1

    # エージェントがユーザーにハンドオフしたときに会話を続けます
    while (
        isinstance(last_message, HandoffMessage)
        and last_message.target == "user"
        and response_index < len(scripted_responses)
    ):
        user_message = scripted_responses[response_index]
        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]
        response_index += 1


async def run_agent_framework() -> None:
    """Agent Framework の HandoffBuilder によるエージェント調整。"""
    from agent_framework import (
        AgentRunUpdateEvent,
        HandoffBuilder,
        HandoffUserInputRequest,
        RequestInfoEvent,
        WorkflowRunState,
        WorkflowStatusEvent,
    )
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4.1-mini")

    # トリアージエージェントを作成します
    triage_agent = client.create_agent(
        name="triage",
        instructions=(
            "You are a triage agent. Analyze the user's request and route to the appropriate specialist:\n"
            "- For billing issues: call handoff_to_billing_agent\n"
            "- For technical issues: call handoff_to_technical_support"
        ),
        description="Routes requests to appropriate specialists",
    )

    # 請求専門家を作成します
    billing_agent = client.create_agent(
        name="billing_agent",
        instructions="You are a billing specialist. Help with payment and billing questions. Provide clear assistance.",
        description="Handles billing and payment questions",
    )

    # 技術サポート専門家を作成します
    tech_support = client.create_agent(
        name="technical_support",
        instructions="You are technical support. Help with technical issues. Provide clear assistance.",
        description="Handles technical support questions",
    )

    # ハンドオフワークフローを作成します - より簡単な設定 専門家が応答した後、制御はユーザーに戻ります（トリアージがコーディネーターとして）
    workflow = (
        HandoffBuilder(
            name="support_handoff",
            participants=[triage_agent, billing_agent, tech_support],
        )
        .set_coordinator(triage_agent)
        .add_handoff(triage_agent, [billing_agent, tech_support])
        .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") > 3)
        .build()
    )

    # スクリプト化されたユーザー応答
    scripted_responses = [
        "I was charged twice for my subscription",
        "Yes, the charge of $49.99 appears twice on my credit card statement.",
        "Thank you for your help!",
    ]

    # 初期メッセージで実行します
    print("[Agent Framework] Handoff conversation:")
    print("---------- user ----------")
    print(scripted_responses[0])

    current_executor = None
    stream_line_open = False
    pending_requests: list[RequestInfoEvent] = []

    async for event in workflow.run_stream(scripted_responses[0]):
        if isinstance(event, AgentRunUpdateEvent):
            # 新しいエージェントに切り替わる際に executor 名のヘッダーを表示します
            if current_executor != event.executor_id:
                if stream_line_open:
                    print()
                    stream_line_open = False
                print(f"---------- {event.executor_id} ----------")
                current_executor = event.executor_id
                stream_line_open = True
            if event.data:
                print(event.data.text, end="", flush=True)
        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                pending_requests.append(event)
        elif isinstance(event, WorkflowStatusEvent):
            if event.state in {WorkflowRunState.IDLE_WITH_PENDING_REQUESTS} and stream_line_open:
                print()
                stream_line_open = False

    # スクリプト化された応答を処理します
    response_index = 1
    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print("---------- user ----------")
        print(user_response)

        responses = {req.request_id: user_response for req in pending_requests}
        pending_requests = []
        current_executor = None
        stream_line_open = False

        async for event in workflow.send_responses_streaming(responses):
            if isinstance(event, AgentRunUpdateEvent):
                # 新しいエージェントに切り替わる際に executor 名のヘッダーを表示します
                if current_executor != event.executor_id:
                    if stream_line_open:
                        print()
                        stream_line_open = False
                    print(f"---------- {event.executor_id} ----------")
                    current_executor = event.executor_id
                    stream_line_open = True
                if event.data:
                    print(event.data.text, end="", flush=True)
            elif isinstance(event, RequestInfoEvent):
                if isinstance(event.data, HandoffUserInputRequest):
                    pending_requests.append(event)
            elif isinstance(event, WorkflowStatusEvent):
                if (
                    event.state in {WorkflowRunState.IDLE_WITH_PENDING_REQUESTS, WorkflowRunState.IDLE}
                    and stream_line_open
                ):
                    print()
                    stream_line_open = False

        response_index += 1

    if stream_line_open:
        print()
    print()  # 会話の最後に改行を入れます


async def main() -> None:
    print("=" * 60)
    print("Swarm / Handoff Pattern Comparison")
    print("=" * 60)
    print("AutoGen: Swarm with handoffs")
    print("Agent Framework: HandoffBuilder\n")
    await run_autogen()
    print()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
