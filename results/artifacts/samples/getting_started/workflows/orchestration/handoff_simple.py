# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncIterable
from typing import cast

from agent_framework import (
    ChatAgent,
    ChatMessage,
    HandoffBuilder,
    HandoffUserInputRequest,
    RequestInfoEvent,
    WorkflowEvent,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""Sample: Simple handoff workflow with single-tier triage-to-specialist routing.

This sample demonstrates the basic handoff pattern where only the triage agent can
route to specialists. Specialists cannot hand off to other specialists - after any
specialist responds, control returns to the user for the next input.

Routing Pattern:
    User → Triage Agent → Specialist → Back to User → Triage Agent → ...

This is the simplest handoff configuration, suitable for straightforward support
scenarios where a triage agent dispatches to domain specialists, and each specialist
works independently.

For multi-tier specialist-to-specialist handoffs, see handoff_specialist_to_specialist.py.

Prerequisites:
    - `az login` (Azure CLI authentication)
    - Environment variables configured for AzureOpenAIChatClient (AZURE_OPENAI_ENDPOINT, etc.)

Key Concepts:
    - Single-tier routing: Only triage agent has handoff capabilities
    - Auto-registered handoff tools: HandoffBuilder creates tools automatically
    - Termination condition: Controls when the workflow stops requesting user input
    - Request/response cycle: Workflow requests input, user responds, cycle continues
"""


def create_agents(chat_client: AzureOpenAIChatClient) -> tuple[ChatAgent, ChatAgent, ChatAgent, ChatAgent]:
    """トリアージとスペシャリストAgentを作成および設定します。

    トリアージAgentは以下を担当します：
    - すべてのユーザー入力を最初に受け取る
    - リクエストを直接処理するかスペシャリストに引き渡すかを決定する
    - 明示的に公開されたhandoffツールのいずれかを呼び出してhandoffを通知する

    スペシャリストAgentはトリアージAgentが明示的にhandoffした場合のみ呼び出されます。
    スペシャリストが応答した後、制御はトリアージAgentに戻ります。

    Returns:
        (triage_agent, refund_agent, order_agent, support_agent)のタプル

    """
    # トリアージAgent：フロントラインのディスパッチャーとして機能します 注意：指示はルーティング時に正しいhandoffツールを呼び出すよう明示的に指示しています。
    # HandoffBuilderはこれらのツール呼び出しを傍受し、対応するスペシャリストにルーティングします。
    triage = chat_client.create_agent(
        instructions=(
            "You are frontline support triage. Read the latest user message and decide whether "
            "to hand off to refund_agent, order_agent, or support_agent. Provide a brief natural-language "
            "response for the user. When delegation is required, call the matching handoff tool "
            "(`handoff_to_refund_agent`, `handoff_to_order_agent`, or `handoff_to_support_agent`)."
        ),
        name="triage_agent",
    )

    # 返金スペシャリスト：返金リクエストを処理します。
    refund = chat_client.create_agent(
        instructions=(
            "You handle refund workflows. Ask for any order identifiers you require and outline the refund steps."
        ),
        name="refund_agent",
    )

    # 注文／配送スペシャリスト：配送問題を解決します。
    order = chat_client.create_agent(
        instructions=(
            "You resolve shipping and fulfillment issues. Clarify the delivery problem and describe the actions "
            "you will take to remedy it."
        ),
        name="order_agent",
    )

    # 一般サポートスペシャリスト：その他の問題のフォールバック。
    support = chat_client.create_agent(
        instructions=(
            "You are a general support agent. Offer empathetic troubleshooting and gather missing details if the "
            "issue does not match other specialists."
        ),
        name="support_agent",
    )

    return triage, refund, order, support


async def _drain(stream: AsyncIterable[WorkflowEvent]) -> list[WorkflowEvent]:
    """非同期ストリームからすべてのイベントをリストに収集します。

    このヘルパーはワークフローのイベントストリームを排出し、
    各ワークフローのステップ完了後に同期的にイベントを処理できるようにします。

    Args:
        stream: WorkflowEventの非同期イテラブル

    Returns:
        ストリームからのすべてのイベントのリスト

    """
    return [event async for event in stream]


def _handle_events(events: list[WorkflowEvent]) -> list[RequestInfoEvent]:
    """ワークフローイベントを処理し、保留中のユーザー入力リクエストを抽出します。

    この関数は各イベントタイプを検査し：
    - ワークフローステータスの変化（IDLE、IDLE_WITH_PENDING_REQUESTSなど）を表示
    - ワークフロー完了時の最終会話スナップショットを表示
    - ユーザー入力リクエストのプロンプトを表示
    - レスポンス処理のためにすべてのRequestInfoEventインスタンスを収集

    Args:
        events: 処理するWorkflowEventのリスト

    Returns:
        保留中のユーザー入力リクエストを表すRequestInfoEventのリスト

    """
    requests: list[RequestInfoEvent] = []

    for event in events:
        # WorkflowStatusEvent：ワークフローの状態変化を示します。
        if isinstance(event, WorkflowStatusEvent) and event.state in {
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        }:
            print(f"[status] {event.state.name}")

        # WorkflowOutputEvent：ワークフロー終了時の最終会話を含みます。
        elif isinstance(event, WorkflowOutputEvent):
            conversation = cast(list[ChatMessage], event.data)
            if isinstance(conversation, list):
                print("\n=== Final Conversation Snapshot ===")
                for message in conversation:
                    speaker = message.author_name or message.role.value
                    print(f"- {speaker}: {message.text}")
                print("===================================")

        # RequestInfoEvent：ワークフローがユーザー入力を要求しています。
        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                _print_handoff_request(event.data)
            requests.append(event)

    return requests


def _print_handoff_request(request: HandoffUserInputRequest) -> None:
    """会話のコンテキストとともにユーザー入力リクエストのプロンプトを表示します。

    HandoffUserInputRequestはこれまでの会話履歴全体を含み、
    ユーザーが次の入力を提供する前に何が話されたかを確認できます。

    Args:
        request: 会話とプロンプトを含むユーザー入力リクエスト

    """
    print("\n=== User Input Requested ===")
    for message in request.conversation:
        speaker = message.author_name or message.role.value
        print(f"- {speaker}: {message.text}")
    print("============================")


async def main() -> None:
    """handoffワークフロードemoのメインエントリポイント。

    この関数は以下を示します：
    1. トリアージとスペシャリストAgentの作成
    2. カスタム終了条件を持つhandoffワークフローの構築
    3. スクリプト化されたユーザーレスポンスでのワークフローの実行
    4. イベントの処理とユーザー入力リクエストのハンドリング

    ワークフローは対話的入力の代わりにスクリプト化されたレスポンスを使用し、
    デモの再現性とテスト可能性を高めています。実際のアプリケーションでは
    scripted_responsesを実際のユーザー入力収集に置き換えます。

    """
    # Azure OpenAIチャットクライアントを初期化します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    # すべてのAgentを作成します：トリアージ＋スペシャリスト。
    triage, refund, order, support = create_agents(chat_client)

    # handoffワークフローを構築します - participants:
    # 参加可能なすべてのAgent（トリアージは最初であるか明示的にset_coordinatorとして設定される必要があります） - set_coordinator:
    # トリアージAgentがすべてのユーザー入力を最初に受け取ります - with_termination_condition:
    # リクエスト／レスポンスループを停止するカスタムロジック デフォルトは10ユーザーメッセージですが、ここではスクリプト化デモに合わせて4で終了します。
    workflow = (
        HandoffBuilder(
            name="customer_support_handoff",
            participants=[triage, refund, order, support],
        )
        .set_coordinator("triage_agent")
        .with_termination_condition(
            # 4ユーザーメッセージ後に終了（初回＋3つのスクリプト化レスポンス）
            # AgentのレスポンスをカウントしないようにUSERロールのメッセージのみカウントします。
            lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 4
        )
        .build()
    )

    # 再現可能なデモのためのスクリプト化されたユーザー応答 コンソールアプリケーションでは、これを以下に置き換えてください: user_input =
    # input("Your response: ") またはUI/Chatインターフェースと統合してください
    scripted_responses = [
        "My order 1234 arrived damaged and the packaging was destroyed.",
        "Yes, I'd like a refund if that's possible.",
        "Thanks for resolving this.",
    ]

    # 初期ユーザーメッセージでワークフローを開始します run_stream()はWorkflowEventの非同期イテレータを返します
    print("\n[Starting workflow with initial user message...]")
    events = await _drain(workflow.run_stream("Hello, I need assistance with my recent purchase."))
    pending_requests = _handle_events(events)

    # リクエスト/レスポンスサイクルを処理します ワークフローは以下のいずれかの条件が満たされるまで入力を要求し続けます: 1.
    # 終了条件が満たされる（この場合は4つのユーザーメッセージ）、または 2. スクリプト化された応答が尽きる
    while pending_requests and scripted_responses:
        # 次のスクリプト化された応答を取得します
        user_response = scripted_responses.pop(0)
        print(f"\n[User responding: {user_response}]")

        # すべての保留中リクエストにレスポンスを送信します このデモでは通常1サイクルにつき1リクエストですが、APIは複数をサポートします
        responses = {req.request_id: user_response for req in pending_requests}

        # レスポンスを送信し、新しいイベントを取得します
        events = await _drain(workflow.send_responses_streaming(responses))
        pending_requests = _handle_events(events)

    """
    Sample Output:

    [Starting workflow with initial user message...]

    === User Input Requested ===
    - user: Hello, I need assistance with my recent purchase.
    - triage_agent: I'd be happy to help you with your recent purchase. Could you please provide more details about the issue you're experiencing?
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User responding: My order 1234 arrived damaged and the packaging was destroyed.]

    === User Input Requested ===
    - user: Hello, I need assistance with my recent purchase.
    - triage_agent: I'd be happy to help you with your recent purchase. Could you please provide more details about the issue you're experiencing?
    - user: My order 1234 arrived damaged and the packaging was destroyed.
    - triage_agent: I'm sorry to hear that your order arrived damaged and the packaging was destroyed. I will connect you with a specialist who can assist you further with this issue.

    Tool Call: handoff_to_support_agent (awaiting approval)
    - support_agent: I'm so sorry to hear that your order arrived in such poor condition. I'll help you get this sorted out.

    To assist you better, could you please let me know:
    - Which item(s) from order 1234 arrived damaged?
    - Could you describe the damage, or provide photos if possible?
    - Would you prefer a replacement or a refund?

    Once I have this information, I can help resolve this for you as quickly as possible.
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User responding: Yes, I'd like a refund if that's possible.]

    === User Input Requested ===
    - user: Hello, I need assistance with my recent purchase.
    - triage_agent: I'd be happy to help you with your recent purchase. Could you please provide more details about the issue you're experiencing?
    - user: My order 1234 arrived damaged and the packaging was destroyed.
    - triage_agent: I'm sorry to hear that your order arrived damaged and the packaging was destroyed. I will connect you with a specialist who can assist you further with this issue.

    Tool Call: handoff_to_support_agent (awaiting approval)
    - support_agent: I'm so sorry to hear that your order arrived in such poor condition. I'll help you get this sorted out.

    To assist you better, could you please let me know:
    - Which item(s) from order 1234 arrived damaged?
    - Could you describe the damage, or provide photos if possible?
    - Would you prefer a replacement or a refund?

    Once I have this information, I can help resolve this for you as quickly as possible.
    - user: Yes, I'd like a refund if that's possible.
    - triage_agent: Thank you for letting me know you'd prefer a refund. I'll connect you with a specialist who can process your refund request.

    Tool Call: handoff_to_refund_agent (awaiting approval)
    - refund_agent: Thank you for confirming that you'd like a refund for order 1234.

    Here's what will happen next:

    ...

    Tool Call: handoff_to_refund_agent (awaiting approval)
    - refund_agent: Thank you for confirming that you'd like a refund for order 1234.

    Here's what will happen next:

    **1. Verification:**
    I will need to verify a few more details to proceed.
    - Can you confirm the items in order 1234 that arrived damaged?
    - Do you have any photos of the damaged items/packaging? (Photos help speed up the process.)

    **2. Refund Request Submission:**
    - Once I have the details, I will submit your refund request for review.

    **3. Return Instructions (if needed):**
    - In some cases, we may provide instructions on how to return the damaged items.
    - You will receive a prepaid return label if necessary.

    **4. Refund Processing:**
    - After your request is approved (and any returns are received if required), your refund will be processed.
    - Refunds usually appear on your original payment method within 5-10 business days.

    Could you please reply with the specific item(s) damaged and, if possible, attach photos? This will help me get your refund started right away.
    - user: Thanks for resolving this.
    ===================================
    [status] IDLE
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
