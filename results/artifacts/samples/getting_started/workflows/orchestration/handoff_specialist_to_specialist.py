# Copyright (c) Microsoft. All rights reserved.

"""サンプル: スペシャリスト間ルーティングを伴う多層ハンドオフワークフロー。

このサンプルは、スペシャリストAgentが他のスペシャリストにハンドオフできる高度なルーティングを示します。
これにより複雑な多層ワークフローが可能になります。単純なハンドオフパターン（handoff_simple.py参照）とは異なり、
ここではスペシャリストがユーザーに制御を戻すことなく他のスペシャリストに委任できます。

ルーティングパターン:
    User → Triage → Specialist A → Specialist B → Back to User

このパターンは、異なるスペシャリストが協力またはエスカレーションを行い、ユーザーに戻る前に連携が必要な複雑なサポートシナリオに有用です。例えば:
    - 交換Agentが配送情報を必要とする → 配送Agentにハンドオフ
    - 技術サポートが請求情報を必要とする → 請求Agentにハンドオフ
    - レベル1サポートがレベル2にエスカレーション → エスカレーションAgentにハンドオフ

設定は`.add_handoff()`を使ってルーティンググラフを明示的に定義します。

前提条件:
    - `az login`（Azure CLI認証）
    - AzureOpenAIChatClient用の環境変数設定
"""

import asyncio
from collections.abc import AsyncIterable
from typing import cast

from agent_framework import (
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


def create_agents(chat_client: AzureOpenAIChatClient):
    """多層ハンドオフ機能を持つトリアージおよびスペシャリストAgentを作成します。

    Returns:
        (triage_agent, replacement_agent, delivery_agent, billing_agent)のタプルを返します

    """
    triage = chat_client.create_agent(
        instructions=(
            "You are a customer support triage agent. Assess the user's issue and route appropriately:\n"
            "- For product replacement issues: call handoff_to_replacement_agent\n"
            "- For delivery/shipping inquiries: call handoff_to_delivery_agent\n"
            "- For billing/payment issues: call handoff_to_billing_agent\n"
            "Be concise and friendly."
        ),
        name="triage_agent",
    )

    replacement = chat_client.create_agent(
        instructions=(
            "You handle product replacement requests. Ask for order number and reason for replacement.\n"
            "If the user also needs shipping/delivery information, call handoff_to_delivery_agent to "
            "get tracking details. Otherwise, process the replacement and confirm with the user.\n"
            "Be concise and helpful."
        ),
        name="replacement_agent",
    )

    delivery = chat_client.create_agent(
        instructions=(
            "You handle shipping and delivery inquiries. Provide tracking information, estimated "
            "delivery dates, and address any delivery concerns.\n"
            "If billing issues come up, call handoff_to_billing_agent.\n"
            "Be concise and clear."
        ),
        name="delivery_agent",
    )

    billing = chat_client.create_agent(
        instructions=(
            "You handle billing and payment questions. Help with refunds, payment methods, "
            "and invoice inquiries. Be concise."
        ),
        name="billing_agent",
    )

    return triage, replacement, delivery, billing


async def _drain(stream: AsyncIterable[WorkflowEvent]) -> list[WorkflowEvent]:
    """非同期ストリームからすべてのイベントをリストに収集します。"""
    return [event async for event in stream]


def _handle_events(events: list[WorkflowEvent]) -> list[RequestInfoEvent]:
    """ワークフローイベントを処理し、保留中のユーザー入力リクエストを抽出します。"""
    requests: list[RequestInfoEvent] = []

    for event in events:
        if isinstance(event, WorkflowStatusEvent) and event.state in {
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        }:
            print(f"[status] {event.state.name}")

        elif isinstance(event, WorkflowOutputEvent):
            conversation = cast(list[ChatMessage], event.data)
            if isinstance(conversation, list):
                print("\n=== Final Conversation ===")
                for message in conversation:
                    # テキストのないメッセージ（ツール呼び出し）を除外します
                    if not message.text.strip():
                        continue
                    speaker = message.author_name or message.role.value
                    print(f"- {speaker}: {message.text}")
                print("==========================")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                _print_handoff_request(event.data)
            requests.append(event)

    return requests


def _print_handoff_request(request: HandoffUserInputRequest) -> None:
    """会話コンテキスト付きでユーザー入力リクエストを表示します。"""
    print("\n=== User Input Requested ===")
    # 表示をすっきりさせるため、テキストのないメッセージを除外します
    messages_with_text = [msg for msg in request.conversation if msg.text.strip()]
    print(f"Last {len(messages_with_text)} messages in conversation:")
    for message in messages_with_text[-5:]:  # Show last 5 for brevity
        speaker = message.author_name or message.role.value
        text = message.text[:100] + "..." if len(message.text) > 100 else message.text
        print(f"  {speaker}: {text}")
    print("============================")


async def main() -> None:
    """多層サポートシナリオにおけるスペシャリスト間ハンドオフを示します。

    このサンプルは以下を示します:
    1. トリアージAgentが交換スペシャリストにルーティング
    2. 交換スペシャリストが配送スペシャリストにハンドオフ
    3. 配送スペシャリストは必要に応じて請求にハンドオフ可能
    4. すべての遷移は完了までユーザーに戻らずシームレス

    ワークフロー設定はどのAgentがどのAgentにハンドオフ可能かを明示的に定義しています:
    - triage_agent → replacement_agent, delivery_agent, billing_agent
    - replacement_agent → delivery_agent, billing_agent
    - delivery_agent → billing_agent

    """
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    triage, replacement, delivery, billing = create_agents(chat_client)

    # fluentなadd_handoff() APIを使って多層ハンドオフを設定します これによりスペシャリストが他のスペシャリストにハンドオフ可能になります
    workflow = (
        HandoffBuilder(
            name="multi_tier_support",
            participants=[triage, replacement, delivery, billing],
        )
        .set_coordinator(triage)
        .add_handoff(triage, [replacement, delivery, billing])  # Triage can route to any specialist
        .add_handoff(replacement, [delivery, billing])  # Replacement can delegate to delivery or billing
        .add_handoff(delivery, billing)  # Delivery can escalate to billing
        # 終了条件: ユーザーメッセージが4つを超えたら停止します。 これによりAgentは4番目のユーザーメッセージに応答し、5番目で終了がトリガーされます。
        # このサンプルでは初期メッセージ＋3つのスクリプト応答＝4メッセージ、その後5番目でワークフロー終了です。
        .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") > 4)
        .build()
    )

    # 多層ハンドオフシナリオを模擬するスクリプト化されたユーザー応答 注意: 初回のrun_stream()呼び出しで最初のユーザーメッセージが送信され、
    # その後これらのスクリプト応答が順に送信されます（合計4つのユーザーメッセージ）。 5番目の応答は4番目の応答後に終了をトリガーします。
    scripted_responses = [
        "I need help with order 12345. I want a replacement and need to know when it will arrive.",
        "The item arrived damaged. I'd like a replacement shipped to the same address.",
        "Great! Can you confirm the shipping cost won't be charged again?",
        "Thank you!",  # Final response to trigger termination after billing agent answers
    ]

    print("\n" + "=" * 80)
    print("SPECIALIST-TO-SPECIALIST HANDOFF DEMONSTRATION")
    print("=" * 80)
    print("\nScenario: Customer needs replacement + shipping info + billing confirmation")
    print("Expected flow: User → Triage → Replacement → Delivery → Billing → User")
    print("=" * 80 + "\n")

    # 初期メッセージでワークフローを開始します
    print("[User]: I need help with order 12345. I want a replacement and need to know when it will arrive.\n")
    events = await _drain(
        workflow.run_stream("I need help with order 12345. I want a replacement and need to know when it will arrive.")
    )
    pending_requests = _handle_events(events)

    # スクリプト化された応答を処理します
    response_index = 0
    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User]: {user_response}\n")

        responses = {req.request_id: user_response for req in pending_requests}
        events = await _drain(workflow.send_responses_streaming(responses))
        pending_requests = _handle_events(events)

        response_index += 1

    """
    Sample Output:

    ================================================================================
    SPECIALIST-TO-SPECIALIST HANDOFF DEMONSTRATION
    ================================================================================

    Scenario: Customer needs replacement + shipping info + billing confirmation
    Expected flow: User → Triage → Replacement → Delivery → Billing → User
    ================================================================================

    [User]: I need help with order 12345. I want a replacement and need to know when it will arrive.


    === User Input Requested ===
    Last 5 messages in conversation:
    user: I need help with order 12345. I want a replacement and need to know when it will arrive.
    triage_agent: I'm connecting you to our replacement team to assist with your request, and to our delivery team for...
    replacement_agent: To assist with your replacement for order 12345 and provide tracking details for delivery, I've reac...
    delivery_agent: I'm handing over your request for a replacement of order 12345, as well as your inquiry about estima...
    billing_agent: I handle billing and payment questions. For replacement and delivery details for order 12345, please...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User]: I need help with order 12345. I want a replacement and need to know when it will arrive.


    === User Input Requested ===
    Last 7 messages in conversation:
    replacement_agent: To assist with your replacement for order 12345 and provide tracking details for delivery, I've reac...
    delivery_agent: I'm handing over your request for a replacement of order 12345, as well as your inquiry about estima...
    billing_agent: I handle billing and payment questions. For replacement and delivery details for order 12345, please...
    user: I need help with order 12345. I want a replacement and need to know when it will arrive.
    triage_agent: I'm connecting you with our replacement team to help with your request, and our delivery team for in...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User]: The item arrived damaged. I'd like a replacement shipped to the same address.


    === User Input Requested ===
    Last 12 messages in conversation:
    user: The item arrived damaged. I'd like a replacement shipped to the same address.
    triage_agent: Thank you for letting us know your item from order 12345 arrived damaged. Im connecting you to our ...
    replacement_agent: Your replacement request for order 12345 (damaged item) will be processed, and shipped to your curre...
    delivery_agent: I've forwarded your request for a replacement of the damaged item from order 12345 to our team and a...
    billing_agent: Thank you for the information. I've forwarded your request to our support team to arrange a replacem...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User]: Great! Can you confirm the shipping cost won't be charged again?


    === Final Conversation ===
    - user: I need help with order 12345. I want a replacement and need to know when it will arrive.
    - triage_agent: I'm connecting you to our replacement team to assist with your request, and to our delivery team for updates on when your replacement will arrive. They'll help you shortly!
    - replacement_agent: To assist with your replacement for order 12345 and provide tracking details for delivery, I've reached out to our teams. Can you please provide the reason for your replacement request? This will help to process things faster.
    - delivery_agent: I'm handing over your request for a replacement of order 12345, as well as your inquiry about estimated delivery timing, to the appropriate team. They'll assist you further with the replacement and delivery details very soon.
    - billing_agent: I handle billing and payment questions. For replacement and delivery details for order 12345, please contact our support team or the order department. Is there a billing or payment issue I can help you with?
    - user: I need help with order 12345. I want a replacement and need to know when it will arrive.
    - triage_agent: I'm connecting you with our replacement team to help with your request, and our delivery team for information about when your replacement will arrive. They'll assist you shortly!
    - user: The item arrived damaged. I'd like a replacement shipped to the same address.
    - triage_agent: Thank you for letting us know your item from order 12345 arrived damaged. I'm connecting you to our replacement team to arrange a replacement shipped to your address, and to our delivery team to update you on the arrival date of your replacement. They'll assist you shortly!
    - replacement_agent: Your replacement request for order 12345 (damaged item) will be processed, and shipped to your current address. The delivery team will also provide tracking and estimated arrival details for your new item soon.
    - delivery_agent: I've forwarded your request for a replacement of the damaged item from order 12345 to our team and asked them to ship it to your current address. You'll also receive tracking information and an estimated delivery date for the replacement soon.
    - billing_agent: Thank you for the information. I've forwarded your request to our support team to arrange a replacement for the damaged item from order 12345. Your replacement will be shipped to the same address, and you'll receive delivery updates soon. If you need a refund instead or have any billing questions, please let me know.
    - user: Great! Can you confirm the shipping cost won't be charged again?
    ==========================
    [status] IDLE
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
