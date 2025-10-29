# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from pydantic import BaseModel
from typing_extensions import Never

"""
Sample: Shared state with agents and conditional routing.

Store an email once by id, classify it with a detector agent, then either draft a reply with an assistant
agent or finish with a spam notice. Stream events as the workflow runs.

Purpose:
Show how to:
- Use shared state to decouple large payloads from messages and pass around lightweight references.
- Enforce structured agent outputs with Pydantic models via response_format for robust parsing.
- Route using conditional edges based on a typed intermediate DetectionResult.
- Compose agent backed executors with function style executors and yield the final output when the workflow completes.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Familiarity with WorkflowBuilder, executors, conditional edges, and streaming runs.
"""

EMAIL_STATE_PREFIX = "email:"
CURRENT_EMAIL_ID_KEY = "current_email_id"


class DetectionResultAgent(BaseModel):
    """spam検出Agentが返す構造化出力。"""

    is_spam: bool
    reason: str


class EmailResponse(BaseModel):
    """email assistant Agentが返す構造化出力。"""

    response: str


@dataclass
class DetectionResult:
    """後で参照可能な共有Stateのemail_idで強化された内部検出結果。"""

    is_spam: bool
    reason: str
    email_id: str


@dataclass
class Email:
    """エッジ上で大きな本文を再送しないように共有Stateに保存されたメモリ内レコード。"""

    email_id: str
    email_content: str


def get_condition(expected_result: bool):
    """DetectionResult.is_spamの条件述語を作成します。

    契約:
    - メッセージがDetectionResultでない場合は誤ったデッドエンドを避けるために通過を許可します。
    - それ以外の場合はis_spamがexpected_resultと一致するときのみTrueを返します。

    """

    def condition(message: Any) -> bool:
        if not isinstance(message, DetectionResult):
            return True
        return message.is_spam == expected_result

    return condition


@executor(id="store_email")
async def store_email(email_text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """生のemail内容を共有Stateに永続化し、spam検出をトリガーします。

    責務:
    - 下流での取得のために一意のemail_id（UUID）を生成します。
    - Emailオブジェクトを名前空間付きキーで保存し、現在のidポインタを設定します。
    - 検出器に応答を求めるAgentExecutorRequestを発行します。

    """
    new_email = Email(email_id=str(uuid4()), email_content=email_text)
    await ctx.set_shared_state(f"{EMAIL_STATE_PREFIX}{new_email.email_id}", new_email)
    await ctx.set_shared_state(CURRENT_EMAIL_ID_KEY, new_email.email_id)

    await ctx.send_message(
        AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=new_email.email_content)], should_respond=True)
    )


@executor(id="to_detection_result")
async def to_detection_result(response: AgentExecutorResponse, ctx: WorkflowContext[DetectionResult]) -> None:
    """spam検出JSONを構造化モデルに解析し、email_idで強化します。

    手順:
    1) AgentのJSON出力をDetectionResultAgentに検証します。
    2) 共有Stateから現在のemail_idを取得します。
    3) 条件付きルーティングのために型付きDetectionResultを送信します。

    """
    parsed = DetectionResultAgent.model_validate_json(response.agent_run_response.text)
    email_id: str = await ctx.get_shared_state(CURRENT_EMAIL_ID_KEY)
    await ctx.send_message(DetectionResult(is_spam=parsed.is_spam, reason=parsed.reason, email_id=email_id))


@executor(id="submit_to_email_assistant")
async def submit_to_email_assistant(detection: DetectionResult, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """spamでないemail内容をドラフトAgentに転送します。

    ガード:
    - この経路はspamでないもののみ受け取るべきです。誤ったルーティングの場合は例外を発生させます。

    """
    if detection.is_spam:
        raise RuntimeError("This executor should only handle non-spam messages.")

    # 共有Stateからidで元の内容を読み込み、assistantに転送します。
    email: Email = await ctx.get_shared_state(f"{EMAIL_STATE_PREFIX}{detection.email_id}")
    await ctx.send_message(
        AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=email.email_content)], should_respond=True)
    )


@executor(id="finalize_and_send")
async def finalize_and_send(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """ドラフトされた返信を検証し、最終出力を生成します。"""
    parsed = EmailResponse.model_validate_json(response.agent_run_response.text)
    await ctx.yield_output(f"Email sent: {parsed.response}")


@executor(id="handle_spam")
async def handle_spam(detection: DetectionResult, ctx: WorkflowContext[Never, str]) -> None:
    """emailがspamとマークされた理由を説明する出力を生成します。"""
    if detection.is_spam:
        await ctx.yield_output(f"Email marked as spam: {detection.reason}")
    else:
        raise RuntimeError("This executor should only handle spam messages.")


async def main() -> None:
    # chat ClientとAgentを作成します。response_formatは各Agentからの構造化JSONを強制します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    spam_detection_agent = chat_client.create_agent(
        instructions=(
            "You are a spam detection assistant that identifies spam emails. "
            "Always return JSON with fields is_spam (bool) and reason (string)."
        ),
        response_format=DetectionResultAgent,
        name="spam_detection_agent",
    )

    email_assistant_agent = chat_client.create_agent(
        instructions=(
            "You are an email assistant that helps users draft responses to emails with professionalism. "
            "Return JSON with a single field 'response' containing the drafted reply."
        ),
        response_format=EmailResponse,
        name="email_assistant_agent",
    )

    # 条件付きエッジを持つワークフローグラフを構築します。 Flow: store_email -> spam_detection_agent ->
    # to_detection_result -> branch: False -> submit_to_email_assistant ->
    # email_assistant_agent -> finalize_and_send True  -> handle_spam
    workflow = (
        WorkflowBuilder()
        .set_start_executor(store_email)
        .add_edge(store_email, spam_detection_agent)
        .add_edge(spam_detection_agent, to_detection_result)
        .add_edge(to_detection_result, submit_to_email_assistant, condition=get_condition(False))
        .add_edge(to_detection_result, handle_spam, condition=get_condition(True))
        .add_edge(submit_to_email_assistant, email_assistant_agent)
        .add_edge(email_assistant_agent, finalize_and_send)
        .build()
    )

    # resources/spam.txtからemailを読み込み、なければデフォルトのサンプルを使用します。
    resources_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "resources",
        "spam.txt",
    )
    if os.path.exists(resources_path):
        with open(resources_path, encoding="utf-8") as f:  # noqa: ASYNC230
            email = f.read()
    else:
        print("Unable to find resource file, using default text.")
        email = "You are a WINNER! Click here for a free lottery offer!!!"

    # 実行して最終結果を表示します。ストリーミングは中間の実行イベントも表示します。
    events = await workflow.run(email)
    outputs = events.get_outputs()

    if outputs:
        print(f"Final result: {outputs[0]}")

    """
    Sample Output:

    Final result: Email marked as spam: This email exhibits several common spam and scam characteristics:
    unrealistic claims of large cash winnings, urgent time pressure, requests for sensitive personal and financial
    information, and a demand for a processing fee. The sender impersonates a generic lottery commission, and the
    message contains a suspicious link. All these are typical of phishing and lottery scam emails.
    """


if __name__ == "__main__":
    asyncio.run(main())
