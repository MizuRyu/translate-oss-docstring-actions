# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from agent_framework import (  # Core chat primitives used to form LLM requests
    AgentExecutor,  # Wraps an agent so it can run inside a workflow
    AgentExecutorRequest,  # Message bundle sent to an AgentExecutor
    AgentExecutorResponse,  # Result returned by an AgentExecutor
    Case,  # Case entry for a switch-case edge group
    ChatMessage,
    Default,  # Default branch when no cases match
    Role,
    WorkflowBuilder,  # Fluent builder for assembling the graph
    WorkflowContext,  # Per-run context and event bus
    executor,  # Decorator to turn a function into a workflow executor
)
from agent_framework.azure import AzureOpenAIChatClient  # Azure OpenAIチャットモデル用のシンクライアント。
from azure.identity import AzureCliCredential  # az CLIログインを資格情報として使用します。
from pydantic import BaseModel  # 検証付きの構造化出力。
from typing_extensions import Never

"""
Sample: Switch-Case Edge Group with an explicit Uncertain branch.

The workflow stores a single email in shared state, asks a spam detection agent for a three way decision,
then routes with a switch-case group: NotSpam to the drafting assistant, Spam to a spam handler, and
Default to an Uncertain handler.

Purpose:
Demonstrate deterministic one of N routing with switch-case edges. Show how to:
- Persist input once in shared state, then pass around a small typed pointer that carries the email id.
- Validate agent JSON with Pydantic models for robust parsing.
- Keep executor responsibilities narrow. Transform model output to a typed DetectionResult, then route based
on that type.
- Use ctx.yield_output() to provide workflow results - the workflow completes when idle with no pending work.

Prerequisites:
- Familiarity with WorkflowBuilder, executors, edges, and events.
- Understanding of switch-case edge groups and how Case and Default are evaluated in order.
- Working Azure OpenAI configuration for AzureOpenAIChatClient, with Azure CLI login and required environment variables.
- Access to workflow/resources/ambiguous_email.txt, or accept the inline fallback string.
"""


EMAIL_STATE_PREFIX = "email:"
CURRENT_EMAIL_ID_KEY = "current_email_id"


class DetectionResultAgent(BaseModel):
    """spam detection agentが返す構造化出力。"""

    # agentはメールを分類し、その理由を提供します。
    spam_decision: Literal["NotSpam", "Spam", "Uncertain"]
    reason: str


class EmailResponse(BaseModel):
    """email assistant agentが返す構造化出力。"""

    # 作成されたプロフェッショナルな返信。
    response: str


@dataclass
class DetectionResult:
    # ルーティングと下流処理に使用される内部の型付きペイロード。
    spam_decision: str
    reason: str
    email_id: str


@dataclass
class Email:
    # 共有Stateに保存されたメール内容のメモリ内記録。
    email_id: str
    email_content: str


def get_case(expected_decision: str):
    """特定のspam_decision値にマッチする述語を返すファクトリー。"""

    def condition(message: Any) -> bool:
        # 上流のペイロードが期待されるdecisionのDetectionResultの場合のみマッチします。
        return isinstance(message, DetectionResult) and message.spam_decision == expected_decision

    return condition


@executor(id="store_email")
async def store_email(email_text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    # 生のメールを一度だけ永続化します。一意のキーで保存し、利便性のために現在のポインターを設定します。
    new_email = Email(email_id=str(uuid4()), email_content=email_text)
    await ctx.set_shared_state(f"{EMAIL_STATE_PREFIX}{new_email.email_id}", new_email)
    await ctx.set_shared_state(CURRENT_EMAIL_ID_KEY, new_email.email_id)

    # メールをユーザーメッセージとしてspam_detection_agentに転送し、検出器を起動します。
    await ctx.send_message(
        AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=new_email.email_content)], should_respond=True)
    )


@executor(id="to_detection_result")
async def to_detection_result(response: AgentExecutorResponse, ctx: WorkflowContext[DetectionResult]) -> None:
    # 検出器のJSONを型付きモデルに解析します。下流の参照用に現在のメールIDを添付します。
    parsed = DetectionResultAgent.model_validate_json(response.agent_run_response.text)
    email_id: str = await ctx.get_shared_state(CURRENT_EMAIL_ID_KEY)
    await ctx.send_message(DetectionResult(spam_decision=parsed.spam_decision, reason=parsed.reason, email_id=email_id))


@executor(id="submit_to_email_assistant")
async def submit_to_email_assistant(detection: DetectionResult, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    # NotSpamブランチのみ進行します。誤ったルーティングを防ぎます。
    if detection.spam_decision != "NotSpam":
        raise RuntimeError("This executor should only handle NotSpam messages.")

    # DetectionResultに含まれるIDを使って共有Stateから元の内容をロードします。
    email: Email = await ctx.get_shared_state(f"{EMAIL_STATE_PREFIX}{detection.email_id}")
    await ctx.send_message(
        AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=email.email_content)], should_respond=True)
    )


@executor(id="finalize_and_send")
async def finalize_and_send(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    # ドラフトブランチの終端ステップ。メールの返信を出力としてyieldします。
    parsed = EmailResponse.model_validate_json(response.agent_run_response.text)
    await ctx.yield_output(f"Email sent: {parsed.response}")


@executor(id="handle_spam")
async def handle_spam(detection: DetectionResult, ctx: WorkflowContext[Never, str]) -> None:
    # Spamパスの終端。検出器の理由を含めます。
    if detection.spam_decision == "Spam":
        await ctx.yield_output(f"Email marked as spam: {detection.reason}")
    else:
        raise RuntimeError("This executor should only handle Spam messages.")


@executor(id="handle_uncertain")
async def handle_uncertain(detection: DetectionResult, ctx: WorkflowContext[Never, str]) -> None:
    # 不確定パスの終端。元の内容を表示して人間のレビューを支援します。
    if detection.spam_decision == "Uncertain":
        email: Email | None = await ctx.get_shared_state(f"{EMAIL_STATE_PREFIX}{detection.email_id}")
        await ctx.yield_output(
            f"Email marked as uncertain: {detection.reason}. Email content: {getattr(email, 'email_content', '')}"
        )
    else:
        raise RuntimeError("This executor should only handle Uncertain messages.")


async def main():
    """ワークフローを実行するメイン関数。"""
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    # Agents。response_formatはLLMがPydanticで検証可能なJSONを返すことを強制します。
    spam_detection_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You are a spam detection assistant that identifies spam emails. "
                "Be less confident in your assessments. "
                "Always return JSON with fields 'spam_decision' (one of NotSpam, Spam, Uncertain) "
                "and 'reason' (string)."
            ),
            response_format=DetectionResultAgent,
        ),
        id="spam_detection_agent",
    )

    email_assistant_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You are an email assistant that helps users draft responses to emails with professionalism."
            ),
            response_format=EmailResponse,
        ),
        id="email_assistant_agent",
    )

    # ワークフロー構築: store -> detection agent -> to_detection_result -> switch (NotSpam or
    # Spam or Default). switch-caseグループは順にケースを評価し、どれもマッチしない場合はDefaultにフォールバックします。
    workflow = (
        WorkflowBuilder()
        .set_start_executor(store_email)
        .add_edge(store_email, spam_detection_agent)
        .add_edge(spam_detection_agent, to_detection_result)
        .add_switch_case_edge_group(
            to_detection_result,
            [
                Case(condition=get_case("NotSpam"), target=submit_to_email_assistant),
                Case(condition=get_case("Spam"), target=handle_spam),
                Default(target=handle_uncertain),
            ],
        )
        .add_edge(submit_to_email_assistant, email_assistant_agent)
        .add_edge(email_assistant_agent, finalize_and_send)
        .build()
    )

    # 曖昧なメールがあれば読み込みます。なければ簡単なインラインサンプルを使用します。
    resources_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "resources", "ambiguous_email.txt"
    )
    if os.path.exists(resources_path):
        with open(resources_path, encoding="utf-8") as f:  # noqa: ASYNC230
            email = f.read()
    else:
        print("Unable to find resource file, using default text.")
        email = (
            "Hey there, I noticed you might be interested in our latest offer—no pressure, but it expires soon. "
            "Let me know if you'd like more details."
        )

    # 完了したブランチの出力を実行して表示します。
    events = await workflow.run(email)
    outputs = events.get_outputs()
    if outputs:
        for output in outputs:
            print(f"Workflow output: {output}")


if __name__ == "__main__":
    asyncio.run(main())
