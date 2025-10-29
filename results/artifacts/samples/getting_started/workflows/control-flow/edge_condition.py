# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from typing import Any

from agent_framework import (  # Core chat primitives used to build requests
    AgentExecutor,  # Wraps an LLM agent that can be invoked inside a workflow
    AgentExecutorRequest,  # Input message bundle for an AgentExecutor
    AgentExecutorResponse,  # Output from an AgentExecutor
    ChatMessage,
    Role,
    WorkflowBuilder,  # Fluent builder for wiring executors and edges
    WorkflowContext,  # Per-run context and event bus
    executor,  # Decorator to declare a Python function as a workflow executor
)
from agent_framework.azure import AzureOpenAIChatClient  # Azure OpenAI chatモデルの薄いクライアントラッパー
from azure.identity import AzureCliCredential  # az CLIログインを資格情報として使用します
from pydantic import BaseModel  # 安全な解析のための構造化された出力
from typing_extensions import Never

"""
Sample: Conditional routing with structured outputs

What this sample is:
- A minimal decision workflow that classifies an inbound email as spam or not spam, then routes to the
appropriate handler.

Purpose:
- Show how to attach boolean edge conditions that inspect an AgentExecutorResponse.
- Demonstrate using Pydantic models as response_format so the agent returns JSON we can validate and parse.
- Illustrate how to transform one agent's structured result into a new AgentExecutorRequest for a downstream agent.

Prerequisites:
- You understand the basics of WorkflowBuilder, executors, and events in this framework.
- You know the concept of edge conditions and how they gate routes using a predicate function.
- Azure OpenAI access is configured for AzureOpenAIChatClient. You should be logged in with Azure CLI (AzureCliCredential)
and have the Azure OpenAI environment variables set as documented in the getting started chat client README.
- The sample email resource file exists at workflow/resources/email.txt.

High level flow:
1) spam_detection_agent reads an email and returns DetectionResult.
2) If not spam, we transform the detection output into a user message for email_assistant_agent, then finish by
yielding the drafted reply as workflow output.
3) If spam, we short circuit to a spam handler that yields a spam notice as workflow output.

Output:
- The final workflow output is printed to stdout, either with a drafted reply or a spam notice.

Notes:
- Conditions read the agent response text and validate it into DetectionResult for robust routing.
- Executors are small and single purpose to keep control flow easy to follow.
- The workflow completes when it becomes idle, not via explicit completion events.
"""


class DetectionResult(BaseModel):
    """スパム検出の結果を表します。"""

    # is_spamはエッジ条件によるルーティング決定を駆動します
    is_spam: bool
    # 検出器からの人間が読める根拠
    reason: str
    # エージェントは元のメールを含める必要があり、下流のエージェントはコンテンツを再読み込みせずに動作できます
    email_content: str


class EmailResponse(BaseModel):
    """メールアシスタントからのレスポンスを表します。"""

    # ユーザーがコピーまたは送信できるドラフト返信
    response: str


def get_condition(expected_result: bool):
    """DetectionResult.is_spamに基づいてルーティングする条件呼び出し可能を作成します。"""

    # 返される関数はエッジ述語として使用されます。 上流のexecutorが生成したものを受け取ります。
    def condition(message: Any) -> bool:
        # 防御的ガード。AgentExecutorResponseでないものが現れた場合、デッドエンドを避けるためにエッジを通過させます。
        if not isinstance(message, AgentExecutorResponse):
            return True

        try:
            # エージェントのJSONテキストから構造化されたDetectionResultを解析することを優先します。
            # model_validate_jsonを使うことで型安全が保証され、形状が間違っている場合は例外が発生します。
            detection = DetectionResult.model_validate_json(message.agent_run_response.text)
            # スパムフラグが期待されるパスと一致する場合のみルーティングします。
            return detection.is_spam == expected_result
        except Exception:
            # 解析エラー時は閉じる方向で失敗し、誤って間違ったパスにルーティングしないようにします。 Falseを返すことでこのエッジの活性化を防ぎます。
            return False

    return condition


@executor(id="send_email")
async def handle_email_response(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    # メールアシスタントの下流。検証済みのEmailResponseを解析し、ワークフロー出力を生成します。
    email_response = EmailResponse.model_validate_json(response.agent_run_response.text)
    await ctx.yield_output(f"Email sent:\n{email_response.response}")


@executor(id="handle_spam")
async def handle_spam_classifier_response(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    # スパムパス。DetectionResultを確認し、ワークフロー出力を生成します。誤って非スパム入力が来た場合に備えたガード。
    detection = DetectionResult.model_validate_json(response.agent_run_response.text)
    if detection.is_spam:
        await ctx.yield_output(f"Email marked as spam: {detection.reason}")
    else:
        # これはルーティング述語とexecutor契約が同期していないことを示します。
        raise RuntimeError("This executor should only handle spam messages.")


@executor(id="to_email_assistant_request")
async def to_email_assistant_request(
    response: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorRequest]
) -> None:
    """検出結果をメールアシスタント用のAgentExecutorRequestに変換します。

    DetectionResult.email_contentを抽出し、ユーザーメッセージとして転送します。

    """
    # ブリッジexecutor。構造化されたDetectionResultをChatMessageに変換し、新しいリクエストとして転送します。
    detection = DetectionResult.model_validate_json(response.agent_run_response.text)
    user_msg = ChatMessage(Role.USER, text=detection.email_content)
    await ctx.send_message(AgentExecutorRequest(messages=[user_msg], should_respond=True))


async def main() -> None:
    # エージェントを作成します AzureCliCredentialは現在のazログインを使用します。これによりコードにSecretを埋め込む必要がなくなります。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    # エージェント1。スパムを分類し、DetectionResultオブジェクトを返します。
    # response_formatはLLMがPydanticモデル用の解析可能なJSONを返すことを強制します。
    spam_detection_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You are a spam detection assistant that identifies spam emails. "
                "Always return JSON with fields is_spam (bool), reason (string), and email_content (string). "
                "Include the original email content in email_content."
            ),
            response_format=DetectionResult,
        ),
        id="spam_detection_agent",
    )

    # エージェント2。プロフェッショナルな返信をドラフトします。信頼性のために構造化JSON出力も使用します。
    email_assistant_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You are an email assistant that helps users draft professional responses to emails. "
                "Your input may be a JSON object that includes 'email_content'; base your reply on that content. "
                "Return JSON with a single field 'response' containing the drafted reply."
            ),
            response_format=EmailResponse,
        ),
        id="email_assistant_agent",
    )

    # ワークフローグラフを構築します。 スパム検出器から開始します。
    # スパムでなければ、新しいAgentExecutorRequestを作成するトランスフォーマーに移動し、 その後メールアシスタントを呼び出し、最後に完了します。
    # スパムの場合は直接スパムハンドラーに行き、完了します。
    workflow = (
        WorkflowBuilder()
        .set_start_executor(spam_detection_agent)
        # 非スパムパス：レスポンスを変換 -> アシスタントへのリクエスト -> アシスタント -> メール送信
        .add_edge(spam_detection_agent, to_email_assistant_request, condition=get_condition(False))
        .add_edge(to_email_assistant_request, email_assistant_agent)
        .add_edge(email_assistant_agent, handle_email_response)
        # スパムパス：スパムハンドラーに送信
        .add_edge(spam_detection_agent, handle_spam_classifier_response, condition=get_condition(True))
        .build()
    )

    # サンプルリソースファイルからメールコンテンツを読み込みます。 これにより、モデルが毎回同じメールを見るためサンプルが決定論的になります。
    email_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "resources", "email.txt")

    with open(email_path) as email_file:  # noqa: ASYNC230
        email = email_file.read()

    # ワークフローを実行します。開始がAgentExecutorなのでAgentExecutorRequestを渡します。
    # ワークフローはアイドル状態（作業なし）になると完了します。
    request = AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=email)], should_respond=True)
    events = await workflow.run(request)
    outputs = events.get_outputs()
    if outputs:
        print(f"Workflow output: {outputs[0]}")

    """
    Sample Output:

    Processing email:
    Subject: Team Meeting Follow-up - Action Items

    Hi Sarah,

    I wanted to follow up on our team meeting this morning and share the action items we discussed:

    1. Update the project timeline by Friday
    2. Schedule client presentation for next week
    3. Review the budget allocation for Q4

    Please let me know if you have any questions or if I missed anything from our discussion.

    Best regards,
    Alex Johnson
    Project Manager
    Tech Solutions Inc.
    alex.johnson@techsolutions.com
    (555) 123-4567
    ----------------------------------------

Workflow output: Email sent:
    Hi Alex,

    Thank you for the follow-up and for summarizing the action items from this morning's meeting. The points you listed accurately reflect our discussion, and I don't have any additional items to add at this time.

    I will update the project timeline by Friday, begin scheduling the client presentation for next week, and start reviewing the Q4 budget allocation. If any questions or issues arise, I'll reach out.

    Thank you again for outlining the next steps.

    Best regards,
    Sarah
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
