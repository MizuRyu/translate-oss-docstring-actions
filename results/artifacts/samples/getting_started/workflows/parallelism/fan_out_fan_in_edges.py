# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass

from agent_framework import (  # Core chat primitives to build LLM requests
    AgentExecutor,  # Wraps an LLM agent for use inside a workflow
    AgentExecutorRequest,  # The message bundle sent to an AgentExecutor
    AgentExecutorResponse,  # The structured result returned by an AgentExecutor
    AgentRunEvent,  # Tracing event for agent execution steps
    ChatMessage,  # Chat message structure
    Executor,  # Base class for custom Python executors
    Role,  # Enum of chat roles (user, assistant, system)
    WorkflowBuilder,  # Fluent builder for wiring the workflow graph
    WorkflowContext,  # Per run context and event bus
    WorkflowOutputEvent,  # Event emitted when workflow yields output
    handler,  # Decorator to mark an Executor method as invokable
)
from agent_framework.azure import AzureOpenAIChatClient  # Azure OpenAIチャットモデルのクライアントラッパー
from azure.identity import AzureCliCredential  # az CLIログインを認証情報として使用します
from typing_extensions import Never

"""
Sample: Concurrent fan out and fan in with three domain agents

A dispatcher fans out the same user prompt to research, marketing, and legal AgentExecutor nodes.
An aggregator then fans in their responses and produces a single consolidated report.

Purpose:
Show how to construct a parallel branch pattern in workflows. Demonstrate:
- Fan out by targeting multiple AgentExecutor nodes from one dispatcher.
- Fan in by collecting a list of AgentExecutorResponse objects and reducing them to a single result.
- Simple tracing using AgentRunEvent to observe execution order and progress.

Prerequisites:
- Familiarity with WorkflowBuilder, executors, edges, events, and streaming runs.
- Azure OpenAI access configured for AzureOpenAIChatClient. Log in with Azure CLI and set any required environment variables.
- Comfort reading AgentExecutorResponse.agent_run_response.text for assistant output aggregation.
"""


class DispatchToExperts(Executor):
    """受信したPromptをすべての専門Agent executorに並列処理のためにディスパッチします（fan out）。"""

    def __init__(self, expert_ids: list[str], id: str | None = None):
        super().__init__(id=id or "dispatch_to_experts")
        self._expert_ids = expert_ids

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        # 受信したPromptを各専門家のユーザーメッセージとしてラップし、レスポンスを要求します。
        # 各send_messageはidで異なるAgentExecutorをターゲットにし、ブランチを並列実行します。
        initial_message = ChatMessage(Role.USER, text=prompt)
        for expert_id in self._expert_ids:
            await ctx.send_message(
                AgentExecutorRequest(messages=[initial_message], should_respond=True),
                target_id=expert_id,
            )


@dataclass
class AggregatedInsights:
    """フォーマッティング前にドメインごとの文字列を保持するための型付きコンテナ。"""

    research: str
    marketing: str
    legal: str


class AggregateInsights(Executor):
    """専門Agentのレスポンスを単一の統合結果に集約します（fan in）。"""

    def __init__(self, expert_ids: list[str], id: str | None = None):
        super().__init__(id=id or "aggregate_insights")
        self._expert_ids = expert_ids

    @handler
    async def aggregate(self, results: list[AgentExecutorResponse], ctx: WorkflowContext[Never, str]) -> None:
        # レスポンスをexecutor idごとにテキストにマッピングし、シンプルで予測可能なデモを実現します。
        by_id: dict[str, str] = {}
        for r in results:
            # AgentExecutorResponse.agent_run_response.textはAgentが生成したアシスタントテキストです。
            by_id[r.executor_id] = r.agent_run_response.text

        research_text = by_id.get("researcher", "")
        marketing_text = by_id.get("marketer", "")
        legal_text = by_id.get("legal", "")

        aggregated = AggregatedInsights(
            research=research_text,
            marketing=marketing_text,
            legal=legal_text,
        )

        # 最終的なワークフロー結果として読みやすく統合された文字列を提供します。
        consolidated = (
            "Consolidated Insights\n"
            "====================\n\n"
            f"Research Findings:\n{aggregated.research}\n\n"
            f"Marketing Angle:\n{aggregated.marketing}\n\n"
            f"Legal/Compliance Notes:\n{aggregated.legal}\n"
        )

        await ctx.yield_output(consolidated)


async def main() -> None:
    # ドメインエキスパートのためのAgent Executorを作成します。
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    researcher = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You're an expert market and product researcher. Given a prompt, provide concise, factual insights,"
                " opportunities, and risks."
            ),
        ),
        id="researcher",
    )
    marketer = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You're a creative marketing strategist. Craft compelling value propositions and target messaging"
                " aligned to the prompt."
            ),
        ),
        id="marketer",
    )
    legal = AgentExecutor(
        chat_client.create_agent(
            instructions=(
                "You're a cautious legal/compliance reviewer. Highlight constraints, disclaimers, and policy concerns"
                " based on the prompt."
            ),
        ),
        id="legal",
    )

    expert_ids = [researcher.id, marketer.id, legal.id]

    dispatcher = DispatchToExperts(expert_ids=expert_ids, id="dispatcher")
    aggregator = AggregateInsights(expert_ids=expert_ids, id="aggregator")

    # シンプルなfan outおよびfan inワークフローを構築します。
    workflow = (
        WorkflowBuilder()
        .set_start_executor(dispatcher)
        .add_fan_out_edges(dispatcher, [researcher, marketer, legal])  # Parallel branches
        .add_fan_in_edges([researcher, marketer, legal], aggregator)  # Join at the aggregator
        .build()
    )

    # 単一のプロンプトで実行し、進行状況と最終的な統合出力を表示します。
    async for event in workflow.run_stream("We are launching a new budget-friendly electric bike for urban commuters."):
        if isinstance(event, AgentRunEvent):
            # どのAgentが実行され、どのステップが完了したかを表示して軽量な可観測性を提供します。
            print(event)
        elif isinstance(event, WorkflowOutputEvent):
            print("===== Final Aggregated Output =====")
            print(event.data)


if __name__ == "__main__":
    asyncio.run(main())
