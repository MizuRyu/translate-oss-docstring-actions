# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentRunEvent,
    ChatMessage,
    Executor,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    WorkflowViz,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from typing_extensions import Never

"""
Sample: Concurrent (Fan-out/Fan-in) with Agents + Visualization

What it does:
- Fan-out: dispatch the same prompt to multiple domain agents (research, marketing, legal).
- Fan-in: aggregate their responses into one consolidated output.
- Visualization: generate Mermaid and GraphViz representations via `WorkflowViz` and optionally export SVG.

Prerequisites:
- Azure AI/ Azure OpenAI for `AzureOpenAIChatClient` agents.
- Authentication via `azure-identity` — uses `AzureCliCredential()` (run `az login`).
- For visualization export: `pip install agent-framework[viz] --pre` and install GraphViz binaries.
"""


class DispatchToExperts(Executor):
    """受信したプロンプトをすべての専門Agent Executorにディスパッチします（fan-out）。"""

    def __init__(self, expert_ids: list[str], id: str | None = None):
        super().__init__(id=id or "dispatch_to_experts")
        self._expert_ids = expert_ids

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        # 受信したプロンプトを各専門家のユーザーメッセージとしてラップし、応答を要求します。
        initial_message = ChatMessage(Role.USER, text=prompt)
        for expert_id in self._expert_ids:
            await ctx.send_message(
                AgentExecutorRequest(messages=[initial_message], should_respond=True),
                target_id=expert_id,
            )


@dataclass
class AggregatedInsights:
    """aggregatorからの構造化出力。"""

    research: str
    marketing: str
    legal: str


class AggregateInsights(Executor):
    """専門Agentの応答を単一の統合結果に集約します（fan-in）。"""

    def __init__(self, expert_ids: list[str], id: str | None = None):
        super().__init__(id=id or "aggregate_insights")
        self._expert_ids = expert_ids

    @handler
    async def aggregate(self, results: list[AgentExecutorResponse], ctx: WorkflowContext[Never, str]) -> None:
        # executor idごとに応答をテキストにマップし、シンプルで予測可能なデモを実現します。
        by_id: dict[str, str] = {}
        for r in results:
            # AgentExecutorResponse.agent_run_response.textには連結されたassistantのテキストが含まれます。
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

    # シンプルなfan-out/fan-inワークフローを構築します。
    workflow = (
        WorkflowBuilder()
        .set_start_executor(dispatcher)
        .add_fan_out_edges(dispatcher, [researcher, marketer, legal])
        .add_fan_in_edges([researcher, marketer, legal], aggregator)
        .build()
    )

    # ワークフローの可視化を生成します。
    print("Generating workflow visualization...")
    viz = WorkflowViz(workflow)
    # mermaid文字列を出力します。
    print("Mermaid string: \n=======")
    print(viz.to_mermaid())
    print("=======")
    # DiGraph文字列を出力します。
    print("DiGraph string: \n=======")
    print(viz.to_digraph())
    print("=======")
    try:
        # DiGraphの可視化をSVGとしてエクスポートします。
        svg_file = viz.export(format="svg")
        print(f"SVG file saved to: {svg_file}")
    except ImportError:
        print("Tip: Install 'viz' extra to export workflow visualization: pip install agent-framework[viz] --pre")

    # 単一のプロンプトで実行します。
    async for event in workflow.run_stream("We are launching a new budget-friendly electric bike for urban commuters."):
        if isinstance(event, AgentRunEvent):
            # どのAgentが実行され、どのステップが完了したかを表示します。
            print(event)
        elif isinstance(event, WorkflowOutputEvent):
            print("===== Final Aggregated Output =====")
            print(event.data)


if __name__ == "__main__":
    asyncio.run(main())
