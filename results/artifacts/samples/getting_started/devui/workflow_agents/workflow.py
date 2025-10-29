# Copyright (c) Microsoft. All rights reserved.

"""Agent Workflow - Content Review with Quality Routing.

このサンプルは以下を示します:
- Agentを直接Executorとして使用
- 構造化された出力に基づく条件付きルーティング
- 品質に基づくワークフローパスと収束

ユースケース: 自動レビュー付きコンテンツ作成。
Writerがコンテンツを作成し、Reviewerが品質を評価:
  - 高品質（スコア >= 80）: → Publisher → Summarizer
  - 低品質（スコア < 80）: → Editor → Publisher → Summarizer
両パスはSummarizerで最終レポートに収束します。
"""

import os
from typing import Any

from agent_framework import AgentExecutorResponse, WorkflowBuilder
from agent_framework.azure import AzureOpenAIChatClient
from pydantic import BaseModel


# レビュー結果の構造化出力を定義します。
class ReviewResult(BaseModel):
    """スコアとフィードバックによるレビュー評価。"""

    score: int  # 総合品質スコア（0-100）
    feedback: str  # 簡潔で実行可能なフィードバック
    clarity: int  # 明瞭さスコア（0-100）
    completeness: int  # 完全性スコア（0-100）
    accuracy: int  # 正確性スコア（0-100）
    structure: int  # 構造スコア（0-100）


# 条件関数: スコアが80未満ならEditorへルーティングします。
def needs_editing(message: Any) -> bool:
    """レビューのスコアに基づき編集が必要かチェックします。"""
    if not isinstance(message, AgentExecutorResponse):
        return False
    try:
        review = ReviewResult.model_validate_json(message.agent_run_response.text)
        return review.score < 80
    except Exception:
        return False


# 条件関数: コンテンツが承認された（スコア >= 80）場合。
def is_approved(message: Any) -> bool:
    """コンテンツが承認されたか（高品質か）をチェックします。"""
    if not isinstance(message, AgentExecutorResponse):
        return True
    try:
        review = ReviewResult.model_validate_json(message.agent_run_response.text)
        return review.score >= 80
    except Exception:
        return True


# Azure OpenAIのChat Clientを作成します。
chat_client = AzureOpenAIChatClient(api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""))

# Writer Agentを作成します - コンテンツを生成します。
writer = chat_client.create_agent(
    name="Writer",
    instructions=(
        "You are an excellent content writer. "
        "Create clear, engaging content based on the user's request. "
        "Focus on clarity, accuracy, and proper structure."
    ),
)

# Reviewer Agentを作成します - 評価し構造化フィードバックを提供します。
reviewer = chat_client.create_agent(
    name="Reviewer",
    instructions=(
        "You are an expert content reviewer. "
        "Evaluate the writer's content based on:\n"
        "1. Clarity - Is it easy to understand?\n"
        "2. Completeness - Does it fully address the topic?\n"
        "3. Accuracy - Is the information correct?\n"
        "4. Structure - Is it well-organized?\n\n"
        "Return a JSON object with:\n"
        "- score: overall quality (0-100)\n"
        "- feedback: concise, actionable feedback\n"
        "- clarity, completeness, accuracy, structure: individual scores (0-100)"
    ),
    response_format=ReviewResult,
)

# Editor Agentを作成します - フィードバックに基づきコンテンツを改善します。
editor = chat_client.create_agent(
    name="Editor",
    instructions=(
        "You are a skilled editor. "
        "You will receive content along with review feedback. "
        "Improve the content by addressing all the issues mentioned in the feedback. "
        "Maintain the original intent while enhancing clarity, completeness, accuracy, and structure."
    ),
)

# Publisher Agentを作成します - 出版用にコンテンツをフォーマットします。
publisher = chat_client.create_agent(
    name="Publisher",
    instructions=(
        "You are a publishing agent. "
        "You receive either approved content or edited content. "
        "Format it for publication with proper headings and structure."
    ),
)

# Summarizer Agentを作成します - 最終出版レポートを作成します。
summarizer = chat_client.create_agent(
    name="Summarizer",
    instructions=(
        "You are a summarizer agent. "
        "Create a final publication report that includes:\n"
        "1. A brief summary of the published content\n"
        "2. The workflow path taken (direct approval or edited)\n"
        "3. Key highlights and takeaways\n"
        "Keep it concise and professional."
    ),
)

# 分岐と収束を含むワークフローを構築します: Writer → Reviewer → [分岐]: - スコア >= 80: → Publisher →
# Summarizer（直接承認パス） - スコア < 80: → Editor → Publisher → Summarizer（改善パス）
# 両パスはSummarizerで最終レポートに収束します。
workflow = (
    WorkflowBuilder(
        name="Content Review Workflow",
        description="Multi-agent content creation workflow with quality-based routing (Writer → Reviewer → Editor/Publisher)",
    )
    .set_start_executor(writer)
    .add_edge(writer, reviewer)
    # 分岐1: 高品質（>= 80）は直接Publisherへ。
    .add_edge(reviewer, publisher, condition=is_approved)
    # 分岐2: 低品質（< 80）はまずEditorへ、その後Publisherへ。
    .add_edge(reviewer, editor, condition=needs_editing)
    .add_edge(editor, publisher)
    # 両パスはPublisher → Summarizerで収束します。
    .add_edge(publisher, summarizer)
    .build()
)


def main():
    """DevUIで分岐ワークフローを起動します。"""
    import logging

    from agent_framework.devui import serve

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Agent Workflow (Content Review with Quality Routing)")
    logger.info("Available at: http://localhost:8093")
    logger.info("\nThis workflow demonstrates:")
    logger.info("- Conditional routing based on structured outputs")
    logger.info("- Path 1 (score >= 80): Reviewer → Publisher → Summarizer")
    logger.info("- Path 2 (score < 80): Reviewer → Editor → Publisher → Summarizer")
    logger.info("- Both paths converge at Summarizer for final report")

    serve(entities=[workflow], port=8093, auto_open=True)


if __name__ == "__main__":
    main()
