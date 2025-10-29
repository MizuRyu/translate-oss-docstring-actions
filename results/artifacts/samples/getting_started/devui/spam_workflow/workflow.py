# Copyright (c) Microsoft. All rights reserved.

"""Spam Detection Workflow Sample for DevUI.

以下のサンプルは、複数のExecutorを用いた包括的な5ステップのワークフローを示します。
このワークフローはメールメッセージを処理、分析、スパム検出、対応します。
複雑な分岐ロジックと現実的な処理遅延を示し、ワークフローフレームワークを実証します。

Workflow Steps:
1. Email Preprocessor - メールをクリーンアップし準備します
2. Content Analyzer - メールの内容と構造を分析します
3. Spam Detector - メッセージがスパムかどうか判定します
4a. Spam Handler - スパムメッセージを隔離、ログ記録、削除します
4b. Message Responder - 正当なメッセージを検証し応答します
5. Final Processor - ロギングとクリーンアップでワークフローを完了します
"""

import asyncio
import logging
from dataclasses import dataclass

from agent_framework import (
    Case,
    Default,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from pydantic import BaseModel, Field
from typing_extensions import Never


@dataclass
class EmailContent:
    """処理済みメール内容を保持するデータクラス。"""

    original_message: str
    cleaned_message: str
    word_count: int
    has_suspicious_patterns: bool = False


@dataclass
class ContentAnalysis:
    """内容分析結果を保持するデータクラス。"""

    email_content: EmailContent
    sentiment_score: float
    contains_links: bool
    has_attachments: bool
    risk_indicators: list[str]


@dataclass
class SpamDetectorResponse:
    """スパム検出結果を保持するデータクラス。"""

    analysis: ContentAnalysis
    is_spam: bool = False
    confidence_score: float = 0.0
    spam_reasons: list[str] | None = None

    def __post_init__(self):
        """spam_reasonsリストをNoneの場合に初期化します。"""
        if self.spam_reasons is None:
            self.spam_reasons = []


@dataclass
class ProcessingResult:
    """最終処理結果を保持するデータクラス。"""

    original_message: str
    action_taken: str
    processing_time: float
    status: str
    is_spam: bool
    confidence_score: float
    spam_reasons: list[str]


class EmailRequest(BaseModel):
    """メール処理用のRequestモデル。"""

    email: str = Field(
        description="The email message to be processed.",
        default="Hi there, are you interested in our new urgent offer today? Click here!",
    )


class EmailPreprocessor(Executor):
    """ステップ1: メール内容を前処理しクリーンアップするExecutor。"""

    @handler
    async def handle_email(self, email: EmailRequest, ctx: WorkflowContext[EmailContent]) -> None:
        """メールメッセージをクリーンアップし前処理します。"""
        await asyncio.sleep(1.5)  # 前処理時間をシミュレートします。

        # メールのクリーンアップをシミュレートします。
        cleaned = email.email.strip().lower()
        word_count = len(email.email.split())

        # 疑わしいパターンをチェックします。
        suspicious_patterns = ["urgent", "limited time", "act now", "free money"]
        has_suspicious = any(pattern in cleaned for pattern in suspicious_patterns)

        result = EmailContent(
            original_message=email.email,
            cleaned_message=cleaned,
            word_count=word_count,
            has_suspicious_patterns=has_suspicious,
        )

        await ctx.send_message(result)


class ContentAnalyzer(Executor):
    """ステップ2: メール内容と構造を分析するExecutor。"""

    @handler
    async def handle_email_content(self, email_content: EmailContent, ctx: WorkflowContext[ContentAnalysis]) -> None:
        """メール内容の各種指標を分析します。"""
        await asyncio.sleep(2.0)  # 分析時間をシミュレートします。

        # 内容分析をシミュレートします。
        sentiment_score = 0.5 if email_content.has_suspicious_patterns else 0.8
        contains_links = "http" in email_content.cleaned_message or "www" in email_content.cleaned_message
        has_attachments = "attachment" in email_content.cleaned_message

        # リスク指標を構築します。
        risk_indicators: list[str] = []
        if email_content.has_suspicious_patterns:
            risk_indicators.append("suspicious_language")
        if contains_links:
            risk_indicators.append("contains_links")
        if has_attachments:
            risk_indicators.append("has_attachments")
        if email_content.word_count < 10:
            risk_indicators.append("too_short")

        analysis = ContentAnalysis(
            email_content=email_content,
            sentiment_score=sentiment_score,
            contains_links=contains_links,
            has_attachments=has_attachments,
            risk_indicators=risk_indicators,
        )

        await ctx.send_message(analysis)


class SpamDetector(Executor):
    """ステップ3: 分析に基づきメッセージがスパムか判定するExecutor。"""

    def __init__(self, spam_keywords: list[str], id: str):
        """スパムキーワードでExecutorを初期化します。"""
        super().__init__(id=id)
        self._spam_keywords = spam_keywords

    @handler
    async def handle_analysis(self, analysis: ContentAnalysis, ctx: WorkflowContext[SpamDetectorResponse]) -> None:
        """内容分析に基づきメッセージがスパムか判定します。"""
        await asyncio.sleep(1.8)  # 検出時間をシミュレートします。

        # スパムキーワードをチェックします。
        email_text = analysis.email_content.cleaned_message
        keyword_matches = [kw for kw in self._spam_keywords if kw in email_text]

        # スパム確率を計算します。
        spam_score = 0.0
        spam_reasons: list[str] = []

        if keyword_matches:
            spam_score += 0.4
            spam_reasons.append(f"spam_keywords: {keyword_matches}")

        if analysis.email_content.has_suspicious_patterns:
            spam_score += 0.3
            spam_reasons.append("suspicious_patterns")

        if len(analysis.risk_indicators) >= 3:
            spam_score += 0.2
            spam_reasons.append("high_risk_indicators")

        if analysis.sentiment_score < 0.4:
            spam_score += 0.1
            spam_reasons.append("negative_sentiment")

        is_spam = spam_score >= 0.5

        result = SpamDetectorResponse(
            analysis=analysis, is_spam=is_spam, confidence_score=spam_score, spam_reasons=spam_reasons
        )

        await ctx.send_message(result)


class SpamHandler(Executor):
    """ステップ4a: スパムメッセージを隔離しログ記録するExecutor。"""

    @handler
    async def handle_spam_detection(
        self,
        spam_result: SpamDetectorResponse,
        ctx: WorkflowContext[ProcessingResult],
    ) -> None:
        """スパムメッセージを隔離しログ記録します。"""
        if not spam_result.is_spam:
            raise RuntimeError("Message is not spam, cannot process with spam handler.")

        await asyncio.sleep(2.2)  # スパム処理時間をシミュレートします。

        result = ProcessingResult(
            original_message=spam_result.analysis.email_content.original_message,
            action_taken="quarantined_and_logged",
            processing_time=2.2,
            status="spam_handled",
            is_spam=spam_result.is_spam,
            confidence_score=spam_result.confidence_score,
            spam_reasons=spam_result.spam_reasons or [],
        )

        await ctx.send_message(result)


class MessageResponder(Executor):
    """ステップ4b: 正当なメッセージに応答するExecutor。"""

    @handler
    async def handle_spam_detection(
        self,
        spam_result: SpamDetectorResponse,
        ctx: WorkflowContext[ProcessingResult],
    ) -> None:
        """正当なメッセージに応答します。"""
        if spam_result.is_spam:
            raise RuntimeError("Message is spam, cannot respond with message responder.")

        await asyncio.sleep(2.5)  # 応答時間をシミュレートします。

        result = ProcessingResult(
            original_message=spam_result.analysis.email_content.original_message,
            action_taken="responded_and_filed",
            processing_time=2.5,
            status="message_processed",
            is_spam=spam_result.is_spam,
            confidence_score=spam_result.confidence_score,
            spam_reasons=spam_result.spam_reasons or [],
        )

        await ctx.send_message(result)


class FinalProcessor(Executor):
    """ステップ5: 最終ロギングとクリーンアップでワークフローを完了するExecutor。"""

    @handler
    async def handle_processing_result(
        self,
        result: ProcessingResult,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        """最終処理とロギングでワークフローを完了します。"""
        await asyncio.sleep(1.5)  # 最終処理時間をシミュレートします。

        total_time = result.processing_time + 1.5

        # 完了メッセージに分類詳細を含めます。
        classification = "SPAM" if result.is_spam else "LEGITIMATE"
        reasons = ", ".join(result.spam_reasons) if result.spam_reasons else "none"

        completion_message = (
            f"Email classified as {classification} (confidence: {result.confidence_score:.2f}). "
            f"Reasons: {reasons}. "
            f"Action: {result.action_taken}, "
            f"Status: {result.status}, "
            f"Total time: {total_time:.1f}s"
        )

        await ctx.yield_output(completion_message)


# DevUIが検出可能なワークフローインスタンスを作成します。
spam_keywords = ["spam", "advertisement", "offer", "click here", "winner", "congratulations", "urgent"]

# 5ステップワークフローの全Executorを作成します。
email_preprocessor = EmailPreprocessor(id="email_preprocessor")
content_analyzer = ContentAnalyzer(id="content_analyzer")
spam_detector = SpamDetector(spam_keywords, id="spam_detector")
spam_handler = SpamHandler(id="spam_handler")
message_responder = MessageResponder(id="message_responder")
final_processor = FinalProcessor(id="final_processor")

# 分岐ロジックを含む包括的な5ステップワークフローを構築します。
workflow = (
    WorkflowBuilder(
        name="Email Spam Detector",
        description="5-step email classification workflow with spam/legitimate routing",
    )
    .set_start_executor(email_preprocessor)
    .add_edge(email_preprocessor, content_analyzer)
    .add_edge(content_analyzer, spam_detector)
    .add_switch_case_edge_group(
        spam_detector,
        [
            Case(condition=lambda x: x.is_spam, target=spam_handler),
            Default(target=message_responder),
        ],
    )
    .add_edge(spam_handler, final_processor)
    .add_edge(message_responder, final_processor)
    .build()
)

# 注意: ワークフローメタデータはExecutorとグラフ構造で決定されます。


def main():
    """DevUIでスパム検出ワークフローを起動します。"""
    from agent_framework.devui import serve

    # ログのセットアップ
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Spam Detection Workflow")
    logger.info("Available at: http://localhost:8090")
    logger.info("Entity ID: workflow_spam_detection")

    # ワークフローでサーバーを起動します。
    serve(entities=[workflow], port=8090, auto_open=True)


if __name__ == "__main__":
    main()
