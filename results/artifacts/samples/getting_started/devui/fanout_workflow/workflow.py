# Copyright (c) Microsoft. All rights reserved.

"""複雑なFan-In/Fan-Outデータ処理ワークフロー。

このワークフローは複数のステージを持つ高度なデータ処理パイプラインを示しています：
1. データ取り込み - 複数のソースからのデータ読み込みをシミュレート
2. データ検証 - 複数のバリデーターが並列でデータ品質をチェック
3. データ変換 - 異なる変換プロセッサへのファンアウト
4. 品質保証 - 複数のQAチェックが並列で実行
5. データ集約 - 処理結果をファンインで結合
6. 最終処理 - レポート生成とワークフロー完了

ワークフローには実際の処理時間をシミュレートする現実的な遅延が含まれており、
条件付き処理を伴う複雑なファンイン/ファンアウトパターンを示しています。
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from pydantic import BaseModel, Field
from typing_extensions import Never


class DataType(Enum):
    """処理されるデータの種類。"""

    CUSTOMER = "customer"
    TRANSACTION = "transaction"
    PRODUCT = "product"
    ANALYTICS = "analytics"


class ValidationResult(Enum):
    """データ検証の結果。"""

    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"


class ProcessingRequest(BaseModel):
    """データ処理ワークフローの複雑な入力構造。"""

    # 基本情報
    data_source: Literal["database", "api", "file_upload", "streaming"] = Field(
        description="The source of the data to be processed", default="database"
    )

    data_type: Literal["customer", "transaction", "product", "analytics"] = Field(
        description="Type of data being processed", default="customer"
    )

    processing_priority: Literal["low", "normal", "high", "critical"] = Field(
        description="Processing priority level", default="normal"
    )

    # 処理設定
    batch_size: int = Field(description="Number of records to process in each batch", default=500, ge=100, le=10000)

    quality_threshold: float = Field(
        description="Minimum quality score required (0.0-1.0)", default=0.8, ge=0.0, le=1.0
    )

    # 検証設定
    enable_schema_validation: bool = Field(description="Enable schema validation checks", default=True)

    enable_security_validation: bool = Field(description="Enable security validation checks", default=True)

    enable_quality_validation: bool = Field(description="Enable data quality validation checks", default=True)

    # 変換オプション
    transformations: list[Literal["normalize", "enrich", "aggregate"]] = Field(
        description="List of transformations to apply", default=["normalize", "enrich"]
    )

    # 任意の説明
    description: str | None = Field(description="Optional description of the processing request", default=None)

    # テスト失敗シナリオ
    force_validation_failure: bool = Field(
        description="Force validation failure for testing (demo purposes)", default=False
    )

    force_transformation_failure: bool = Field(
        description="Force transformation failure for testing (demo purposes)", default=False
    )


@dataclass
class DataBatch:
    """処理中のデータバッチを表す。"""

    batch_id: str
    data_type: DataType
    size: int
    content: str
    source: str = "unknown"
    timestamp: float = 0.0


@dataclass
class ValidationReport:
    """データ検証のレポート。"""

    batch_id: str
    validator_id: str
    result: ValidationResult
    issues_found: int
    processing_time: float
    details: str


@dataclass
class TransformationResult:
    """データ変換の結果。"""

    batch_id: str
    transformer_id: str
    original_size: int
    processed_size: int
    transformation_type: str
    processing_time: float
    success: bool


@dataclass
class QualityAssessment:
    """品質評価結果。"""

    batch_id: str
    assessor_id: str
    quality_score: float
    recommendations: list[str]
    processing_time: float


@dataclass
class ProcessingSummary:
    """すべての処理ステージの概要。"""

    batch_id: str
    total_processing_time: float
    validation_reports: list[ValidationReport]
    transformation_results: list[TransformationResult]
    quality_assessments: list[QualityAssessment]
    final_status: str


# データ取り込みステージ
class DataIngestion(Executor):
    """複数のソースからのデータ取り込みを遅延付きでシミュレート。"""

    @handler
    async def ingest_data(self, request: ProcessingRequest, ctx: WorkflowContext[DataBatch]) -> None:
        """入力設定に基づいた現実的な遅延でデータ取り込みをシミュレート。"""
        # データソースに基づくネットワーク遅延のシミュレーション
        delay_map = {"database": 1.5, "api": 3.0, "file_upload": 4.0, "streaming": 1.0}
        delay = delay_map.get(request.data_source, 3.0)
        await asyncio.sleep(delay)  # デモ用の固定遅延

        # 優先度と設定に基づくデータサイズのシミュレーション
        base_size = request.batch_size
        if request.processing_priority == "critical":
            size_multiplier = 1.7  # クリティカル優先度は最大のバッチを取得
        elif request.processing_priority == "high":
            size_multiplier = 1.3  # 高優先度は大きめのバッチを取得
        elif request.processing_priority == "low":
            size_multiplier = 0.6  # 低優先度は小さめのバッチを取得
        else:  # normal
            size_multiplier = 1.0  # 通常優先度は基本サイズを使用

        actual_size = int(base_size * size_multiplier)

        batch = DataBatch(
            batch_id=f"batch_{5555}",  # Fixed batch ID for demo
            data_type=DataType(request.data_type),
            size=actual_size,
            content=f"Processing {request.data_type} data from {request.data_source}",
            source=request.data_source,
            timestamp=asyncio.get_event_loop().time(),
        )

        # バッチデータと元のRequestの両方を共有Stateに保存
        await ctx.set_shared_state(f"batch_{batch.batch_id}", batch)
        await ctx.set_shared_state(f"request_{batch.batch_id}", request)

        await ctx.send_message(batch)


# 検証ステージ（ファンアウト）
class SchemaValidator(Executor):
    """データのスキーマと構造を検証。"""

    @handler
    async def validate_schema(self, batch: DataBatch, ctx: WorkflowContext[ValidationReport]) -> None:
        """処理遅延を伴うスキーマ検証を実行。"""
        # スキーマ検証が有効かチェック
        request = await ctx.get_shared_state(f"request_{batch.batch_id}")
        if not request or not request.enable_schema_validation:
            return

        # スキーマ検証処理をシミュレート
        processing_time = 2.0  # 固定処理時間
        await asyncio.sleep(processing_time)

        # 検証結果をシミュレート - 強制失敗フラグを考慮
        issues = 4 if request.force_validation_failure else 2  # 固定の問題数

        result = (
            ValidationResult.VALID
            if issues <= 1
            else (ValidationResult.WARNING if issues <= 2 else ValidationResult.ERROR)
        )

        report = ValidationReport(
            batch_id=batch.batch_id,
            validator_id=self.id,
            result=result,
            issues_found=issues,
            processing_time=processing_time,
            details=f"Schema validation found {issues} issues in {batch.data_type.value} data from {batch.source}",
        )

        await ctx.send_message(report)


class DataQualityValidator(Executor):
    """データの品質と完全性を検証。"""

    @handler
    async def validate_quality(self, batch: DataBatch, ctx: WorkflowContext[ValidationReport]) -> None:
        """データ品質検証を実行。"""
        # 品質検証が有効かチェック
        request = await ctx.get_shared_state(f"request_{batch.batch_id}")
        if not request or not request.enable_quality_validation:
            return

        processing_time = 2.5  # 固定処理時間
        await asyncio.sleep(processing_time)

        # 高優先度データはより厳しい品質チェック
        issues = (
            2  # Fixed issue count for high priority
            if request.processing_priority in ["critical", "high"]
            else 3  # Fixed issue count for normal priority
        )

        if request.force_validation_failure:
            issues = max(issues, 4)  # 失敗を確実にする

        result = (
            ValidationResult.VALID
            if issues <= 1
            else (ValidationResult.WARNING if issues <= 3 else ValidationResult.ERROR)
        )

        report = ValidationReport(
            batch_id=batch.batch_id,
            validator_id=self.id,
            result=result,
            issues_found=issues,
            processing_time=processing_time,
            details=f"Quality check found {issues} data quality issues (priority: {request.processing_priority})",
        )

        await ctx.send_message(report)


class SecurityValidator(Executor):
    """セキュリティとコンプライアンスの問題を検証。"""

    @handler
    async def validate_security(self, batch: DataBatch, ctx: WorkflowContext[ValidationReport]) -> None:
        """セキュリティ検証を実行。"""
        # セキュリティ検証が有効かチェック
        request = await ctx.get_shared_state(f"request_{batch.batch_id}")
        if not request or not request.enable_security_validation:
            return

        processing_time = 3.0  # 固定処理時間
        await asyncio.sleep(processing_time)

        # 顧客/取引データはより厳格なセキュリティ
        issues = 1 if batch.data_type in [DataType.CUSTOMER, DataType.TRANSACTION] else 2

        if request.force_validation_failure:
            issues = max(issues, 1)  # 少なくとも1つのセキュリティ問題を強制

        # セキュリティエラーはより深刻で許容度が低い
        result = ValidationResult.VALID if issues == 0 else ValidationResult.ERROR

        report = ValidationReport(
            batch_id=batch.batch_id,
            validator_id=self.id,
            result=result,
            issues_found=issues,
            processing_time=processing_time,
            details=f"Security scan found {issues} security issues in {batch.data_type.value} data",
        )

        await ctx.send_message(report)


# 検証集約器（ファンイン）
class ValidationAggregator(Executor):
    """検証結果を集約し次のステップを決定。"""

    @handler
    async def aggregate_validations(
        self, reports: list[ValidationReport], ctx: WorkflowContext[DataBatch, str]
    ) -> None:
        """すべての検証レポートを集約し処理判断を行う。"""
        if not reports:
            return

        batch_id = reports[0].batch_id
        request = await ctx.get_shared_state(f"request_{batch_id}")

        await asyncio.sleep(1)  # 集約処理時間

        total_issues = sum(report.issues_found for report in reports)
        has_errors = any(report.result == ValidationResult.ERROR for report in reports)

        # 品質スコア（0.0から1.0）を計算
        max_possible_issues = len(reports) * 5  # バリデーターごとに最大5件の問題を想定
        quality_score = max(0.0, 1.0 - (total_issues / max_possible_issues))

        # 判断ロジック：エラーがあるか品質が閾値以下なら失敗
        should_fail = has_errors or (quality_score < request.quality_threshold)

        if should_fail:
            failure_reason: list[str] = []
            if has_errors:
                failure_reason.append("validation errors detected")
            if quality_score < request.quality_threshold:
                failure_reason.append(
                    f"quality score {quality_score:.2f} below threshold {request.quality_threshold:.2f}"
                )

            reason = " and ".join(failure_reason)
            await ctx.yield_output(
                f"Batch {batch_id} failed validation: {reason}. "
                f"Total issues: {total_issues}, Quality score: {quality_score:.2f}"
            )
            return

        # 共有Stateから元のバッチを取得
        batch_data = await ctx.get_shared_state(f"batch_{batch_id}")
        if batch_data:
            await ctx.send_message(batch_data)
        else:
            # フォールバック：簡略化したバッチを作成
            batch = DataBatch(
                batch_id=batch_id,
                data_type=DataType.ANALYTICS,
                size=500,
                content="Validated data ready for transformation",
            )
            await ctx.send_message(batch)


# 変換ステージ（ファンアウト）
class DataNormalizer(Executor):
    """データを正規化しクリーンアップ。"""

    @handler
    async def normalize_data(self, batch: DataBatch, ctx: WorkflowContext[TransformationResult]) -> None:
        """データ正規化を実行。"""
        request = await ctx.get_shared_state(f"request_{batch.batch_id}")

        # 正規化が有効かチェック
        if not request or "normalize" not in request.transformations:
            # "skipped"結果を送信
            result = TransformationResult(
                batch_id=batch.batch_id,
                transformer_id=self.id,
                original_size=batch.size,
                processed_size=batch.size,
                transformation_type="normalization",
                processing_time=0.1,
                success=True,  # Consider skipped as successful
            )
            await ctx.send_message(result)
            return

        processing_time = 4.0  # 固定処理時間
        await asyncio.sleep(processing_time)

        # 正規化中のデータサイズ変化をシミュレート
        processed_size = int(batch.size * 1.0)  # デモ用にサイズ変更なし

        # 強制失敗フラグを考慮
        success = not request.force_transformation_failure  # 75%成功率を常に成功に簡略化

        result = TransformationResult(
            batch_id=batch.batch_id,
            transformer_id=self.id,
            original_size=batch.size,
            processed_size=processed_size,
            transformation_type="normalization",
            processing_time=processing_time,
            success=success,
        )

        await ctx.send_message(result)


class DataEnrichment(Executor):
    """データを追加情報で強化。"""

    @handler
    async def enrich_data(self, batch: DataBatch, ctx: WorkflowContext[TransformationResult]) -> None:
        """データ強化を実行。"""
        request = await ctx.get_shared_state(f"request_{batch.batch_id}")

        # 強化が有効かチェック
        if not request or "enrich" not in request.transformations:
            # "skipped"結果を送信
            result = TransformationResult(
                batch_id=batch.batch_id,
                transformer_id=self.id,
                original_size=batch.size,
                processed_size=batch.size,
                transformation_type="enrichment",
                processing_time=0.1,
                success=True,  # Consider skipped as successful
            )
            await ctx.send_message(result)
            return

        processing_time = 5.0  # 固定処理時間
        await asyncio.sleep(processing_time)

        processed_size = int(batch.size * 1.3)  # 強化はデータを増加させる

        # 強制失敗フラグを考慮
        success = not request.force_transformation_failure  # 67%成功率を常に成功に簡略化

        result = TransformationResult(
            batch_id=batch.batch_id,
            transformer_id=self.id,
            original_size=batch.size,
            processed_size=processed_size,
            transformation_type="enrichment",
            processing_time=processing_time,
            success=success,
        )

        await ctx.send_message(result)


class DataAggregator(Executor):
    """データを集約し要約。"""

    @handler
    async def aggregate_data(self, batch: DataBatch, ctx: WorkflowContext[TransformationResult]) -> None:
        """データ集約を実行。"""
        request = await ctx.get_shared_state(f"request_{batch.batch_id}")

        # 集約が有効かチェック
        if not request or "aggregate" not in request.transformations:
            # "skipped"結果を送信
            result = TransformationResult(
                batch_id=batch.batch_id,
                transformer_id=self.id,
                original_size=batch.size,
                processed_size=batch.size,
                transformation_type="aggregation",
                processing_time=0.1,
                success=True,  # Consider skipped as successful
            )
            await ctx.send_message(result)
            return

        processing_time = 2.5  # 固定処理時間
        await asyncio.sleep(processing_time)

        processed_size = int(batch.size * 0.5)  # 集約はデータを減少させる

        # 強制失敗フラグを考慮
        success = not request.force_transformation_failure  # 80%成功率を常に成功に簡略化

        result = TransformationResult(
            batch_id=batch.batch_id,
            transformer_id=self.id,
            original_size=batch.size,
            processed_size=processed_size,
            transformation_type="aggregation",
            processing_time=processing_time,
            success=success,
        )

        await ctx.send_message(result)


# 品質保証ステージ（ファンアウト）
class PerformanceAssessor(Executor):
    """処理済みデータのパフォーマンス特性を評価。"""

    @handler
    async def assess_performance(
        self, results: list[TransformationResult], ctx: WorkflowContext[QualityAssessment]
    ) -> None:
        """変換のパフォーマンスを評価。"""
        if not results:
            return

        batch_id = results[0].batch_id

        processing_time = 2.0  # 固定処理時間
        await asyncio.sleep(processing_time)

        avg_processing_time = sum(r.processing_time for r in results) / len(results)
        success_rate = sum(1 for r in results if r.success) / len(results)

        quality_score = (success_rate * 0.7 + (1 - min(avg_processing_time / 10, 1)) * 0.3) * 100

        recommendations: list[str] = []
        if success_rate < 0.8:
            recommendations.append("Consider improving transformation reliability")
        if avg_processing_time > 5:
            recommendations.append("Optimize processing performance")
        if quality_score < 70:
            recommendations.append("Review overall data pipeline efficiency")

        assessment = QualityAssessment(
            batch_id=batch_id,
            assessor_id=self.id,
            quality_score=quality_score,
            recommendations=recommendations,
            processing_time=processing_time,
        )

        await ctx.send_message(assessment)


class AccuracyAssessor(Executor):
    """処理済みデータの正確性と妥当性を評価。"""

    @handler
    async def assess_accuracy(
        self, results: list[TransformationResult], ctx: WorkflowContext[QualityAssessment]
    ) -> None:
        """変換の正確性を評価。"""
        if not results:
            return

        batch_id = results[0].batch_id

        processing_time = 3.0  # 固定処理時間
        await asyncio.sleep(processing_time)

        # 正確性分析をシミュレート
        accuracy_score = 85.0  # 固定の正確性スコア

        recommendations: list[str] = []
        if accuracy_score < 85:
            recommendations.append("Review data transformation algorithms")
        if accuracy_score < 80:
            recommendations.append("Implement additional validation steps")

        assessment = QualityAssessment(
            batch_id=batch_id,
            assessor_id=self.id,
            quality_score=accuracy_score,
            recommendations=recommendations,
            processing_time=processing_time,
        )

        await ctx.send_message(assessment)


# 最終処理と完了
class FinalProcessor(Executor):
    """すべての結果を結合する最終処理ステージ。"""

    @handler
    async def process_final_results(
        self, assessments: list[QualityAssessment], ctx: WorkflowContext[Never, str]
    ) -> None:
        """最終処理の概要を生成しワークフローを完了。"""
        if not assessments:
            await ctx.yield_output("No quality assessments received")
            return

        batch_id = assessments[0].batch_id

        # 最終処理遅延をシミュレート
        await asyncio.sleep(2)

        # 全体のメトリクスを計算
        avg_quality_score = sum(a.quality_score for a in assessments) / len(assessments)
        total_recommendations = sum(len(a.recommendations) for a in assessments)
        total_processing_time = sum(a.processing_time for a in assessments)

        # 最終ステータスを決定
        if avg_quality_score >= 85:
            final_status = "EXCELLENT"
        elif avg_quality_score >= 75:
            final_status = "GOOD"
        elif avg_quality_score >= 65:
            final_status = "ACCEPTABLE"
        else:
            final_status = "NEEDS_IMPROVEMENT"

        completion_message = (
            f"Batch {batch_id} processing completed!\n"
            f"📊 Overall Quality Score: {avg_quality_score:.1f}%\n"
            f"⏱️  Total Processing Time: {total_processing_time:.1f}s\n"
            f"💡 Total Recommendations: {total_recommendations}\n"
            f"🎖️  Final Status: {final_status}"
        )

        await ctx.yield_output(completion_message)


# ワークフロービルダー ヘルパー
class WorkflowSetupHelper:
    """共有State管理で複雑なワークフローをセットアップするヘルパークラス。"""

    @staticmethod
    async def store_batch_data(batch: DataBatch, ctx: WorkflowContext) -> None:
        """後で取得するために共有Stateにバッチデータを保存。"""
        await ctx.set_shared_state(f"batch_{batch.batch_id}", batch)


# ワークフローインスタンスを作成
def create_complex_workflow():
    """複雑なファンイン/ファンアウトワークフローを作成。"""
    # すべてのExecutorを作成
    data_ingestion = DataIngestion(id="data_ingestion")

    # 検証ステージ（ファンアウト）
    schema_validator = SchemaValidator(id="schema_validator")
    quality_validator = DataQualityValidator(id="quality_validator")
    security_validator = SecurityValidator(id="security_validator")
    validation_aggregator = ValidationAggregator(id="validation_aggregator")

    # 変換ステージ（ファンアウト）
    data_normalizer = DataNormalizer(id="data_normalizer")
    data_enrichment = DataEnrichment(id="data_enrichment")
    data_aggregator_exec = DataAggregator(id="data_aggregator")

    # 品質保証ステージ（ファンアウト）
    performance_assessor = PerformanceAssessor(id="performance_assessor")
    accuracy_assessor = AccuracyAssessor(id="accuracy_assessor")

    # 最終処理
    final_processor = FinalProcessor(id="final_processor")

    # 複雑なファンイン/ファンアウトパターンでワークフローを構築
    return (
        WorkflowBuilder(
            name="Data Processing Pipeline",
            description="Complex workflow with parallel validation, transformation, and quality assurance stages",
        )
        .set_start_executor(data_ingestion)
        # 検証ステージへのファンアウト
        .add_fan_out_edges(data_ingestion, [schema_validator, quality_validator, security_validator])
        # 検証から集約器へのファンイン
        .add_fan_in_edges([schema_validator, quality_validator, security_validator], validation_aggregator)
        # 変換ステージへのファンアウト
        .add_fan_out_edges(validation_aggregator, [data_normalizer, data_enrichment, data_aggregator_exec])
        # 品質保証ステージへのファンイン（両方の評価者がすべての変換結果を受信）
        .add_fan_in_edges([data_normalizer, data_enrichment, data_aggregator_exec], performance_assessor)
        .add_fan_in_edges([data_normalizer, data_enrichment, data_aggregator_exec], accuracy_assessor)
        # 最終プロセッサへのファンイン
        .add_fan_in_edges([performance_assessor, accuracy_assessor], final_processor)
        .build()
    )


# DevUI検出用にワークフローをエクスポート
workflow = create_complex_workflow()


def main():
    """DevUIでファンアウトワークフローを起動。"""
    from agent_framework.devui import serve

    # ログ設定
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Complex Fan-In/Fan-Out Data Processing Workflow")
    logger.info("Available at: http://localhost:8090")
    logger.info("Entity ID: workflow_complex_workflow")

    # ワークフローでサーバーを起動
    serve(entities=[workflow], port=8090, auto_open=True)


if __name__ == "__main__":
    main()
