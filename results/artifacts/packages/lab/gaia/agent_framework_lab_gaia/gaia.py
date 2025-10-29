# Copyright (c) Microsoft. All rights reserved.

"""Agent Framework用のGAIAベンチマーク実装。"""

import asyncio
import json
import os
import random
import re
import string
import tempfile
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.trace import NoOpTracer, SpanKind, get_tracer
from tqdm import tqdm

from ._types import Evaluation, Evaluator, Prediction, Task, TaskResult, TaskRunner

__all__ = ["GAIA", "GAIATelemetryConfig", "gaia_scorer"]


class GAIATelemetryConfig:
    """GAIAのテレメトリとトレーシングの設定。"""

    def __init__(
        self,
        enable_tracing: bool = False,
        otlp_endpoint: str | None = None,
        applicationinsights_connection_string: str | None = None,
        trace_to_file: bool = False,
        file_path: str | None = None,
    ):
        """テレメトリ設定を初期化します。

        Args:
            enable_tracing: OpenTelemetryトレーシングを有効にするかどうか
            otlp_endpoint: トレースエクスポート用のOTLPエンドポイント
            applicationinsights_connection_string: Azure Monitorの接続文字列
            trace_to_file: トレースをローカルファイルにエクスポートするかどうか
            file_path: ローカルファイルエクスポートのパス（デフォルトはgaia_traces.json）

        """
        self.enable_tracing = enable_tracing
        self.otlp_endpoint = otlp_endpoint
        self.applicationinsights_connection_string = applicationinsights_connection_string
        self.trace_to_file = trace_to_file
        self.file_path = file_path or "gaia_traces.json"

    def setup_observability(self) -> None:
        """設定に基づいてOpenTelemetryをセットアップします。"""
        if not self.enable_tracing:
            return

        from agent_framework.observability import setup_observability

        setup_observability(
            enable_sensitive_data=True,  # Enable for detailed task traces
            otlp_endpoint=self.otlp_endpoint,
            applicationinsights_connection_string=self.applicationinsights_connection_string,
        )

        # 要求された場合にローカルファイルエクスポートをセットアップします。
        if self.trace_to_file:
            self._setup_file_export()

    def _setup_file_export(self) -> None:
        """トレースのローカルファイルエクスポートをセットアップします。"""
        try:
            import json
            import os
            from collections.abc import Sequence

            from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
            from opentelemetry.trace import get_tracer_provider

            class FileSpanExporter(SpanExporter):
                def __init__(self, file_path: str):
                    self.file_path = file_path
                    # ディレクトリが存在することを確認します。
                    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

                def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
                    try:
                        with open(self.file_path, "a", encoding="utf-8") as f:
                            for span in spans:
                                span_data = {
                                    "trace_id": format(span.context.trace_id, "032x") if span.context else "unknown",
                                    "span_id": format(span.context.span_id, "016x") if span.context else "unknown",
                                    "name": span.name,
                                    "start_time": span.start_time,
                                    "end_time": span.end_time,
                                    "duration_ns": (span.end_time - span.start_time)
                                    if (span.end_time and span.start_time)
                                    else None,
                                    "attributes": dict(span.attributes) if span.attributes else {},
                                    "status": {
                                        "status_code": span.status.status_code.name if span.status else "UNSET",
                                        "description": span.status.description if span.status else None,
                                    },
                                }
                                f.write(json.dumps(span_data, default=str) + "\n")
                        return SpanExportResult.SUCCESS
                    except Exception:
                        return SpanExportResult.FAILURE

                def shutdown(self) -> None:
                    pass

            tracer_provider = get_tracer_provider()
            if isinstance(tracer_provider, TracerProvider):
                file_exporter = FileSpanExporter(self.file_path)
                tracer_provider.add_span_processor(BatchSpanProcessor(file_exporter))

        except ImportError:
            print("Warning: Could not set up file export for traces. Missing dependencies.")


def _normalize_number_str(number_str: str) -> float:
    """比較のために数値文字列を正規化します。"""
    for ch in ["$", "%", ","]:
        number_str = number_str.replace(ch, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _split_string(s: str, chars: list[str] | None = None) -> list[str]:
    """複数の区切り文字で文字列を分割します。"""
    if chars is None:
        chars = [",", ";"]
    return re.split(f"[{''.join(chars)}]", s)


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    """比較のために文字列を正規化します。"""
    no_spaces = re.sub(r"\s", "", s or "")
    if remove_punct:
        table = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(table)
    return no_spaces.lower()


def gaia_scorer(model_answer: str, ground_truth: str) -> bool:
    """公式GAIAスコアリング関数。

    Args:
        model_answer: モデルの回答
        ground_truth: 正解の回答

    Returns:
        回答が正しければTrue、そうでなければFalse

    """

    def is_float(x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    if model_answer is None:
        model_answer = "None"

    if is_float(ground_truth):
        # 正規化後の数値の完全一致。
        return _normalize_number_str(model_answer) == float(ground_truth)
    if any(ch in ground_truth for ch in [",", ";"]):
        # 要素ごとの比較を行うリスト（数値または文字列）。
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma, gt in zip(ma_elems, gt_elems, strict=False):
            if is_float(gt):
                comparisons.append(_normalize_number_str(ma) == float(gt))
            else:
                comparisons.append(_normalize_str(ma, remove_punct=False) == _normalize_str(gt, remove_punct=False))
        return all(comparisons)
    # 文字列の正規化＋完全一致。
    return _normalize_str(model_answer) == _normalize_str(ground_truth)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """JSONLファイルを読み込み、解析したレコードをyieldします。"""
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                import orjson

                yield orjson.loads(line)
            except Exception:
                yield json.loads(line)


def _load_gaia_local(repo_dir: Path, wanted_levels: list[int] | None = None, max_n: int | None = None) -> list[Task]:
    """ローカルリポジトリディレクトリからGAIAタスクをロードします。"""
    tasks: list[Task] = []

    for p in repo_dir.rglob("metadata.jsonl"):
        for rec in _read_jsonl(p):
            # バリアント間で使用されるフィールドを堅牢に抽出します。
            q = rec.get("Question") or rec.get("question") or rec.get("query") or rec.get("prompt")
            ans = rec.get("Final answer") or rec.get("answer") or rec.get("final_answer")
            qid = str(
                rec.get("task_id")
                or rec.get("question_id")
                or rec.get("id")
                or rec.get("uuid")
                or f"{p.stem}:{len(tasks)}"
            )
            lvl = rec.get("Level") or rec.get("level")
            fname = rec.get("file_name") or rec.get("filename") or None

            # 公開された回答のみを評価します（dev/validation分割）。
            if not q or ans is None:
                continue

            if wanted_levels and (lvl not in wanted_levels):
                continue

            tasks.append(Task(task_id=qid, question=q, answer=str(ans), level=lvl, file_name=fname, metadata=rec))

    # max_nが指定された場合、レート制限や公平性のためにシャッフルします。
    random.shuffle(tasks)
    if max_n:
        tasks = tasks[:max_n]
    return tasks


class GAIA:
    """Agent Framework用のGAIAベンチマークランナー。

    GAIA（General AI Assistant）は汎用AIアシスタントのベンチマークです。
    このクラスはカスタムAgentでベンチマークを実行するためのユーティリティを提供します。

    """

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        data_dir: str | None = None,
        hf_token: str | None = None,
        telemetry_config: GAIATelemetryConfig | None = None,
    ):
        """GAIAベンチマークランナーを初期化します。

        Args:
            evaluator: カスタム評価関数。Noneの場合はデフォルトのGAIAスコアラーを使用します。
            data_dir: GAIAデータをキャッシュするディレクトリ。デフォルトは一時ディレクトリです。
            hf_token: GAIAデータセットにアクセスするためのHugging Faceトークン。
            telemetry_config: テレメトリおよびトレーシングの設定。Noneの場合はトレーシングは行われません。

        """
        self.evaluator = evaluator or self._default_evaluator
        self.data_dir = Path(data_dir or Path(tempfile.gettempdir()) / "data_gaia_hub")
        self.hf_token = hf_token
        self.telemetry_config = telemetry_config or GAIATelemetryConfig()

        # テレメトリを設定する
        self.telemetry_config.setup_observability()

        # トレーサーを初期化する
        if self.telemetry_config.enable_tracing:
            self.tracer = get_tracer("gaia_benchmark", "1.0.0")
        else:
            self.tracer = NoOpTracer()

    async def _default_evaluator(self, task: Task, prediction: Prediction) -> Evaluation:
        """GAIA公式スコアリングを使用したデフォルトの評価関数。"""
        is_correct = gaia_scorer(prediction.prediction, task.answer or "")
        return Evaluation(is_correct=is_correct, score=1.0 if is_correct else 0.0)

    def _ensure_data(self) -> Path:
        """GAIAデータがローカルに存在することを確認する。"""
        if self.data_dir.exists() and any(self.data_dir.rglob("metadata.jsonl")):
            return self.data_dir

        # データがない場合はダウンロードする
        token = self.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable or hf_token parameter is required "
                "to access the GAIA dataset. Please set your Hugging Face token "
                "with access to gaia-benchmark/GAIA."
            )

        print(f"Downloading GAIA dataset to {self.data_dir}...")
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(  # type: ignore
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            token=token,
            local_dir=str(self.data_dir),
            force_download=False,
        )
        return Path(local_dir)

    async def _run_single_task(
        self, task: Task, task_runner: TaskRunner, semaphore: asyncio.Semaphore, timeout: int | None = None
    ) -> TaskResult:
        """エラーハンドリングとタイミングを含めて単一タスクを実行する。"""
        async with semaphore:
            with self.tracer.start_as_current_span(
                "gaia.task.run",
                kind=SpanKind.INTERNAL,
                attributes={
                    "gaia.task.id": task.task_id,
                    "gaia.task.level": task.level or 0,
                    "gaia.task.has_file": task.file_name is not None,
                    "gaia.task.timeout": timeout or 0,
                },
            ) as span:
                start_time = time.time()
                try:
                    # タスク実行のスパンを追加する
                    with self.tracer.start_as_current_span(
                        "gaia.task.execute",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            "gaia.task.question_length": len(task.question or ""),
                            "gaia.task.file_name": task.file_name or "",
                        },
                    ):
                        if timeout:
                            prediction = await asyncio.wait_for(task_runner(task), timeout=timeout)
                        else:
                            prediction = await task_runner(task)

                    # 評価のスパンを追加する
                    with self.tracer.start_as_current_span("gaia.task.evaluate", kind=SpanKind.INTERNAL):
                        evaluation = await self.evaluator(task, prediction)

                    runtime_seconds = time.time() - start_time

                    # 結果をスパンに追加する
                    if span:
                        span.set_attributes({
                            "gaia.task.runtime_seconds": runtime_seconds,
                            "gaia.task.is_correct": evaluation.is_correct,
                            "gaia.task.score": evaluation.score,
                            "gaia.task.prediction_length": len(prediction.prediction or ""),
                        })

                    return TaskResult(
                        task_id=task.task_id,
                        task=task,
                        prediction=prediction,
                        evaluation=evaluation,
                        runtime_seconds=runtime_seconds,
                    )
                except Exception as e:
                    runtime_seconds = time.time() - start_time

                    # スパンにエラーを記録する
                    if span:
                        span.set_attributes({
                            "gaia.task.runtime_seconds": runtime_seconds,
                            "gaia.task.error": str(e),
                            "gaia.task.is_correct": False,
                            "gaia.task.score": 0.0,
                        })
                        span.record_exception(e)

                    return TaskResult(
                        task_id=task.task_id,
                        task=task,
                        prediction=Prediction(prediction="", messages=[]),
                        evaluation=Evaluation(is_correct=False, score=0.0),
                        runtime_seconds=runtime_seconds,
                        error=str(e),
                    )

    async def run(
        self,
        task_runner: TaskRunner,
        level: int | list[int] = 1,
        max_n: int | None = None,
        parallel: int = 1,
        timeout: int | None = None,
        out: str | None = None,
    ) -> list[TaskResult]:
        """GAIAベンチマークを実行します。

        Args:
            task_runner: Taskを受け取りPredictionを返す関数
            level: 実行するGAIAレベル（1, 2, 3、またはレベルのリスト）
            max_n: レベルごとに実行する最大タスク数
            parallel: 並列実行するタスク数
            timeout: タスクごとのタイムアウト秒数
            out: 詳細なトレースを含む結果を保存する出力ファイル（オプション）

        Returns:
            TaskResultオブジェクトのリスト

        """
        with self.tracer.start_as_current_span(
            "gaia.benchmark.run",
            kind=SpanKind.INTERNAL,
            attributes={
                "gaia.benchmark.levels": str(level),
                "gaia.benchmark.max_n": max_n or 0,
                "gaia.benchmark.parallel": parallel,
                "gaia.benchmark.timeout": timeout or 0,
            },
        ) as benchmark_span:
            # データが利用可能であることを確認する
            with self.tracer.start_as_current_span("gaia.data.ensure", kind=SpanKind.INTERNAL):
                data_path = self._ensure_data()

            # レベルパラメータを解析する
            levels = [level] if isinstance(level, int) else level

            # タスクをロードする
            with self.tracer.start_as_current_span(
                "gaia.tasks.load",
                kind=SpanKind.INTERNAL,
                attributes={
                    "gaia.tasks.levels": str(levels),
                    "gaia.tasks.max_n": max_n or 0,
                },
            ) as load_span:
                tasks = _load_gaia_local(data_path, wanted_levels=levels, max_n=max_n)

                if load_span:
                    load_span.set_attributes({
                        "gaia.tasks.loaded_count": len(tasks),
                    })

            if not tasks:
                raise RuntimeError(
                    f"No GAIA tasks found for levels {levels}. "
                    "Make sure you have dataset access and selected valid levels."
                )

            print(f"Running {len(tasks)} GAIA tasks (levels={levels}) with {parallel} parallel workers...")

            # ベンチマークスパンをタスク情報で更新する
            if benchmark_span:
                benchmark_span.set_attributes({
                    "gaia.benchmark.total_tasks": len(tasks),
                })

            # タスクを実行する
            semaphore = asyncio.Semaphore(parallel)
            results = []

            tasks_coroutines = [self._run_single_task(task, task_runner, semaphore, timeout) for task in tasks]

            with self.tracer.start_as_current_span("gaia.tasks.execute_all", kind=SpanKind.INTERNAL):
                for coro in tqdm(
                    asyncio.as_completed(tasks_coroutines), total=len(tasks_coroutines), desc="Evaluating tasks"
                ):
                    result = await coro
                    results.append(result)

            # 集計統計を計算する
            correct = sum(1 for r in results if r.evaluation.is_correct)
            accuracy = correct / len(results) if results else 0.0
            avg_runtime = sum(r.runtime_seconds or 0 for r in results) / len(results) if results else 0.0

            # ベンチマークスパンを最終結果で更新する
            if benchmark_span:
                benchmark_span.set_attributes({
                    "gaia.benchmark.accuracy": accuracy,
                    "gaia.benchmark.correct_count": correct,
                    "gaia.benchmark.total_count": len(results),
                    "gaia.benchmark.avg_runtime_seconds": avg_runtime,
                })

            print("\nGAIA Benchmark Results:")
            print(f"Accuracy: {accuracy:.3f} ({correct}/{len(results)})")
            print(f"Average runtime: {avg_runtime:.2f}s")

            # 要求があれば結果を保存する
            if out:
                with self.tracer.start_as_current_span(
                    "gaia.results.save", kind=SpanKind.INTERNAL, attributes={"gaia.results.output_file": out}
                ):
                    self._save_results(results, out)
                    print(f"Results saved to {out}")

            return results

    def _save_results(self, results: list[TaskResult], output_path: str) -> None:
        """詳細なトレース情報を含む結果をJSONLファイルに保存する。"""
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                # メッセージをシリアライズ可能な形式に変換する
                serializable_messages = []
                if result.prediction.messages:
                    for msg in result.prediction.messages:
                        if hasattr(msg, "model_dump"):
                            # Pydanticモデル
                            serializable_messages.append(msg.model_dump())
                        elif hasattr(msg, "__dict__"):
                            # 属性を持つ通常のオブジェクト
                            serializable_messages.append(vars(msg))
                        else:
                            # 文字列表現にフォールバックする
                            serializable_messages.append(str(msg))

                record = {
                    "task_id": result.task_id,
                    "level": result.task.level,
                    "question": result.task.question,
                    "answer": result.task.answer,
                    "prediction": result.prediction.prediction,
                    "is_correct": result.evaluation.is_correct,
                    "score": result.evaluation.score,
                    "runtime_seconds": result.runtime_seconds,
                    "error": result.error,
                    "timestamp": datetime.now().isoformat(),
                    # 詳細なトレース情報を含める
                    "task_metadata": result.task.metadata,
                    "file_name": result.task.file_name,
                    "messages": serializable_messages,
                    "prediction_metadata": result.prediction.metadata,
                    "evaluation_details": result.evaluation.details,
                }
                try:
                    import orjson

                    f.write(orjson.dumps(record, default=str).decode("utf-8") + "\n")
                except ImportError:
                    f.write(json.dumps(record, default=str) + "\n")


def viewer_main() -> None:
    """gaia_viewerスクリプトのメイン関数。"""
    import argparse

    parser = argparse.ArgumentParser(description="View GAIA benchmark results")
    parser.add_argument("results_file", help="Path to results JSONL file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed view")
    parser.add_argument("--level", type=int, help="Filter by level")
    parser.add_argument("--correct-only", action="store_true", help="Show only correct answers")
    parser.add_argument("--incorrect-only", action="store_true", help="Show only incorrect answers")

    args = parser.parse_args()

    # 結果をロードする
    results = []
    with open(args.results_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    import orjson

                    results.append(orjson.loads(line))
                except ImportError:
                    results.append(json.loads(line))

    # フィルターを適用する
    if args.level is not None:
        results = [r for r in results if r.get("level") == args.level]

    if args.correct_only:
        results = [r for r in results if r.get("is_correct")]
    elif args.incorrect_only:
        results = [r for r in results if not r.get("is_correct")]

    # 結果を表示する
    if not results:
        print("No results match the filters.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct / total if total > 0 else 0.0

    print("GAIA Results Summary:")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.3f}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        status = "✓" if result.get("is_correct") else "✗"
        level = result.get("level", "?")
        task_id = result.get("task_id", "unknown")

        print(f"[{i}/{total}] {status} Level {level} - {task_id}")

        if args.detailed:
            print(f"Question: {result.get('question', 'N/A')[:100]}...")
            print(f"Answer: {result.get('answer', 'N/A')}")
            print(f"Prediction: {result.get('prediction', 'N/A')}")
            if result.get("error"):
                print(f"Error: {result.get('error')}")
            if result.get("runtime_seconds"):
                print(f"Runtime: {result.get('runtime_seconds'):.2f}s")
            print("-" * 40)


if __name__ == "__main__":
    viewer_main()
