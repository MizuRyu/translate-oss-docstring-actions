# Copyright (c) Microsoft. All rights reserved.

"""GAIAベンチマークのサンプル。

このサンプルを実行するには、agent-frameworkリポジトリのルートディレクトリから実行してください:
    cd /path/to/agent-framework
    uv run python python/packages/lab/gaia/gaia_sample.py

これにより、gaiaパッケージディレクトリ内から実行したときに発生する名前空間パッケージの競合を回避できます。
"""

from agent_framework.azure import AzureAIAgentClient
from agent_framework.lab.gaia import GAIA, Evaluation, GAIATelemetryConfig, Prediction, Task
from azure.identity.aio import AzureCliCredential


def evaluate_task(task: Task, prediction: Prediction) -> Evaluation:
    """指定されたタスクの予測を評価する。"""
    # 単純な評価：予測が答えを含むかをチェックする
    is_correct = (task.answer or "").lower() in prediction.prediction.lower()
    return Evaluation(is_correct=is_correct, score=1 if is_correct else 0)


async def main() -> None:
    """テレメトリ設定でGAIAベンチマークを実行する。"""
    # トレーシングのためのテレメトリを設定する
    telemetry_config = GAIATelemetryConfig(
        enable_tracing=True,  # Enable OpenTelemetry tracing
        # ローカルファイルトレーシングを設定する
        trace_to_file=True,  # Export traces to local file
        file_path="gaia_benchmark_traces.jsonl",  # Custom file path for traces
    )

    # 単一のAgentを一度作成し、すべてのタスクで再利用する
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="GaiaAgent",
            instructions="Solve tasks to your best ability.",
        ) as agent,
    ):

        async def run_task(task: Task) -> Prediction:
            """共有Agentを使って単一のGAIAタスクを実行し、予測を返す。"""
            input_message = f"Task: {task.question}"
            if task.file_name:
                input_message += f"\nFile: {task.file_name}"
            result = await agent.run(input_message)
            return Prediction(prediction=result.text, messages=result.messages)

        # テレメトリ設定でGAIAベンチマークランナーを作成する
        runner = GAIA(evaluator=evaluate_task, telemetry_config=telemetry_config)

        # task_runnerでベンチマークを実行する。
        # デフォルトでは、ローカルにキャッシュされたベンチマークデータを確認し、見つからなければHuggingFaceから最新バージョンをチェックアウトします。
        results = await runner.run(
            run_task,
            level=1,  # Level 1, 2, or 3 or multiple levels like [1, 2]
            max_n=5,  # Maximum number of tasks to run per level
            parallel=2,  # Number of parallel tasks to run
            timeout=60,  # Timeout per task in seconds
            out="gaia_results_level1.jsonl",  # Output file to save results including detailed traces (optional)
        )

    # 結果を出力する。
    print("\n=== GAIA Benchmark Results ===")
    for result in results:
        print(f"\n--- Task ID: {result.task_id} ---")
        print(f"Task: {result.task.question[:100]}...")
        print(f"Prediction: {result.prediction.prediction}")
        print(f"Evaluation: Correct={result.evaluation.is_correct}, Score={result.evaluation.score}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
