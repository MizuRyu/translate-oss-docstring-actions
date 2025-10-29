# Copyright (c) Microsoft. All rights reserved.

import argparse
import asyncio
import json
import os
import traceback
from datetime import datetime
from typing import Any

from agent_framework.lab.tau2 import TaskRunner, patch_env_set_state
from agent_framework.openai import OpenAIChatClient
from loguru import logger
from tau2.domains.airline.environment import get_tasks


def to_dumpable(result: dict[str, Any]) -> dict[str, Any]:
    """ベンチマーク結果をJSONLシリアライズ可能な形式に変換します。

    成功した実行とエラーケースの両方を処理し、下流の分析のために一貫した出力形式を保証します。
    Pydanticモデルを辞書に変換し、JSON互換性のためにenum値を抽出します。

    """
    if "error" in result:
        # エラーケース: 報酬がゼロの最小限の構造
        return {
            "id": result["task"].id,
            "error": result["error"],
            "evaluation": {
                "reward": 0.0,  # Standard zero reward for failed runs
            },
            "config": result["config"],
            "task": result["task"].model_dump(),
        }
    # 成功ケース: 完全な結果構造
    return {
        "id": result["task"].id,
        "evaluation": result["evaluation"].model_dump(),  # Detailed evaluation metrics
        "config": result["config"],  # Model configuration used
        "termination_reason": result["termination_reason"].value,  # Enum to string
        "messages": [m.model_dump() for m in result["messages"]],  # Full conversation
        "task": result["task"].model_dump(),  # Task specification
    }


async def run_benchmark(assistant_model: str, user_model: str, debug_task_id: str | None, max_steps: int):
    """agentフレームワークを使用して包括的なtau2ベンチマーク評価を実行します。

    これは主な関数であり、以下を行います:

    1. 出力ファイル処理の設定（完全なベンチマークモードとデバッグモード）
    2. tau2タスクデータセットの読み込みとLLMクライアントの設定
    3. 各タスクをagentフレームワークのワークフローで実行
    4. tau2の多次元メトリクスを使用したパフォーマンス評価
    5. 結果の集計と全体的なベンチマークスコアの計算

    Args:
        assistant_model: カスタマーサービスAgentのモデルID（例: "gpt-4o"）
        user_model: ユーザーシミュレーターのモデルID（例: "gpt-4o"）
        debug_task_id: 実行する特定のタスクID（オプション、バッチ処理を無効化）
        max_steps: 強制終了前の最大会話ステップ数

    Output:
        分析用の詳細な結果を含むタイムスタンプ付きJSONLファイルを作成
        カラーロギング付きの要約統計をコンソールに出力

    """
    # STEP 1: 実行モードに基づく出力処理の設定
    result_filename = None
    if debug_task_id is None:
        # 完全なベンチマークモード: タイムスタンプ付き結果ファイルを作成
        timestamp = datetime.now().strftime("%m%d%H%M")  # フォーマット: MMDDHHMM
        result_filename = f"results/{assistant_model}_user-{user_model}_{timestamp}.jsonl"
        os.makedirs("results", exist_ok=True)
        logger.info(f"Results will be saved to: {result_filename}")
    else:
        # デバッグモード: 単一タスク、ファイル出力なし、詳細なロギング
        logger.info(f"Debug mode: targeting task ID {debug_task_id}")

    # STEP 2: tau2データセットの読み込みと環境の検証
    tasks = get_tasks()  # すべてのtau2航空会社カスタマーサービスタスクを読み込みます
    logger.info(f"Found {len(tasks)} tasks in the dataset")

    logger_ = logger.opt(colors=True)  # カラフルなコンソール出力を有効化

    # 必要なOpenAI設定を検証 両モデルは同じエンドポイントを使用しますが、異なるモデルタイプでも可能です
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_base_url is None:
        raise ValueError("OPENAI_BASE_URL must be set")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY must be set")

    # STEP 3: 両Agent役割のLLMクライアントを初期化 アシスタント: ツールとポリシーにアクセスしカスタマーサービスを担当
    assistant_chat_client = OpenAIChatClient(
        base_url=openai_base_url,
        api_key=openai_api_key,
        model_id=assistant_model,
    )

    # ユーザーシミュレーター: 現実的な顧客行動とリクエストをシミュレート
    user_simulator_chat_client = OpenAIChatClient(
        base_url=openai_base_url,
        api_key=openai_api_key,
        model_id=user_model,
    )

    # STEP 4: デバッグモード用にタスクセットをフィルタリング
    if debug_task_id is not None:
        tasks = [task for task in tasks if task.id == debug_task_id]
        if not tasks:
            logger.error(f"Task ID {debug_task_id} not found in dataset")
            return

    # STEP 5: 評価トラッキングを初期化
    all_rewards: list[float] = []  # 最終統計用に報酬スコアを保存
    task_runner = TaskRunner(max_steps=max_steps)  # 再利用可能なワークフローオーケストレーター

    # STEP 6: 適切なファイル処理で全タスクにわたるベンチマークを実行
    def write_result(result_fp, result):
        """ファイルポインタが提供されていれば結果をファイルに書き込みます。"""
        if result_fp is not None:
            result_fp.write(json.dumps(to_dumpable(result), default=str) + "\n")

    # ファイル処理にコンテキストマネージャを使用
    if result_filename:
        with open(result_filename, "a") as result_fp:
            for task in tasks:
                logger_.info(f"<red>Testing task #{task.id}</red>")
                logger_.info(f"<cyan>Purpose:</cyan> {task.description.purpose}")  # type: ignore

                # このタスク用の結果構造を初期化
                result: dict[str, Any] = {
                    "config": {
                        "assistant": assistant_chat_client.model_id,
                        "user": user_simulator_chat_client.model_id,
                    },
                    "task": task,
                }

                # 透明性のためユーザーシナリオのコンテキストをログに記録
                if task.user_scenario and task.user_scenario.instructions:
                    logger_.info(f"<cyan>User scenario:</cyan> {task.user_scenario.instructions.reason_for_call}")  # type: ignore

                try:
                    # ワークフローを実行: agent + ユーザーシミュレーターの会話
                    conversation = await task_runner.run(task, assistant_chat_client, user_simulator_chat_client)

                    # tau2の包括的なメトリクスを使用してパフォーマンスを評価
                    reward_value = task_runner.evaluate(task, conversation, task_runner.termination_reason)

                    # 分析用に詳細な結果を保存
                    result["evaluation"] = task_runner.full_reward_info  # 完全な評価内訳
                    result["messages"] = conversation  # 完全な会話履歴
                    result["termination_reason"] = task_runner.termination_reason  # 会話の終了方法

                    # 評価結果をログに記録（カラフル出力のためHTMLをエスケープ）
                    reward_str = str(task_runner.full_reward_info).replace("<", r"\<")
                    logger_.info(f"<cyan>Final evaluation:</cyan> {reward_str}")

                except Exception as e:
                    # 堅牢なエラー処理: すべての失敗を分析用にキャプチャ
                    logger_.error(f"<red>Error testing task #{task.id}:</red> {e}")
                    result["error"] = traceback.format_exc()  # デバッグ用の完全なスタックトレース

                    traceback.print_exc()  # 即時デバッグ用のコンソール出力
                    reward_value = 0.0  # 失敗した実行はスコアゼロ

                # STEP 7: 結果を逐次保存（部分的な分析を可能に）
                write_result(result_fp, result)

                all_rewards.append(reward_value)  # 最終統計用に追跡

                # 次のタスクのためにrunnerの状態をリセット
                task_runner.reinit()
    else:
        # ファイル出力なしのデバッグモード
        for task in tasks:
            logger_.info(f"<red>Testing task #{task.id}</red>")
            logger_.info(f"<cyan>Purpose:</cyan> {task.description.purpose}")  # type: ignore

            # このタスク用の結果構造を初期化
            result: dict[str, Any] = {
                "config": {
                    "assistant": assistant_chat_client.model_id,
                    "user": user_simulator_chat_client.model_id,
                },
                "task": task,
            }

            # 透明性のためユーザーシナリオのコンテキストをログに記録
            if task.user_scenario and task.user_scenario.instructions:
                logger_.info(f"<cyan>User scenario:</cyan> {task.user_scenario.instructions.reason_for_call}")  # type: ignore

            try:
                # ワークフローを実行: agent + ユーザーシミュレーターの会話
                conversation = await task_runner.run(task, assistant_chat_client, user_simulator_chat_client)

                # tau2の包括的なメトリクスを使用してパフォーマンスを評価
                reward_value = task_runner.evaluate(task, conversation, task_runner.termination_reason)

                # 評価結果をログに記録（カラフル出力のためHTMLをエスケープ）
                reward_str = str(task_runner.full_reward_info).replace("<", r"\<")
                logger_.info(f"<cyan>Final evaluation:</cyan> {reward_str}")

            except Exception as e:
                # 堅牢なエラー処理: すべての失敗を分析用にキャプチャ
                logger_.error(f"<red>Error testing task #{task.id}:</red> {e}")
                traceback.print_exc()  # 即時デバッグ用のコンソール出力
                reward_value = 0.0  # 失敗した実行はスコアゼロ

            all_rewards.append(reward_value)  # 最終統計用に追跡

            # 次のタスクのためにrunnerの状態をリセット
            task_runner.reinit()

    # STEP 8: 全体のベンチマークパフォーマンスを計算し最終統計を報告
    all_accuracy = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    # カラーフォーマット付きで最終統計を報告
    logger_.info("<green>Final Results:</green>")
    logger_.info(f"<cyan>All tasks accuracy:</cyan> {all_accuracy:.2f} ({int(sum(all_rewards))}/{len(tasks)})")


if __name__ == "__main__":
    """Command-line interface for tau2 benchmark execution.

    Provides flexible execution modes:

    - Full benchmark: Runs all tasks and generates timestamped results file
    - Debug mode: Single task execution with verbose logging for development
    - Environment patching: Optional compatibility layer for tau2-bench integration

    Usage Examples:
        # Full benchmark with default models
        python run_benchmark.py

        # Custom models
        python run_benchmark.py --assistant gpt-4o --user gpt-4o-mini

        # Debug specific task
        python run_benchmark.py --debug-task-id task_123

        # Disable environment patching for testing
        python run_benchmark.py --disable-env-patch
    """

    parser = argparse.ArgumentParser(description="Run tau2-agent-framework model test")

    # モデル設定引数
    parser.add_argument("--assistant", type=str, default="gpt-4.1", help="Assistant model id, e.g., gpt-4.1-mini")
    parser.add_argument("--user", type=str, default="gpt-4.1", help="User model id")

    # 実行モード引数
    parser.add_argument(
        "--debug-task-id", type=str, default=None, help="Debug a specific task ID (disables result file creation)"
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps to run")

    # 環境設定引数
    parser.add_argument("--disable-env-patch", action="store_true", help="Disable patching tau2-bench environment")

    args = parser.parse_args()

    # tau2-bench互換性のため環境パッチを適用 これはtau2の環境をツール呼び出し検証に対してより柔軟にします
    if not args.disable_env_patch:
        patch_env_set_state()

    # 設定されたパラメータでベンチマークを実行
    asyncio.run(
        run_benchmark(
            assistant_model=args.assistant,
            user_model=args.user,
            debug_task_id=args.debug_task_id,
            max_steps=args.max_steps,
        )
    )
