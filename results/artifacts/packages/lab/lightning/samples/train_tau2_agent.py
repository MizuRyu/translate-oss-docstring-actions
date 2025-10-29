# Copyright (c) Microsoft. All rights reserved.

"""Tau2ベンチマークを使ったマルチエージェントRLトレーニングの高度な例。

これにより以下を示す：
- LitAgentクラスベースのアプローチ（@rolloutデコレータとの比較）
- エージェントフィルタリングを伴うマルチエージェントシナリオ
- 複雑なセットアップのためのリソース管理
- 外部ベンチマークとの統合

train_math_agent.pyの概念に基づき、さらに複雑さを追加。
少なくとも80GBのメモリを持つGPUが1台必要。
"""

import argparse
import asyncio
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import TypedDict, cast

from agent_framework.lab.lightning import AgentFrameworkTracer
from agent_framework.lab.tau2 import ASSISTANT_AGENT_ID, patch_env_set_state  # type: ignore
from agent_framework.lab.tau2 import TaskRunner as Tau2TaskRunner  # type: ignore
from agent_framework.openai import OpenAIChatClient
from agentlightning import LLM, Dataset, LitAgent, NamedResources, Rollout, Trainer
from agentlightning.algorithm.verl import VERL
from tau2.data_model.tasks import Task as Tau2Task  # type: ignore[import-untyped]


# Tau2タスクは複雑なオブジェクトで、分散トレーニング中に特別な処理が必要。
class SerializedTask(TypedDict):
    """Tau2タスクオブジェクトのタイプ。"""

    id: str
    data: str  # HuggingFaceの変換問題を防ぐためのJSONシリアライズされたタスクデータ


def _load_dataset() -> tuple[Dataset[SerializedTask], Dataset[SerializedTask]]:
    """適切なシリアライズでTau2データセットをロードおよび準備する。

    外部データ依存（TAU2_DATA_DIR）を取り、再現性のために決定論的なtrain/val分割を使用。

    """
    data_dir = os.getenv("TAU2_DATA_DIR")
    if data_dir is None:
        raise ValueError("TAU2_DATA_DIR must be set")
    tasks_path = Path(data_dir) / "tau2/domains/airline/tasks.json"
    with tasks_path.open("r") as f:
        dataset = json.load(f)

    # HuggingFaceトークナイザーの問題を防ぐために複雑なタスクオブジェクトをシリアライズする
    dataset = [{"id": task["id"], "data": json.dumps(task)} for task in dataset]

    # 再現可能な実験のための決定論的なtrain/val分割（25/25）
    random_state = random.Random(42)  # noqa: S311
    indices = list(range(len(dataset)))
    random_state.shuffle(indices)
    train_indices = indices[: int(len(dataset) * 0.5)]
    val_indices = indices[int(len(dataset) * 0.5) :]
    print(f"Train indices: {train_indices}")
    print(f"Val indices: {val_indices}")
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    return cast(Dataset[SerializedTask], train_dataset), cast(Dataset[SerializedTask], val_dataset)


# @rolloutの代替：高度なシナリオ向けLitAgentクラス 以下が必要な場合にこのアプローチを使用： -
# エージェントフィルタリング（マルチエージェントセットアップで特定のエージェントのみをトレーニング） - リソース管理（複数のLLM、データベースなど） -
# 複雑な初期化ロジック
class Tau2Agent(LitAgent):
    """高度なリソース管理とエージェントフィルタリングを備えたクラスベースのAgent。"""

    async def rollout_async(self, task: SerializedTask, resources: NamedResources, rollout: Rollout) -> float:
        """メインのrolloutメソッド。@rolloutに似ているがより多くの制御が可能。"""
        llm = resources.get("main_llm")
        if not isinstance(llm, LLM):
            raise ValueError("main_llm must be an instance of LLM")

        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_base_url is None:
            raise ValueError("OPENAI_BASE_URL must be set")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY must be set")

        # 複雑なタスクオブジェクトをデシリアライズする
        task_data = json.loads(task["data"])
        task_obj = Tau2Task(**task_data)

        # マルチエージェントセットアップ：assistant（トレーニング可能）＋user simulator（固定）
        runner = Tau2TaskRunner(
            max_steps=100,
            assistant_window_size=4000,
            assistant_sampling_temperature=llm.sampling_parameters.get("temperature", 0.0),
        )

        # Assistantエージェント：トレーニング中のモデルを使用
        assistant_chat_client = OpenAIChatClient(
            base_url=llm.endpoint,  # vLLM endpoint for the model being trained
            api_key=openai_api_key,
            model_id=llm.model,  # Model ID being trained
        )

        # User simulator：一貫したシミュレーションのために固定された有能なモデルを使用
        user_simulator_chat_client = OpenAIChatClient(
            base_url=openai_base_url,  # External API endpoint
            api_key=openai_api_key,
            model_id="gpt-4.1",  # Fixed model for user simulator
        )

        try:
            # マルチエージェント会話を実行する
            conversation = await runner.run(task_obj, assistant_chat_client, user_simulator_chat_client)
        except Exception:
            # 失敗を適切に処理 - 問題のある行動を抑制するために低報酬を割り当てる 一般的な問題：ツール呼び出しエラー、タイムアウト、無効なレスポンス
            traceback.print_exc()
            return 0.0

        # Tau2の組み込み評価指標を使用する
        evaluation = runner.evaluate(task_obj, conversation, runner.termination_reason)

        # 評価スコアを返す
        return evaluation  # noqa: RET504


def main():
    """メインのエントリポイント。"""
    # より高いリソース要件とW&Bログを備えたRL設定
    rl_training_config = {
        "algorithm": {"adv_estimator": "grpo"},
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 8192,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 8,  # Higher repetition for more data per task
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.8,  # Higher utilization for 80GB GPU
            },
            "actor": {
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            # Referenceモデルの設定
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 8,
                "fsdp_config": {"param_offload": True},
            },
            # モデルの共通設定
            "model": {
                "path": "Qwen/Qwen2.5-1.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": True,
            "logger": ["console", "wandb"],  # Wandb for experiment tracking
            "project_name": "agent-framework-lab-lightning",
            "experiment_name": "tau2_agent",
            "nnodes": 1,
            "test_freq": 4,
            "total_epochs": 8,
        },
    }

    patch_env_set_state()  # Tau2固有の環境セットアップ

    train_dataset, val_dataset = _load_dataset()

    # math_agentとの主な違い：trained_agentsパラメータはトレーニングするエージェントを指定
    # assistantエージェントのみがトレーニングされ、user simulatorは固定のまま
    tau2_agent = Tau2Agent(trained_agents=ASSISTANT_AGENT_ID)

    tracer = AgentFrameworkTracer()
    trainer = Trainer(algorithm=VERL(rl_training_config), tracer=tracer, n_workers=4)
    trainer.fit(tau2_agent, train_dataset, val_dataset=val_dataset)


def debug():
    """マルチエージェントセットアップとTau2統合のテスト用デバッグモード。"""
    train_dataset, _ = _load_dataset()
    tau2_agent = Tau2Agent(trained_agents=ASSISTANT_AGENT_ID)

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_base_url is None:
        raise ValueError("OPENAI_BASE_URL must be set")

    patch_env_set_state()  # Tau2環境に必要

    # resources辞書でテスト（@rolloutのLLMパラメータとは異なる）
    asyncio.run(
        tau2_agent.rollout_async(
            train_dataset[0],
            resources={"main_llm": LLM(model="gpt-4.1", endpoint=openai_base_url)},
            rollout=Rollout(rollout_id="dummy", input="dummy_input", start_time=time.time()),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        main()
