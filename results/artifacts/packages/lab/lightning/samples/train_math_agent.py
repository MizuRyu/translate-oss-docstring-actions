# Copyright (c) Microsoft. All rights reserved.

"""このサンプルはagent-framework-lab-lightningの基本的な使用パターンを示します。

`data/math/`のデータセットを使って数学エージェントをトレーニングし、
MCP計算機ツールを用いて数学問題を解きます。

40GBメモリのGPU1台で十分です。
"""

import argparse
import asyncio
import json
import math
import os
import re
import string
from typing import TypedDict, cast

import sympy  # type: ignore[import-untyped,reportMissingImports]
from agent_framework import AgentRunResponse, ChatAgent, MCPStdioTool
from agent_framework.lab.lightning import AgentFrameworkTracer
from agent_framework.openai import OpenAIChatClient
from agentlightning import LLM, Dataset, Trainer, rollout
from agentlightning.algorithm.verl import VERL


class MathProblem(TypedDict):
    """このTypedDictは各トレーニングサンプルの構造を定義します。

    タスク構造には以下の情報が含まれている必要があります:

    - エージェントがタスクを処理するための情報（例：'question'）
    - 評価用の情報（例：正解の'result'）

    この型はオプションであり、例を動作させるために必須ではありません。

    """

    # フィールドはデータセットから取得されます
    id: str
    question: str  # エージェントが解く数学問題
    chain: str  # ステップバイステップの解答（トレーニングでは使用しません）
    result: str  # 評価用の正解
    source: str


def _load_jsonl(file_path: str) -> Dataset[MathProblem]:
    """データセットをタスクサンプルのリストとしてロードします。

    各サンプルはタスク構造（この場合はMathProblem）に一致している必要があります。

    """
    with open(file_path) as f:
        raw_data = [MathProblem(**json.loads(line)) for line in f]
    return cast(Dataset[MathProblem], raw_data)


# 評価ロジック これらの関数はエージェントの答えが正解と一致するかを評価します。 堅牢な評価はRLトレーニングに不可欠で、報酬信号が学習を導きます。


def _normalize_option(option: str) -> str:
    return re.sub(r"(\s+|\(|\))", "", option)


def _is_option_result(result: str) -> bool:
    return _normalize_option(result) in list(string.ascii_letters)


def _float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def _scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    pred_result = str(pred_result) if pred_result is not None else ""
    true_result = str(true_result) if true_result is not None else ""

    if pred_result.strip() == true_result.strip():
        return True

    if _is_option_result(true_result):
        # タスクは正しい選択肢を選ぶことです
        true_result = _normalize_option(true_result)
        pred_result = _normalize_option(pred_result)
        return pred_result == true_result

    # タスクは結果を数値として計算することです
    try:
        pred_float = _float_eval(pred_result)
        true_float = _float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:  # noqa: S110
        pass

    return False


def _is_result_correct(prediction: str, ground_truth: str) -> float:
    return float(_scalar_are_results_same(prediction, ground_truth, 1e-2))


def evaluate(result: AgentRunResponse, ground_truth: str) -> float:
    """エージェントの答えを抽出し正解と比較するメイン評価関数。

    この関数は:
    1. エージェントの応答から最終答えを抽出する（###以降）
    2. 数学的同値性を用いて正解と比較する
    3. RLトレーニング用に報酬スコア（通常0.0か1.0）を返す

    報酬信号は重要で、モデルの学習内容に直接影響します。

    """
    # エージェントが応答を提供したかをチェックする
    if len(result.messages) == 0:
        print("No response from agent. Assuming incorrect.")
        return 0.0
    final_message = result.messages[-1].text

    # ###マーカー以降の答えを抽出する（エージェントの指示に従う）
    answer = re.search(r"###\s*(.+?)(\s*###|$)", final_message)
    if answer is None:
        print("No answer can be extracted from agent's response. Assuming incorrect.")
        return 0.0
    answer = answer.group(1)

    # 抽出した答えを正解と比較する
    reward = _is_result_correct(answer, ground_truth)
    print(f"Reward: {reward}")
    return reward


# エージェントロジック 一貫したエージェント動作のために明確な指示が重要です ###フォーマットは評価時の答え抽出を確実にします
AGENT_INSTRUCTION = """
Solve the following math problem. Use the calculator tool to help you calculate math expressions.

Output the answer when you are ready. The answer should be after three sharps (`###`), with no extra punctuations or texts. For example: ### 123
""".strip()  # noqa: E501


# @rolloutデコレータはagent-lightningとの統合の重要なポイントです。
# この関数がトレーニング可能なエージェントを定義していることをトレーニングシステムに伝えます。
@rollout
async def math_agent(task: MathProblem, llm: LLM) -> float:
    """これはあなたのトレーニング可能なエージェント関数です。

    重要なポイント:

    1. @rolloutデコレータで装飾する必要があります
    2. タスクサンプルとLLMオブジェクトをパラメータに取ります
    3. 浮動小数点の報酬スコア（通常0.0から1.0）を返します
    4. LLMオブジェクトはトレーニング中のモデルとその設定を含みます

    トレーニング中:
    - llm.model: トレーニング中のモデルチェックポイント
    - llm.endpoint: 推論用のvLLMサーバーエンドポイント
    - llm.sampling_parameters: Temperatureなど

    """
    # Agent Frameworkコンポーネントを作成する MCPStdioToolはMCPプロトコルを通じて計算機能を提供します
    async with (
        MCPStdioTool(name="calculator", command="uvx", args=["mcp-server-calculator"]) as mcp_server,
        ChatAgent(
            chat_client=OpenAIChatClient(
                model_id=llm.model,  # This is the model being trained
                api_key=os.getenv("OPENAI_API_KEY") or "dummy",  # Can be dummy when connecting to training LLM
                base_url=llm.endpoint,  # vLLM server endpoint provided by agent-lightning
            ),
            name="MathAgent",
            instructions=AGENT_INSTRUCTION,
            temperature=llm.sampling_parameters.get("temperature", 0.0),
        ) as agent,
    ):
        print(f"Task: {task['question'][:10]}...")
        # タスクでエージェントを実行する
        result = await agent.run(task["question"], tools=mcp_server)
        print(f"Agent responses: {result}")

        # 評価して報酬を返す - これがRLトレーニングを駆動します
        return evaluate(result, task["result"])


def main():
    """メインエントリポイント。"""
    # RLトレーニングを設定する この設定はRLトレーニングプロセスのすべての側面を制御します。 主なセクション：algorithm, data, rollout,
    # actor, trainer
    rl_training_config = {
        "algorithm": {
            # アドバンテージ推定器のタイプ："gae", "grpo", "reinforce_plus_plus"など
            "adv_estimator": "grpo"
        },
        "data": {
            # ロールアウトに使用するデータセットからのタスク数
            "train_batch_size": 8,
            # 過度に長いプロンプト-レスポンスペアをフィルタリングするために使用
            "max_prompt_length": 4096,
            "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            # ロールアウトプロセスを制御する
            "rollout": {
                # 複数GPUでTPを使わない限り1に設定する
                "tensor_model_parallel_size": 1,
                # 各タスクをN回繰り返す。G(rouped)RPOで必要
                "n": 4,
                # ログ確率計算時のGPUごとのバッチサイズを制御する
                "log_prob_micro_batch_size_per_gpu": 2,
                # マルチターンフォーマットを制御する（使用するLLMにバインドされています） 詳細は
                # https://docs.vllm.ai/en/stable/features/tool_calling.html を参照してください
                "multi_turn": {"format": "hermes"},
                # 現在はvllmのみサポートされています
                "name": "vllm",
                # vLLMのGPUメモリ使用率を制御する OOMを防ぐために0.8未満に設定することを推奨します
                "gpu_memory_utilization": 0.7,
            },
            "actor": {
                # PPO用に各サンプルをこのサイズのサブバッチに分割する
                "ppo_mini_batch_size": 8,
                # GPUごとのローカルマイクロバッチサイズ
                "ppo_micro_batch_size_per_gpu": 2,
                # Optimizerの設定
                "optim": {"lr": 1e-6},
                # トレーニング中にKL lossを使用するかどうか
                "use_kl_loss": False,
                # ポリシー更新のためのPPOクリッピング比率
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                # メモリ効率のためのFSDP（Fully Sharded Data Parallel）設定 GPUメモリが不足している場合に有用
                "fsdp_config": {
                    # パラメータをCPUにオフロードするかどうか
                    "param_offload": True,
                    # Optimizerの状態をCPUにオフロードするかどうか
                    "optimizer_offload": True,
                },
            },
            # Referenceモデルの設定
            "ref": {
                # Referenceモデルのlog-prob計算時のGPUあたりのバッチサイズを制御
                "log_prob_micro_batch_size_per_gpu": 2,
                "fsdp_config": {"param_offload": True},
            },
            # モデルの共通設定
            "model": {
                # Huggingfaceモデルのパス。 別のモデルをトレーニングしたい場合はここでパスを変更してください。
                "path": "Qwen/Qwen2.5-1.5B-Instruct",
                # トレーニング中に入力のパディングトークンを削除するかどうか
                "use_remove_padding": True,
                # メモリ効率のために勾配チェックポイントを有効にする
                "enable_gradient_checkpointing": True,
            },
        },
        # トレーナーの設定
        "trainer": {
            # ノードあたりのGPU数
            "n_gpus_per_node": 1,
            # トレーニング開始前にバリデーションを実行するかどうか
            "val_before_train": True,
            # 使用するログバックエンド："console"、"wandb"など
            "logger": ["console"],
            # トレーニングに使用するノード数
            "nnodes": 1,
            # バリデーションの頻度（トレーニングイテレーション単位）
            "test_freq": 4,
            # トレーニングのエポック数
            "total_epochs": 2,
        },
    }

    # データセットをロードする
    train_dataset = _load_jsonl("data/math/train.jsonl")
    val_dataset = _load_jsonl("data/math/test.jsonl")

    # データが正しくロードされているかプレビューする
    print("First 5 rows of train dataset:")
    for i in range(5):
        print(train_dataset[i])
    print("First 5 rows of val dataset:")
    for i in range(5):
        print(val_dataset[i])

    # VERLアルゴリズムでトレーナーを作成しトレーニングを開始する n_workers: 並列データ収集のためのロールアウトワーカー（プロセス）数
    trainer = Trainer(algorithm=VERL(rl_training_config), tracer=AgentFrameworkTracer(), n_workers=2)

    # 実際のRLトレーニングループを開始する： 1. 現在のモデルを使ってロールアウトを収集 2. アドバンテージを計算しモデルをトレーニング 3.
    # 指定されたエポック数だけ繰り返す
    trainer.fit(math_agent, train_dataset, val_dataset=val_dataset)


def debug():
    """デバッグモードではトレーニング前にAgent関数をテストできます。

    高価なRLトレーニングを始める前に必ずデバッグモードを最初に実行してください！

    """
    train_dataset = _load_jsonl("data/math/train.jsonl")
    train_sample = train_dataset[0]

    # デバッグ用に既知の良好なモデルを使用（トレーニング中のモデルではない）
    model = "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY must be set")
    if base_url is None:
        raise ValueError("OPENAI_BASE_URL must be set")

    # サンプルタスクでAgent関数をテストする
    asyncio.run(math_agent(train_sample, LLM(model=model, endpoint=base_url)))  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        main()
