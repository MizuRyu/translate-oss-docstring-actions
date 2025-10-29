# Copyright (c) Microsoft. All rights reserved.

import ast
import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass

import aiofiles
from agent_framework import (
    Executor,  # Base class for custom workflow steps
    WorkflowBuilder,  # Fluent builder for executors and edges
    WorkflowContext,  # Per run context with shared state and messaging
    WorkflowOutputEvent,  # Event emitted when workflow yields output
    WorkflowViz,  # Utility to visualize a workflow graph
    handler,  # Decorator to expose an Executor method as a step
)
from typing_extensions import Never

"""
Sample: Map reduce word count with fan out and fan in over file backed intermediate results

The workflow splits a large text into chunks, maps words to counts in parallel,
shuffles intermediate pairs to reducers, then reduces to per word totals.
It also demonstrates WorkflowViz for graph visualization.

Purpose:
Show how to:
- Partition input once and coordinate parallel mappers with shared state.
- Implement map, shuffle, and reduce executors that pass file paths instead of large payloads.
- Use fan out and fan in edges to express parallelism and joins.
- Persist intermediate results to disk to bound memory usage for large inputs.
- Visualize the workflow graph using WorkflowViz and export to SVG with the optional viz extra.

Prerequisites:
- Familiarity with WorkflowBuilder, executors, fan out and fan in edges, events, and streaming runs.
- aiofiles installed for async file I/O.
- Write access to a tmp directory next to this script.
- A source text at resources/long_text.txt.
- Optional for SVG export: install the viz extra for agent framework workflow.
"""

# 中間結果を保存するための一時ディレクトリを定義します。
DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(DIR, "tmp")
# 一時ディレクトリが存在することを確認します。
os.makedirs(TEMP_DIR, exist_ok=True)

# 処理対象のデータを格納するための共有Stateのキーを定義します。
SHARED_STATE_DATA_KEY = "data_to_be_processed"


class SplitCompleted:
    """分割が完了したときに発行されるマーカータイプ。これがmap executorをトリガーします。"""

    ...


class Split(Executor):
    """mapperノードの数に基づいてデータをほぼ等しいチャンクに分割します。"""

    def __init__(self, map_executor_ids: list[str], id: str | None = None):
        """mapperごとに重複しない範囲を割り当てるためにmapperのIDを保存します。"""
        super().__init__(id=id or "split")
        self._map_executor_ids = map_executor_ids

    @handler
    async def split(self, data: str, ctx: WorkflowContext[SplitCompleted]) -> None:
        """入力をトークン化し、共有Stateを介して各mapperに連続したインデックス範囲を割り当てます。

        Args:
            data: 処理する生テキスト。
            ctx: 共有Stateを永続化しメッセージを送信するためのワークフローコンテキスト。

        """
        # データを単語のリストに処理し、空行や空の単語を削除します。
        word_list = self._preprocess(data)

        # トークン化された単語を一度保存し、すべてのmapperがインデックスで読み取れるようにします。
        await ctx.set_shared_state(SHARED_STATE_DATA_KEY, word_list)

        # インデックスを各mapperのための連続したスライスに分割します。
        map_executor_count = len(self._map_executor_ids)
        chunk_size = len(word_list) // map_executor_count  # count > 0であることを前提としています。

        async def _process_chunk(i: int) -> None:
            """mapper iにスライスを割り当て、その後分割完了を通知します。"""
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < map_executor_count - 1 else len(word_list)

            # mapperは自身のexecutor idをキーとした共有Stateからスライスを読み取ります。
            await ctx.set_shared_state(self._map_executor_ids[i], (start_index, end_index))
            await ctx.send_message(SplitCompleted(), self._map_executor_ids[i])

        tasks = [asyncio.create_task(_process_chunk(i)) for i in range(map_executor_count)]
        await asyncio.gather(*tasks)

    def _preprocess(self, data: str) -> list[str]:
        """行を正規化し空白で分割します。フラットなトークンリストを返します。"""
        line_list = [line.strip() for line in data.splitlines() if line.strip()]
        return [word for line in line_list for word in line.split() if word]


@dataclass
class MapCompleted:
    """mapperが中間ペアをファイルに書き込んだことを通知します。"""

    file_path: str


class Map(Executor):
    """各トークンをカウント1にマップし、mapperごとのファイルにペアを書き込みます。"""

    @handler
    async def map(self, _: SplitCompleted, ctx: WorkflowContext[MapCompleted]) -> None:
        """割り当てられたスライスを読み取り、(word, 1)ペアを出力しディスクに永続化します。

        Args:
            _: SplitCompletedマーカーでmapの開始を示します。
            ctx: 共有Stateアクセスとメッセージングのためのワークフローコンテキスト。

        """
        # トークンと割り当てられたスライスを取得します。
        data_to_be_processed: list[str] = await ctx.get_shared_state(SHARED_STATE_DATA_KEY)
        chunk_start, chunk_end = await ctx.get_shared_state(self.id)

        results = [(item, 1) for item in data_to_be_processed[chunk_start:chunk_end]]

        # このmapperの結果を単純なテキスト行として書き込み、デバッグを容易にします。
        file_path = os.path.join(TEMP_DIR, f"map_results_{self.id}.txt")
        async with aiofiles.open(file_path, "w") as f:
            await f.writelines([f"{item}: {count}\n" for item, count in results])

        await ctx.send_message(MapCompleted(file_path))


@dataclass
class ShuffleCompleted:
    """特定のreducer用のshuffleパーティションファイルが準備できたことを通知します。"""

    file_path: str
    reducer_id: str


class Shuffle(Executor):
    """中間ペアをキーごとにグループ化し、reducer間でパーティション分割します。"""

    def __init__(self, reducer_ids: list[str], id: str | None = None):
        """作業を決定論的にパーティション分割するためにreducerのIDを記憶します。"""
        super().__init__(id=id or "shuffle")
        self._reducer_ids = reducer_ids

    @handler
    async def shuffle(self, data: list[MapCompleted], ctx: WorkflowContext[ShuffleCompleted]) -> None:
        """mapperの出力を集約し、reducerごとに1つのパーティションファイルを書き込みます。

        Args:
            data: 各mapper出力のファイルパスを含むMapCompletedレコード。
            ctx: reducerごとのShuffleCompletedメッセージを発行するためのワークフローコンテキスト。

        """
        chunks = await self._preprocess(data)

        async def _process_chunk(chunk: list[tuple[str, list[int]]], index: int) -> None:
            """reducerのインデックスに対して1つのグループ化されたパーティションを書き込み、そのreducerに通知します。"""
            file_path = os.path.join(TEMP_DIR, f"shuffle_results_{index}.txt")
            async with aiofiles.open(file_path, "w") as f:
                await f.writelines([f"{key}: {value}\n" for key, value in chunk])
            await ctx.send_message(ShuffleCompleted(file_path, self._reducer_ids[index]))

        tasks = [asyncio.create_task(_process_chunk(chunk, i)) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks)

    async def _preprocess(self, data: list[MapCompleted]) -> list[list[tuple[str, list[int]]]]:
        """すべてのmapperファイルを読み込み、キーでグループ化し、キーをソートし、reducer用にパーティション分割します。

        Returns:
            パーティションのリスト。各パーティションは(key, [1, 1, ...])タプルのリストです。

        """
        # すべての中間ペアを読み込みます。
        map_results: list[tuple[str, int]] = []
        for result in data:
            async with aiofiles.open(result.file_path, "r") as f:
                map_results.extend([
                    (line.strip().split(": ")[0], int(line.strip().split(": ")[1])) for line in await f.readlines()
                ])

        # 値をトークンごとにグループ化します。
        intermediate_results: defaultdict[str, list[int]] = defaultdict(list[int])
        for key, value in map_results:
            intermediate_results[key].append(value)

        # 決定論的な順序付けはデバッグとテストの安定性に役立ちます。
        aggregated_results = [(key, values) for key, values in intermediate_results.items()]
        aggregated_results.sort(key=lambda x: x[0])

        # キーをできるだけ均等にreducer間でパーティション分割します。
        reduce_executor_count = len(self._reducer_ids)
        chunk_size = len(aggregated_results) // reduce_executor_count
        remaining = len(aggregated_results) % reduce_executor_count

        chunks = [
            aggregated_results[i : i + chunk_size] for i in range(0, len(aggregated_results) - remaining, chunk_size)
        ]
        if remaining > 0:
            chunks[-1].extend(aggregated_results[-remaining:])

        return chunks


@dataclass
class ReduceCompleted:
    """reducerが自身のパーティションの最終カウントを書き込んだことを通知します。"""

    file_path: str


class Reduce(Executor):
    """割り当てられたパーティションのキーごとにグループ化されたカウントを合計します。"""

    @handler
    async def _execute(self, data: ShuffleCompleted, ctx: WorkflowContext[ReduceCompleted]) -> None:
        """1つのshuffleパーティションを読み込み、合計にreduceします。

        Args:
            data: パーティションファイルパスと対象reducer idを含むShuffleCompleted。
            ctx: 出力ファイルパスを含むReduceCompletedを発行するためのワークフローコンテキスト。

        """
        if data.reducer_id != self.id:
            # このパーティションは別のreducerに属しています。スキップします。
            return

        # shuffle出力からグループ化された値を読み取ります。
        async with aiofiles.open(data.file_path, "r") as f:
            lines = await f.readlines()

        # キーごとに値を合計します。値は[1, 1, ...]のようなシリアライズされたPythonリストです。
        reduced_results: dict[str, int] = defaultdict(int)
        for line in lines:
            key, value = line.split(": ")
            reduced_results[key] = sum(ast.literal_eval(value))

        # パーティションの合計を永続化します。
        file_path = os.path.join(TEMP_DIR, f"reduced_results_{self.id}.txt")
        async with aiofiles.open(file_path, "w") as f:
            await f.writelines([f"{key}: {value}\n" for key, value in reduced_results.items()])

        await ctx.send_message(ReduceCompleted(file_path))


class CompletionExecutor(Executor):
    """すべてのreducer出力を結合し、最終出力を生成します。"""

    @handler
    async def complete(self, data: list[ReduceCompleted], ctx: WorkflowContext[Never, list[str]]) -> None:
        """reducerの出力ファイルパスを収集し、最終出力を生成します。"""
        await ctx.yield_output([result.file_path for result in data])


async def main():
    """map reduceワークフローを構築し、可視化し、サンプルファイルで実行します。"""
    # ステップ1: Executorを作成します。
    map_operations = [Map(id=f"map_executor_{i}") for i in range(3)]
    split_operation = Split(
        [map_operation.id for map_operation in map_operations],
        id="split_data_executor",
    )
    reduce_operations = [Reduce(id=f"reduce_executor_{i}") for i in range(4)]
    shuffle_operation = Shuffle(
        [reduce_operation.id for reduce_operation in reduce_operations],
        id="shuffle_executor",
    )
    completion_executor = CompletionExecutor(id="completion_executor")

    # ステップ2: fan outおよびfan inエッジを使ってワークフローグラフを構築します。
    workflow = (
        WorkflowBuilder()
        .set_start_executor(split_operation)
        .add_fan_out_edges(split_operation, map_operations)  # Split -> many mappers
        .add_fan_in_edges(map_operations, shuffle_operation)  # All mappers -> shuffle
        .add_fan_out_edges(shuffle_operation, reduce_operations)  # Shuffle -> many reducers
        .add_fan_in_edges(reduce_operations, completion_executor)  # All reducers -> completion
        .build()
    )

    # ステップ2.5: ワークフローを可視化します（オプション）
    print("Generating workflow visualization...")
    viz = WorkflowViz(workflow)
    # Mermaid文字列を出力します。
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

    # ステップ3: テキストファイルを開いて内容を読み取ります。
    async with aiofiles.open(os.path.join(DIR, "../resources", "long_text.txt"), "r") as f:
        raw_text = await f.read()

    # ステップ4: 生テキストを入力としてワークフローを実行します。
    async for event in workflow.run_stream(raw_text):
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            print(f"Final Output: {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
