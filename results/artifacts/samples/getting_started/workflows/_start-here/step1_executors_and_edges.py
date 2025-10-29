# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from typing_extensions import Never

"""
Step 1: Foundational patterns: Executors and edges

What this example shows
- Two ways to define a unit of work (an Executor node):
    1) Custom class that subclasses Executor with an async method marked by @handler.
         Possible handler signatures:
            - (text: str, ctx: WorkflowContext) -> None,
            - (text: str, ctx: WorkflowContext[str]) -> None, or
            - (text: str, ctx: WorkflowContext[Never, str]) -> None.
         The first parameter is the typed input to this node, the input type is str here.
         The second parameter is a WorkflowContext[T_Out, T_W_Out].
         WorkflowContext[T_Out] is used for nodes that send messages to downstream nodes with ctx.send_message(T_Out).
         WorkflowContext[T_Out, T_W_Out] is used for nodes that also yield workflow
            output with ctx.yield_output(T_W_Out).
         WorkflowContext without type parameters is equivalent to WorkflowContext[Never, Never], meaning this node
            neither sends messages to downstream nodes nor yields workflow output.

    2) Standalone async function decorated with @executor using the same signature.
         Simple steps can use this form; a terminal step can yield output
         using ctx.yield_output() to provide workflow results.

- Fluent WorkflowBuilder API:
    add_edge(A, B) to connect nodes, set_start_executor(A), then build() -> Workflow.

- Running and results:
    workflow.run(initial_input) executes the graph. Terminal nodes yield
    outputs using ctx.yield_output(). The workflow runs until idle.

Prerequisites
- No external services required.
"""


# Example 1: カスタムExecutorサブクラス ------------------------------------
# Executorをサブクラス化すると、必要に応じてライフサイクルフックを持つ名前付きノードを定義できる。
# 実際の処理は@handlerで装飾された非同期メソッドで実装される。  Handlerのシグネチャ契約: - 最初のパラメータはこのノードへの型付き入力（ここではtext:
# str） -
# 2番目のパラメータはWorkflowContext[T_Out]で、T_Outはこのノードがctx.send_messageで送信するデータの型（ここではstr）
# Handler内では通常: - 結果を計算する - ctx.send_message(result)を使って結果を下流ノードに転送する
class UpperCase(Executor):
    def __init__(self, id: str):
        super().__init__(id=id)

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """入力を大文字に変換し、次のノードに転送する。

        注意: WorkflowContextはこのハンドラが送信する型でパラメータ化されている。
        ここでWorkflowContext[str]は下流ノードがstrを期待することを意味する。

        """
        result = text.upper()

        # 結果をワークフロー内の次のExecutorに送信する。
        await ctx.send_message(result)


# Example 2: スタンドアロンの関数ベースのexecutor -----------------------------------------------
# 単純なステップの場合はサブクラス化を省略し、同じシグネチャパターン（型付き入力 + WorkflowContext[T_Out,
# T_W_Out]）の非同期関数を定義し、@executorでデコレートできます。これにより、フローに接続可能な完全な機能を持つノードが作成されます。


@executor(id="reverse_text_executor")
async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """入力文字列を逆順にしてワークフロー出力をyieldする。

    このノードはctx.yield_output(result)を使って最終出力をyieldする。
    ワークフローはアイドル状態（処理がなくなる）になると完了する。

    WorkflowContextは2つの型でパラメータ化されている:
    - T_Out = Never: このノードは下流ノードにメッセージを送信しない。
    - T_W_Out = str: このノードはstr型のワークフロー出力をyieldする。

    """
    result = text[::-1]

    # 出力をyieldします - ワークフローはアイドル状態になると完了します
    await ctx.yield_output(result)


async def main():
    """フルーエントビルダーAPIを使ってシンプルな2ステップのワークフローを構築し実行します。"""

    upper_case = UpperCase(id="upper_case_executor")

    # フルーエントパターンでワークフローを構築します: 1) add_edge(from_node, to_node) は有向エッジ upper_case ->
    # reverse_text を定義します 2) set_start_executor(node) はエントリーポイントを宣言します 3) build()
    # は最終化して不変のWorkflowオブジェクトを返します
    workflow = WorkflowBuilder().add_edge(upper_case, reverse_text).set_start_executor(upper_case).build()

    # 開始ノードに初期メッセージを送信してワークフローを実行します。 run(...) 呼び出しはイベントコレクションを返し、その get_outputs() メソッドで
    # 終端ノードからyieldされた出力を取得します。
    events = await workflow.run("hello world")
    print(events.get_outputs())
    # 最終的なrun状態を要約します（例: COMPLETED）
    print("Final state:", events.get_final_state())

    """
    Sample Output:

    ['DLROW OLLEH']
    Final state: WorkflowRunState.COMPLETED
    """


if __name__ == "__main__":
    asyncio.run(main())
