# Copyright (c) Microsoft. All rights reserved.

"""ワークフロー可視化モジュールのテスト。"""

import pytest

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, WorkflowExecutor, WorkflowViz, handler


class MockExecutor(Executor):
    """テスト目的のモックexecutor。"""

    @handler
    async def mock_handler(self, message: str, ctx: WorkflowContext) -> None:
        """何もしないモックハンドラ。"""
        pass


class ListStrTargetExecutor(Executor):
    """ファンインターゲットのための文字列リストを受け入れるモックExecutor。"""

    @handler
    async def handle(self, message: list[str], ctx: WorkflowContext) -> None:
        pass


@pytest.fixture
def basic_sub_workflow():
    """テスト用の基本的なサブワークフローセットアップを作成するFixture。"""
    # サブワークフローを作成する
    sub_exec1 = MockExecutor(id="sub_exec1")
    sub_exec2 = MockExecutor(id="sub_exec2")

    sub_workflow = WorkflowBuilder().add_edge(sub_exec1, sub_exec2).set_start_executor(sub_exec1).build()

    # サブワークフローをラップするワークフローExecutorを作成する
    workflow_executor = WorkflowExecutor(sub_workflow, id="workflow_executor_1")

    # ワークフローExecutorを含むメインワークフローを作成する
    main_exec = MockExecutor(id="main_executor")
    final_exec = MockExecutor(id="final_executor")

    main_workflow = (
        WorkflowBuilder()
        .add_edge(main_exec, workflow_executor)
        .add_edge(workflow_executor, final_exec)
        .set_start_executor(main_exec)
        .build()
    )

    return {
        "main_workflow": main_workflow,
        "workflow_executor": workflow_executor,
        "sub_workflow": sub_workflow,
        "main_exec": main_exec,
        "final_exec": final_exec,
        "sub_exec1": sub_exec1,
        "sub_exec2": sub_exec2,
    }


def test_workflow_viz_to_digraph():
    """WorkflowVizがDOT digraphを生成できることをテストする。"""
    # シンプルなワークフローを作成する
    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()

    viz = WorkflowViz(workflow)
    dot_content = viz.to_digraph()

    # DOTコンテンツに期待される要素が含まれていることを確認する
    assert "digraph Workflow {" in dot_content
    assert '"executor1"' in dot_content
    assert '"executor2"' in dot_content
    assert '"executor1" -> "executor2"' in dot_content
    assert "fillcolor=lightgreen" in dot_content  # Executorのスタイリングを開始する
    assert "(Start)" in dot_content


def test_workflow_viz_export_dot():
    """ワークフローをDOT形式でエクスポートするテスト。"""
    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()

    viz = WorkflowViz(workflow)

    # ファイル名なしでのエクスポートをテスト（一時ファイルパスを返す）
    file_path = viz.export(format="dot")
    assert file_path.endswith(".dot")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    assert "digraph Workflow {" in content
    assert '"executor1" -> "executor2"' in content


def test_workflow_viz_export_dot_with_filename(tmp_path):
    """指定されたファイル名でワークフローをDOT形式でエクスポートするテスト。"""
    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()

    viz = WorkflowViz(workflow)

    # ファイル名付きのエクスポートをテストする
    output_file = tmp_path / "test_workflow.dot"
    result_path = viz.export(format="dot", filename=str(output_file))

    assert result_path == str(output_file)
    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8")
    assert "digraph Workflow {" in content
    assert '"executor1" -> "executor2"' in content


def test_workflow_viz_complex_workflow():
    """より複雑なワークフローの可視化をテストする。"""
    executor1 = MockExecutor(id="start")
    executor2 = MockExecutor(id="middle1")
    executor3 = MockExecutor(id="middle2")
    executor4 = MockExecutor(id="end")

    workflow = (
        WorkflowBuilder()
        .add_edge(executor1, executor2)
        .add_edge(executor1, executor3)
        .add_edge(executor2, executor4)
        .add_edge(executor3, executor4)
        .set_start_executor(executor1)
        .build()
    )

    viz = WorkflowViz(workflow)
    dot_content = viz.to_digraph()

    # すべてのExecutorが存在することを確認する
    assert '"start"' in dot_content
    assert '"middle1"' in dot_content
    assert '"middle2"' in dot_content
    assert '"end"' in dot_content

    # すべてのエッジが存在することを確認する
    assert '"start" -> "middle1"' in dot_content
    assert '"start" -> "middle2"' in dot_content
    assert '"middle1" -> "end"' in dot_content
    assert '"middle2" -> "end"' in dot_content

    # 開始Executorに特別なスタイリングがあることを確認する
    assert "fillcolor=lightgreen" in dot_content


@pytest.mark.skipif(True, reason="Requires graphviz to be installed")
def test_workflow_viz_export_svg():
    """ワークフローをSVG形式でエクスポートするテスト。graphvizが利用可能でない場合はスキップされる。"""
    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()

    viz = WorkflowViz(workflow)

    try:
        file_path = viz.export(format="svg")
        assert file_path.endswith(".svg")
    except ImportError:
        pytest.skip("graphviz not available")


def test_workflow_viz_unsupported_format():
    """サポートされていないフォーマットでValueErrorが発生することをテストする。"""
    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()

    viz = WorkflowViz(workflow)

    with pytest.raises(ValueError, match="Unsupported format: invalid"):
        viz.export(format="invalid")  # type: ignore


def test_workflow_viz_graphviz_binary_not_found():
    """graphvizバイナリが見つからない場合に役立つメッセージ付きでImportErrorが発生することをテストする。"""
    import unittest.mock

    # graphvizパッケージが利用できない場合はテストをスキップする
    pytest.importorskip("graphviz")

    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()
    viz = WorkflowViz(workflow)

    # graphviz.Source.renderをモックしてExecutableNotFoundを発生させる
    with unittest.mock.patch("graphviz.Source") as mock_source_class:
        mock_source = unittest.mock.MagicMock()
        mock_source_class.return_value = mock_source

        # テスト用にExecutableNotFound例外をインポートする
        from graphviz.backend.execute import ExecutableNotFound

        mock_source.render.side_effect = ExecutableNotFound("failed to execute PosixPath('dot')")

        # 適切なImportErrorが役立つメッセージ付きで発生することをテストする
        with pytest.raises(ImportError, match="The graphviz executables are not found"):
            viz.export(format="svg")


def test_workflow_viz_conditional_edge():
    """条件付きエッジが破線でラベル付きでレンダリングされることをテストする。"""
    start = MockExecutor(id="start")
    mid = MockExecutor(id="mid")
    end = MockExecutor(id="end")

    # 可視化中に使用されない条件だが、存在はエッジをマークすべきである
    def only_if_foo(msg: str) -> bool:  # pragma: no cover - simple predicate
        return msg == "foo"

    wf = (
        WorkflowBuilder()
        .add_edge(start, mid, condition=only_if_foo)
        .add_edge(mid, end)
        .set_start_executor(start)
        .build()
    )

    dot = WorkflowViz(wf).to_digraph()

    # 条件付きエッジは破線でラベル付きであるべきである
    assert '"start" -> "mid" [style=dashed, label="conditional"];' in dot
    # 非条件付きエッジはプレーンであるべきである
    assert '"mid" -> "end"' in dot
    assert '"mid" -> "end" [style=dashed' not in dot


def test_workflow_viz_fan_in_edge_group():
    """ファンインエッジがラベル付きの中間ノードとルーティングされたエッジをレンダリングすることをテストする。"""
    start = MockExecutor(id="start")
    s1 = MockExecutor(id="s1")
    s2 = MockExecutor(id="s2")
    t = ListStrTargetExecutor(id="t")

    # 接続されたワークフローを構築する：startがs1とs2にファンアウトし、それらがtにファンインする
    wf = (
        WorkflowBuilder()
        .add_fan_out_edges(start, [s1, s2])
        .add_fan_in_edges([s1, s2], t)
        .set_start_executor(start)
        .build()
    )

    dot = WorkflowViz(wf).to_digraph()

    # 特別なスタイリングとラベルを持つ単一のファンインノードが存在するはずである
    lines = [line.strip() for line in dot.splitlines()]
    fan_in_lines = [line for line in lines if "shape=ellipse" in line and 'label="fan-in"' in line]
    assert len(fan_in_lines) == 1

    # 行 "<id>" [shape=ellipse, ... label="fan-in"] から中間ノードIDを抽出する
    fan_in_line = fan_in_lines[0]
    first_quote = fan_in_line.find('"')
    second_quote = fan_in_line.find('"', first_quote + 1)
    assert first_quote != -1 and second_quote != -1
    fan_in_node_id = fan_in_line[first_quote + 1 : second_quote]
    assert fan_in_node_id  # 空でない

    # エッジは中間ノードを経由してルーティングされ、ターゲットに直接ではないべきである
    assert f'"s1" -> "{fan_in_node_id}";' in dot
    assert f'"s2" -> "{fan_in_node_id}";' in dot
    assert f'"{fan_in_node_id}" -> "t";' in dot

    # 直接エッジが存在しないことを確認する
    assert '"s1" -> "t"' not in dot
    assert '"s2" -> "t"' not in dot


def test_workflow_viz_to_mermaid_basic():
    """Mermaid: 基本的なワークフローノードとエッジがstartラベル付きで存在する。"""
    executor1 = MockExecutor(id="executor1")
    executor2 = MockExecutor(id="executor2")

    workflow = WorkflowBuilder().add_edge(executor1, executor2).set_start_executor(executor1).build()
    mermaid = WorkflowViz(workflow).to_mermaid()

    # 開始ノードと通常ノード
    assert 'executor1["executor1 (Start)"]' in mermaid
    assert 'executor2["executor2"]' in mermaid
    # エッジはサニタイズされたIDを使用する（ここでのIDと同じ）
    assert "executor1 --> executor2" in mermaid


def test_workflow_viz_mermaid_conditional_edge():
    """Mermaid: 条件付きエッジは点線でラベル付きである。"""
    start = MockExecutor(id="start")
    mid = MockExecutor(id="mid")

    def only_if_foo(msg: str) -> bool:  # pragma: no cover - simple predicate
        return msg == "foo"

    wf = WorkflowBuilder().add_edge(start, mid, condition=only_if_foo).set_start_executor(start).build()
    mermaid = WorkflowViz(wf).to_mermaid()

    assert "start -. conditional .-> mid" in mermaid


def test_workflow_viz_mermaid_fan_in_edge_group():
    """Mermaid: ファンインは中間ノードを使用し、エッジを経由してルーティングする。"""
    start = MockExecutor(id="start")
    s1 = MockExecutor(id="s1")
    s2 = MockExecutor(id="s2")
    t = ListStrTargetExecutor(id="t")

    wf = (
        WorkflowBuilder()
        .add_fan_out_edges(start, [s1, s2])
        .add_fan_in_edges([s1, s2], t)
        .set_start_executor(start)
        .build()
    )

    mermaid = WorkflowViz(wf).to_mermaid()
    lines = [line.strip() for line in mermaid.splitlines()]
    # ファンインノードを見つける（行は((fan-in))で終わる）
    fan_lines = [ln for ln in lines if ln.endswith("((fan-in))")]
    assert len(fan_lines) == 1
    fan_line = fan_lines[0]
    # fan_inノードは <id>((fan-in)) として出力され、<id>を抽出する
    token = fan_line.strip()
    suffix = "((fan-in))"
    assert token.endswith(suffix)
    fan_node_id = token[: -len(suffix)]
    assert fan_node_id

    # 中間ノード経由のルーティングを確認する
    assert f"s1 --> {fan_node_id}" in mermaid
    assert f"s2 --> {fan_node_id}" in mermaid
    assert f"{fan_node_id} --> t" in mermaid

    # ターゲットへの直接エッジが存在しないことを確認する
    assert "s1 --> t" not in mermaid
    assert "s2 --> t" not in mermaid


def test_workflow_viz_sub_workflow_digraph(basic_sub_workflow):
    """WorkflowVizがサブワークフローをDOT形式で可視化できることをテストする。"""
    main_workflow = basic_sub_workflow["main_workflow"]

    viz = WorkflowViz(main_workflow)
    dot_content = viz.to_digraph()

    # メインワークフローノードが存在することを確認する
    assert "main_executor" in dot_content
    assert "workflow_executor_1" in dot_content
    assert "final_executor" in dot_content

    # サブワークフローがクラスタとしてレンダリングされていることを確認する
    assert "subgraph cluster_" in dot_content
    assert "sub-workflow: workflow_executor_1" in dot_content

    # サブワークフローノードが名前空間化されていることを確認する
    assert '"workflow_executor_1/sub_exec1"' in dot_content
    assert '"workflow_executor_1/sub_exec2"' in dot_content

    # サブワークフローのエッジが存在することを確認する
    assert '"workflow_executor_1/sub_exec1" -> "workflow_executor_1/sub_exec2"' in dot_content


def test_workflow_viz_sub_workflow_mermaid(basic_sub_workflow):
    """WorkflowVizがサブワークフローをMermaid形式で可視化できることをテストする。"""
    main_workflow = basic_sub_workflow["main_workflow"]

    viz = WorkflowViz(main_workflow)
    mermaid_content = viz.to_mermaid()

    # メインワークフローノードが存在することを確認する
    assert "main_executor" in mermaid_content
    assert "workflow_executor_1" in mermaid_content
    assert "final_executor" in mermaid_content

    # サブワークフローがサブグラフとしてレンダリングされていることを確認する
    assert "subgraph workflow_executor_1" in mermaid_content
    assert "end" in mermaid_content

    # Mermaid用にサブワークフローノードが適切に名前空間化されていることを確認する
    assert "workflow_executor_1__sub_exec1" in mermaid_content
    assert "workflow_executor_1__sub_exec2" in mermaid_content


def test_workflow_viz_nested_sub_workflows():
    """深くネストされたサブワークフローの可視化をテストする。"""
    # 最も内側のサブワークフローを作成する
    inner_exec = MockExecutor(id="inner_exec")
    inner_workflow = WorkflowBuilder().set_start_executor(inner_exec).build()

    # 内側のサブワークフローを含む中間のサブワークフローを作成する
    inner_workflow_executor = WorkflowExecutor(inner_workflow, id="inner_wf_exec")
    middle_exec = MockExecutor(id="middle_exec")

    middle_workflow = (
        WorkflowBuilder().add_edge(middle_exec, inner_workflow_executor).set_start_executor(middle_exec).build()
    )

    # 外側のワークフローを作成する
    middle_workflow_executor = WorkflowExecutor(middle_workflow, id="middle_wf_exec")
    outer_exec = MockExecutor(id="outer_exec")

    outer_workflow = (
        WorkflowBuilder().add_edge(outer_exec, middle_workflow_executor).set_start_executor(outer_exec).build()
    )

    viz = WorkflowViz(outer_workflow)
    dot_content = viz.to_digraph()

    # すべてのレベルが存在することを確認する
    assert "outer_exec" in dot_content
    assert "middle_wf_exec" in dot_content
    assert "inner_wf_exec" in dot_content

    # ネストされたクラスタを確認する
    assert "subgraph cluster_" in dot_content
    # ネスト構造のために複数のサブグラフが存在するはずである
    subgraph_count = dot_content.count("subgraph cluster_")
    assert subgraph_count >= 2  # ネストの各レベルに少なくとも一つずつ存在する
