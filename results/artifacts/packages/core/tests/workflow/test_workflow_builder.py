# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Any

import pytest

from agent_framework import (
    AgentExecutor,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Executor,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)


class DummyAgent(BaseAgent):
    async def run(self, messages=None, *, thread: AgentThread | None = None, **kwargs):  # type: ignore[override]
        norm: list[ChatMessage] = []
        if messages:
            for m in messages:  # type: ignore[iteration-over-optional]
                if isinstance(m, ChatMessage):
                    norm.append(m)
                elif isinstance(m, str):
                    norm.append(ChatMessage(role=Role.USER, text=m))
        return AgentRunResponse(messages=norm)

    async def run_stream(self, messages=None, *, thread: AgentThread | None = None, **kwargs):  # type: ignore[override]
        # 最小限のasync generator
        yield AgentRunResponseUpdate()


def test_builder_accepts_agents_directly():
    agent1 = DummyAgent(id="agent1", name="writer")
    agent2 = DummyAgent(id="agent2", name="reviewer")

    wf = WorkflowBuilder().set_start_executor(agent1).add_edge(agent1, agent2).build()

    # 自動ラップされたexecutorsがagent名をIDとして使用することを確認する
    assert wf.start_executor_id == "writer"
    assert any(isinstance(e, AgentExecutor) and e.id in {"writer", "reviewer"} for e in wf.executors.values())


@dataclass
class MockMessage:
    """テスト用のモックメッセージ。"""

    data: Any


class MockExecutor(Executor):
    """テスト用のモックexecutor。"""

    @handler
    async def mock_handler(self, message: MockMessage, ctx: WorkflowContext[MockMessage]) -> None:
        """何もしないモックハンドラ。"""
        pass


class MockAggregator(Executor):
    """複数のexecutorsの結果を集約するモックexecutor。"""

    @handler
    async def mock_handler(self, messages: list[MockMessage], ctx: WorkflowContext[MockMessage]) -> None:
        # このモックは単にデータを1増やして返す
        pass


def test_workflow_builder_without_start_executor_throws():
    """start executorなしでworkflow builderを作成するテスト。"""

    builder = WorkflowBuilder()
    with pytest.raises(ValueError):
        builder.build()


def test_workflow_builder_fluent_api():
    """workflow builderのフルーエントAPIをテストする。"""
    executor_a = MockExecutor(id="executor_a")
    executor_b = MockExecutor(id="executor_b")
    executor_c = MockExecutor(id="executor_c")
    executor_d = MockExecutor(id="executor_d")
    executor_e = MockAggregator(id="executor_e")
    executor_f = MockExecutor(id="executor_f")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_fan_out_edges(executor_b, [executor_c, executor_d])
        .add_fan_in_edges([executor_c, executor_d], executor_e)
        .add_chain([executor_e, executor_f])
        .set_max_iterations(5)
        .build()
    )

    assert len(workflow.edge_groups) == 4
    assert workflow.start_executor_id == executor_a.id
    assert len(workflow.executors) == 6


def test_add_agent_with_custom_parameters():
    """カスタムパラメータでagentを追加するテスト。"""
    agent = DummyAgent(id="agent_custom", name="custom_agent")
    builder = WorkflowBuilder()

    # カスタムパラメータでagentを追加する
    result = builder.add_agent(agent, output_response=True, id="my_custom_id")

    # add_agentがチェーン用にbuilderを返すことを検証する
    assert result is builder

    # workflowをビルドしexecutorが存在することを検証する
    workflow = builder.set_start_executor(agent).build()
    assert "my_custom_id" in workflow.executors

    # executorが正しいパラメータで作成されたことを検証する
    executor = workflow.executors["my_custom_id"]
    assert isinstance(executor, AgentExecutor)
    assert executor.id == "my_custom_id"
    assert getattr(executor, "_output_response", False) is True


def test_add_agent_reuses_same_wrapper():
    """同じagentインスタンスを複数回使うと同じラッパーを再利用することをテストする。"""
    agent = DummyAgent(id="agent_reuse", name="reuse_agent")
    builder = WorkflowBuilder()

    # 特定のパラメータでagentを追加する
    builder.add_agent(agent, output_response=True, id="agent_exec")

    # add_edgeで同じagentインスタンスを使う - 同じラッパーを再利用すべき
    builder.set_start_executor(agent)

    workflow = builder.build()

    # このagentに対してexecutorが1つだけ存在することを検証する
    assert workflow.start_executor_id == "agent_exec"
    assert "agent_exec" in workflow.executors
    assert len([e for e in workflow.executors.values() if isinstance(e, AgentExecutor)]) == 1

    # executorがadd_agentのパラメータを持っていることを検証する
    start_executor = workflow.get_start_executor()
    assert isinstance(start_executor, AgentExecutor)
    assert getattr(start_executor, "_output_response", False) is True


def test_add_agent_then_use_in_edges():
    """add_agentで追加したagentがedge定義で使えることをテストする。"""
    agent1 = DummyAgent(id="agent1", name="first")
    agent2 = DummyAgent(id="agent2", name="second")
    builder = WorkflowBuilder()

    # 特定の設定でagentを追加する
    builder.add_agent(agent1, output_response=False, id="exec1")
    builder.add_agent(agent2, output_response=True, id="exec2")

    # 同じagentインスタンスを使ってエッジを作成する
    workflow = builder.set_start_executor(agent1).add_edge(agent1, agent2).build()

    # executorが設定を維持していることを検証する
    assert workflow.start_executor_id == "exec1"
    assert "exec1" in workflow.executors
    assert "exec2" in workflow.executors

    e1 = workflow.executors["exec1"]
    e2 = workflow.executors["exec2"]

    assert isinstance(e1, AgentExecutor)
    assert isinstance(e2, AgentExecutor)
    assert getattr(e1, "_output_response", True) is False
    assert getattr(e2, "_output_response", False) is True


def test_add_agent_without_explicit_id_uses_agent_name():
    """明示的なidがない場合、add_agentがagent名をidとして使うことをテストする。"""
    agent = DummyAgent(id="agent_x", name="named_agent")
    builder = WorkflowBuilder()

    result = builder.add_agent(agent)

    # add_agentがチェーン用にbuilderを返すことを検証する
    assert result is builder

    workflow = builder.set_start_executor(agent).build()
    assert "named_agent" in workflow.executors

    # executorのidがagent名と一致することを検証する
    executor = workflow.executors["named_agent"]
    assert executor.id == "named_agent"


def test_add_agent_duplicate_id_raises_error():
    """重複IDのagent追加がエラーを発生させることをテストする。"""
    agent1 = DummyAgent(id="agent1", name="first")
    agent2 = DummyAgent(id="agent2", name="first")  # agent1と同じ名前
    builder = WorkflowBuilder()

    # 最初のagentを追加する
    builder.add_agent(agent1)

    # 同じ名前の2番目のagent追加はValueErrorを発生させるべき
    with pytest.raises(ValueError, match="Duplicate executor ID"):
        builder.add_agent(agent2)
