# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from agent_framework import (
    Executor,
    InProcRunnerContext,
    Message,
    SharedState,
    WorkflowContext,
    handler,
)
from agent_framework._workflows._edge import (
    Edge,
    FanInEdgeGroup,
    FanOutEdgeGroup,
    SingleEdgeGroup,
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)
from agent_framework._workflows._edge_runner import create_edge_runner
from agent_framework.observability import EdgeGroupDeliveryStatus

# テスト用に追加。


@dataclass
class MockMessage:
    """テスト用のモックメッセージ。"""

    data: Any


@dataclass
class MockMessageSecondary:
    """テスト用の二次モックメッセージ。"""

    data: Any


class MockExecutor(Executor):
    """テスト用のモックexecutor。"""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.call_count: int = 0
        self.last_message: MockMessage | None = None

    @handler
    async def mock_handler(self, message: MockMessage, ctx: WorkflowContext) -> None:
        """何もしないモックハンドラー。"""
        self.call_count += 1
        self.last_message = message


class MockExecutorSecondary(Executor):
    """テスト用の二次モックexecutor。"""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.call_count: int = 0
        self.last_message: MockMessageSecondary | None = None

    @handler
    async def mock_handler_secondary(self, message: MockMessageSecondary, ctx: WorkflowContext) -> None:
        """何もしない二次モックハンドラー。"""
        self.call_count += 1
        self.last_message = message


class MockAggregator(Executor):
    """テスト用のモックアグリゲーター。"""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.call_count: int = 0
        self.last_message: list[MockMessage] | list[MockMessageSecondary] | None = None

    @handler
    async def mock_aggregator_handler(self, message: list[MockMessage], ctx: WorkflowContext) -> None:
        """何もしないモックアグリゲーターハンドラー。"""
        self.call_count += 1
        self.last_message = message

    @handler
    async def mock_aggregator_handler_secondary(
        self,
        message: list[MockMessageSecondary],
        ctx: WorkflowContext,
    ) -> None:
        """何もしないモックアグリゲーターハンドラー。"""
        self.call_count += 1
        self.last_message = message


class MockAggregatorSecondary(Executor):
    """テスト用にunion型のハンドラーを持つモックアグリゲーター。"""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.call_count: int = 0
        self.last_message: list[MockMessage | MockMessageSecondary] | None = None

    @handler
    async def mock_aggregator_handler_combine(
        self,
        message: list[MockMessage | MockMessageSecondary],
        ctx: WorkflowContext,
    ) -> None:
        """何もしないモックアグリゲーターハンドラー。"""
        self.call_count += 1
        self.last_message = message


# region Edge


def test_create_edge():
    """sourceとtarget executorを持つエッジの作成をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    edge = Edge(source_id=source.id, target_id=target.id)

    assert edge.source_id == "source_executor"
    assert edge.target_id == "target_executor"
    assert edge.id == f"{edge.source_id}{Edge.ID_SEPARATOR}{edge.target_id}"


def test_edge_can_handle():
    """sourceとtarget executorを持つエッジの作成をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    edge = Edge(source_id=source.id, target_id=target.id)

    assert edge.should_route(MockMessage(data="test"))


# endregion Edge region SingleEdgeGroup


def test_single_edge_group():
    """単一エッジグループの作成をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    assert edge_group.source_executor_ids == [source.id]
    assert edge_group.target_executor_ids == [target.id]
    assert edge_group.edges[0].source_id == "source_executor"
    assert edge_group.edges[0].target_id == "target_executor"


def test_single_edge_group_with_condition():
    """条件付きの単一エッジグループの作成をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id, condition=lambda x: x.data == "test")

    assert edge_group.source_executor_ids == [source.id]
    assert edge_group.target_executor_ids == [target.id]
    assert edge_group.edges[0].source_id == "source_executor"
    assert edge_group.edges[0].target_id == "target_executor"
    assert edge_group.edges[0]._condition is not None  # type: ignore


async def test_single_edge_group_send_message() -> None:
    """単一エッジランナーを通じたメッセージ送信をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True


async def test_single_edge_group_send_message_with_target() -> None:
    """単一エッジランナーを通じたメッセージ送信をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id=target.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True


async def test_single_edge_group_send_message_with_invalid_target() -> None:
    """単一エッジランナーを通じたメッセージ送信をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id="invalid_target")

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_single_edge_group_send_message_with_invalid_data() -> None:
    """無効なデータで単一エッジランナーを通じたメッセージ送信をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = "invalid_data"
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_single_edge_group_send_message_with_condition_pass() -> None:
    """条件が通る単一エッジランナーを通じたメッセージ送信をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    # data == "test"のときに通る条件付きのエッジグループを作成します。
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id, condition=lambda x: x.data == "test")

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True
    assert target.call_count == 1
    assert target.last_message.data == "test"


async def test_single_edge_group_send_message_with_condition_fail() -> None:
    """条件が失敗する単一エッジランナーを通じたメッセージ送信をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    # data == "test"のときに通る条件付きのエッジグループを作成します。
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id, condition=lambda x: x.data == "test")

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="different")
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    # メッセージは処理されたが条件が失敗したためTrueを返すはずです。
    assert success is True
    # 条件が失敗したためtargetは呼び出されるべきではありません。
    assert target.call_count == 0


async def test_single_edge_group_tracing_success(span_exporter) -> None:
    """単一エッジグループ処理が適切な成功スパンを作成することをテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    # トレース情報を持つメッセージをシミュレートするためにトレースコンテキストとスパンIDを作成します。
    trace_contexts = [{"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}]
    source_span_ids = ["00f067aa0ba902b7"]

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, trace_contexts=trace_contexts, source_span_ids=source_span_ids)

    # ビルドスパンをクリアします。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "SingleEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is True
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DELIVERED.value
    assert span.attributes.get("edge_group.id") is not None
    assert span.attributes.get("message.source_id") == source.id

    # スパンリンクが作成されていることを検証します。
    assert span.links is not None
    assert len(span.links) == 1

    link = span.links[0]
    # リンクが正しいトレースとスパンを指していることを検証します。
    assert link.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert link.context.span_id == int("00f067aa0ba902b7", 16)


async def test_single_edge_group_tracing_condition_failure(span_exporter) -> None:
    """単一エッジグループ処理が条件失敗に対して適切なスパンを作成することをテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id, condition=lambda x: x.data == "pass")

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="fail")
    message = Message(data=data, source_id=source.id)

    # ビルドスパンをクリアします。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True  # Trueを返すが条件は失敗しました。

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "SingleEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is False
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DROPPED_CONDITION_FALSE.value


async def test_single_edge_group_tracing_type_mismatch(span_exporter) -> None:
    """単一エッジグループ処理が型不一致に対して適切なスパンを作成することをテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    # 互換性のないデータ型を送信します。
    data = "invalid_data"
    message = Message(data=data, source_id=source.id)

    # ビルドスパンをクリアします。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "SingleEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is False
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH.value


async def test_single_edge_group_tracing_target_mismatch(span_exporter) -> None:
    """単一エッジグループ処理がターゲット不一致に対して適切なスパンを作成することをテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    executors: dict[str, Executor] = {source.id: source, target.id: target}
    edge_group = SingleEdgeGroup(source_id=source.id, target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id="wrong_target")

    # ビルドスパンをクリアします。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "SingleEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is False
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DROPPED_TARGET_MISMATCH.value
    assert span.attributes.get("message.target_id") == "wrong_target"


# endregion SingleEdgeGroup region FanOutEdgeGroup


def test_source_edge_group():
    """ファンアウトグループの作成をテストします。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    assert edge_group.source_executor_ids == [source.id]
    assert edge_group.target_executor_ids == [target1.id, target2.id]
    assert len(edge_group.edges) == 2
    assert edge_group.edges[0].source_id == "source_executor"
    assert edge_group.edges[0].target_id == "target_executor_1"
    assert edge_group.edges[1].source_id == "source_executor"
    assert edge_group.edges[1].target_id == "target_executor_2"


def test_source_edge_group_invalid_number_of_targets() -> None:
    """無効なターゲット数でのファンアウトグループの作成をテストします。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    with pytest.raises(ValueError, match="FanOutEdgeGroup must contain at least two targets"):
        FanOutEdgeGroup(source_id=source.id, target_ids=[target.id])


async def test_source_edge_group_send_message() -> None:
    """fan-out edge runner を通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)

    assert success is True
    assert target1.call_count == 1
    assert target2.call_count == 1


async def test_source_edge_group_send_message_with_target() -> None:
    """ターゲット付きの fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id=target1.id)

    success = await edge_runner.send_message(message, shared_state, ctx)

    assert success is True
    assert target1.call_count == 1
    assert target2.call_count == 0  # message のターゲットが target1 のため、target2 は呼び出されるべきではない。


async def test_source_edge_group_send_message_with_invalid_target() -> None:
    """無効なターゲットを持つ fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id="invalid_target")

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_source_edge_group_send_message_with_invalid_data() -> None:
    """無効なデータを持つ fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = "invalid_data"
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_source_edge_group_send_message_only_one_successful_send() -> None:
    """メッセージを処理できるエッジが1つだけの fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutorSecondary(id="target_executor_2")

    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)

    assert success is True
    assert target1.call_count == 1  # target1 は MockMessage を処理できる。
    assert target2.call_count == 0  # target2 (MockExecutorSecondary) は MockMessage を処理できない。


def test_source_edge_group_with_selection_func():
    """パーティショニングエッジグループを作成するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id],
    )

    assert edge_group.source_executor_ids == [source.id]
    assert edge_group.target_executor_ids == [target1.id, target2.id]
    assert len(edge_group.edges) == 2
    assert edge_group.edges[0].source_id == "source_executor"
    assert edge_group.edges[0].target_id == "target_executor_1"
    assert edge_group.edges[1].source_id == "source_executor"
    assert edge_group.edges[1].target_id == "target_executor_2"


async def test_source_edge_group_with_selection_func_send_message() -> None:
    """選択関数を持つ fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id, target2.id],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id)

    with patch("agent_framework._workflows._edge_runner.EdgeRunner._execute_on_target") as mock_send:
        success = await edge_runner.send_message(message, shared_state, ctx)

        assert success is True

        assert mock_send.call_count == 2


async def test_source_edge_group_with_selection_func_send_message_with_invalid_selection_result() -> None:
    """無効な選択結果を持つ選択関数付き fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id, "invalid_target"],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id)

    with pytest.raises(RuntimeError):
        await edge_runner.send_message(message, shared_state, ctx)


async def test_source_edge_group_with_selection_func_send_message_with_target() -> None:
    """ターゲット付きの選択関数を持つ fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id, target2.id],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id=target1.id)

    with patch("agent_framework._workflows._edge_runner.EdgeRunner._execute_on_target") as mock_send:
        success = await edge_runner.send_message(message, shared_state, ctx)

        assert success is True
        assert mock_send.call_count == 1
        assert mock_send.call_args[0][0] == target1.id


async def test_source_edge_group_with_selection_func_send_message_with_target_not_in_selection() -> None:
    """選択に含まれないターゲットを持つ選択関数付き fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id],  # Only target1 will receive the message
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, target_id=target2.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_source_edge_group_with_selection_func_send_message_with_invalid_data() -> None:
    """無効なデータを持つ選択関数付き fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id, target2.id],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = "invalid_data"
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_source_edge_group_with_selection_func_send_message_with_target_invalid_data() -> None:
    """ターゲットと無効なデータを持つ選択関数付き fan-out グループを通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = FanOutEdgeGroup(
        source_id=source.id,
        target_ids=[target1.id, target2.id],
        selection_func=lambda data, target_ids: [target1.id, target2.id],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = "invalid_data"
    message = Message(data=data, source_id=source.id, target_id=target1.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_fan_out_edge_group_tracing_success(span_exporter) -> None:
    """fan-out edge group の処理が適切な成功スパンを作成することをテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    # トレース情報を持つメッセージをシミュレートするためにトレースコンテキストとスパンIDを作成。
    trace_contexts = [{"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}]
    source_span_ids = ["00f067aa0ba902b7"]

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source.id, trace_contexts=trace_contexts, source_span_ids=source_span_ids)

    # ビルドスパンをクリアする。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "FanOutEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is True
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DELIVERED.value
    assert span.attributes.get("edge_group.id") is not None
    assert span.attributes.get("message.source_id") == source.id

    # スパンリンクが作成されていることを検証。
    assert span.links is not None
    assert len(span.links) == 1

    link = span.links[0]
    # リンクが正しいトレースとスパンを指していることを検証。
    assert link.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert link.context.span_id == int("00f067aa0ba902b7", 16)


async def test_fan_out_edge_group_tracing_with_target(span_exporter) -> None:
    """ターゲットメッセージに対して fan-out edge group の処理が適切なスパンを作成することをテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_group = FanOutEdgeGroup(source_id=source.id, target_ids=[target1.id, target2.id])

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    # トレース情報を持つメッセージをシミュレートするためにトレースコンテキストとスパンIDを作成。
    trace_contexts = [{"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}]
    source_span_ids = ["00f067aa0ba902b7"]

    data = MockMessage(data="test")
    message = Message(
        data=data,
        source_id=source.id,
        target_id=target1.id,
        trace_contexts=trace_contexts,
        source_span_ids=source_span_ids,
    )

    # ビルドスパンをクリアする。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "FanOutEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is True
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DELIVERED.value
    assert span.attributes.get("message.target_id") == target1.id

    # スパンリンクが作成されていることを検証。
    assert span.links is not None
    assert len(span.links) == 1

    link = span.links[0]
    # リンクが正しいトレースとスパンを指していることを検証。
    assert link.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert link.context.span_id == int("00f067aa0ba902b7", 16)


# endregion FanOutEdgeGroup region FanInEdgeGroup


def test_target_edge_group():
    """fan-in edge group を作成するテスト。"""
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    assert edge_group.source_executor_ids == [source1.id, source2.id]
    assert edge_group.target_executor_ids == [target.id]
    assert len(edge_group.edges) == 2
    assert edge_group.edges[0].source_id == "source_executor_1"
    assert edge_group.edges[0].target_id == "target_executor"
    assert edge_group.edges[1].source_id == "source_executor_2"
    assert edge_group.edges[1].target_id == "target_executor"


def test_target_edge_group_invalid_number_of_sources():
    """無効な数のソースを持つ fan-in edge group を作成するテスト。"""
    source = MockExecutor(id="source_executor")
    target = MockAggregator(id="target_executor")

    with pytest.raises(ValueError, match="FanInEdgeGroup must contain at least two sources"):
        FanInEdgeGroup(source_ids=[source.id], target_id=target.id)


async def test_target_edge_group_send_message_buffer() -> None:
    """バッファリング付きの fan-in edge group を通じてメッセージを送信するテスト。"""
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")

    with patch("agent_framework._workflows._edge_runner.EdgeRunner._execute_on_target") as mock_send:
        success = await edge_runner.send_message(
            Message(data=data, source_id=source1.id),
            shared_state,
            ctx,
        )

        assert success is True
        assert mock_send.call_count == 0  # メッセージはバッファリングされ、2番目のソースを待つべき。
        assert len(edge_runner._buffer[source1.id]) == 1  # type: ignore

        success = await edge_runner.send_message(
            Message(data=data, source_id=source2.id),
            shared_state,
            ctx,
        )
        assert success is True
        assert mock_send.call_count == 1  # 両方のソースがメッセージを送信したので、メッセージは今送信されるべき。

        # 送信後にバッファはクリアされるべき。
        assert not edge_runner._buffer  # type: ignore


async def test_target_edge_group_send_message_with_invalid_target() -> None:
    """無効なターゲットを持つ fan-in edge group を通じてメッセージを送信するテスト。"""
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")
    message = Message(data=data, source_id=source1.id, target_id="invalid_target")

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_target_edge_group_send_message_with_invalid_data() -> None:
    """無効なデータを持つ fan-in edge group を通じてメッセージを送信するテスト。"""
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = "invalid_data"
    message = Message(data=data, source_id=source1.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_fan_in_edge_group_tracing_buffered(span_exporter) -> None:
    """バッファリングされたメッセージに対して fan-in edge group の処理が適切なスパンを作成することをテスト。"""
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")

    # トレース情報を持つメッセージをシミュレートするためにトレースコンテキストとスパンIDを作成。
    trace_contexts1 = [{"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}]
    source_span_ids1 = ["00f067aa0ba902b7"]

    trace_contexts2 = [{"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b8-01"}]
    source_span_ids2 = ["00f067aa0ba902b8"]

    # ビルドスパンをクリアする。
    span_exporter.clear()

    # 最初のメッセージを送信（バッファリングされるべき）。
    success = await edge_runner.send_message(
        Message(data=data, source_id=source1.id, trace_contexts=trace_contexts1, source_span_ids=source_span_ids1),
        shared_state,
        ctx,
    )
    assert success is True

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "FanInEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is True
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.BUFFERED.value
    assert span.attributes.get("message.source_id") == source1.id

    # 最初のメッセージに対してスパンリンクが作成されていることを検証。
    assert span.links is not None
    assert len(span.links) == 1

    link = span.links[0]
    # リンクが正しいトレースとスパンを指していることを検証。
    assert link.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert link.context.span_id == int("00f067aa0ba902b7", 16)

    # スパンをクリアして2番目のメッセージを送信（配信をトリガーすべき）。
    span_exporter.clear()

    success = await edge_runner.send_message(
        Message(data=data, source_id=source2.id, trace_contexts=trace_contexts2, source_span_ids=source_span_ids2),
        shared_state,
        ctx,
    )
    assert success is True

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "FanInEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is True
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DELIVERED.value
    assert span.attributes.get("message.source_id") == source2.id

    # 2番目のメッセージに対してスパンリンクが作成されていることを検証。
    assert span.links is not None
    assert len(span.links) == 1

    link = span.links[0]
    # 2番目のメッセージの正しいトレースとスパンを指すリンクを検証。
    assert link.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert link.context.span_id == int("00f067aa0ba902b8", 16)


async def test_fan_in_edge_group_tracing_type_mismatch(span_exporter) -> None:
    """型不一致に対して fan-in edge group の処理が適切なスパンを作成することをテスト。"""
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    edge_runner = create_edge_runner(edge_group, executors)
    shared_state = SharedState()
    ctx = InProcRunnerContext()

    # 互換性のないデータ型を送信。
    data = "invalid_data"
    message = Message(data=data, source_id=source1.id)

    # ビルドスパンをクリアする。
    span_exporter.clear()

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False

    spans = span_exporter.get_finished_spans()
    edge_group_spans = [s for s in spans if s.name == "edge_group.process"]

    assert len(edge_group_spans) == 1

    span = edge_group_spans[0]
    assert span.attributes is not None
    assert span.attributes.get("edge_group.type") == "FanInEdgeGroup"
    assert span.attributes.get("edge_group.delivered") is False
    assert span.attributes.get("edge_group.delivery_status") == EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH.value


async def test_fan_in_edge_group_with_multiple_message_types() -> None:
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregatorSecondary(id="target_executor")

    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")

    success = await edge_runner.send_message(
        Message(data=data, source_id=source1.id),
        shared_state,
        ctx,
    )
    assert success

    data2 = MockMessageSecondary(data="test")
    success = await edge_runner.send_message(
        Message(data=data2, source_id=source2.id),
        shared_state,
        ctx,
    )
    assert success


async def test_fan_in_edge_group_with_multiple_message_types_failed() -> None:
    source1 = MockExecutor(id="source_executor_1")
    source2 = MockExecutor(id="source_executor_2")
    target = MockAggregator(id="target_executor")

    edge_group = FanInEdgeGroup(source_ids=[source1.id, source2.id], target_id=target.id)

    executors: dict[str, Executor] = {source1.id: source1, source2.id: source2, target.id: target}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data="test")

    success = await edge_runner.send_message(
        Message(data=data, source_id=source1.id),
        shared_state,
        ctx,
    )
    assert success

    with pytest.raises(RuntimeError):
        # `MockAggregator` は `list[MockMessage]` と `list[MockMessageSecondary]`
        # を個別に処理できる（つまり、それぞれの型に対してハンドラを持つ）が、`list[MockMessage |
        # MockMessageSecondary]`（両方の型が混在するリスト）は処理できない。fan-in edge group では、ターゲットの
        # executor はソース executor からのすべてのメッセージ型をユニオンとして処理する必要がある。
        data2 = MockMessageSecondary(data="test")
        _ = await edge_runner.send_message(
            Message(data=data2, source_id=source2.id),
            shared_state,
            ctx,
        )


# endregion FanInEdgeGroup region SwitchCaseEdgeGroup


def test_switch_case_edge_group() -> None:
    """switch case edge group を作成するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = SwitchCaseEdgeGroup(
        source_id=source.id,
        cases=[
            SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target1.id),
            SwitchCaseEdgeGroupDefault(target_id=target2.id),
        ],
    )

    assert edge_group.source_executor_ids == [source.id]
    assert edge_group.target_executor_ids == [target1.id, target2.id]
    assert len(edge_group.edges) == 2
    assert edge_group.edges[0].source_id == "source_executor"
    assert edge_group.edges[0].target_id == "target_executor_1"
    assert edge_group.edges[1].source_id == "source_executor"
    assert edge_group.edges[1].target_id == "target_executor_2"

    assert edge_group._selection_func is not None  # type: ignore
    assert edge_group._selection_func(MockMessage(data=-1), [target1.id, target2.id]) == [target1.id]  # type: ignore
    assert edge_group._selection_func(MockMessage(data=1), [target1.id, target2.id]) == [target2.id]  # type: ignore


def test_switch_case_edge_group_invalid_number_of_cases():
    """無効な数のケースを持つ switch case edge group を作成するテスト。"""
    source = MockExecutor(id="source_executor")
    target = MockExecutor(id="target_executor")

    with pytest.raises(
        ValueError, match=r"SwitchCaseEdgeGroup must contain at least two cases \(including the default case\)."
    ):
        SwitchCaseEdgeGroup(
            source_id=source.id,
            cases=[
                SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target.id),
            ],
        )

    with pytest.raises(ValueError, match="SwitchCaseEdgeGroup must contain exactly one default case."):
        SwitchCaseEdgeGroup(
            source_id=source.id,
            cases=[
                SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target.id),
                SwitchCaseEdgeGroupCase(condition=lambda x: x.data >= 0, target_id=target.id),
            ],
        )


def test_switch_case_edge_group_invalid_number_of_default_cases():
    """無効な数の条件を持つ switch case edge group を作成するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    with pytest.raises(ValueError, match="SwitchCaseEdgeGroup must contain exactly one default case."):
        SwitchCaseEdgeGroup(
            source_id=source.id,
            cases=[
                SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target1.id),
                SwitchCaseEdgeGroupDefault(target_id=target2.id),
                SwitchCaseEdgeGroupDefault(target_id=target2.id),
            ],
        )


async def test_switch_case_edge_group_send_message() -> None:
    """switch case edge group を通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = SwitchCaseEdgeGroup(
        source_id=source.id,
        cases=[
            SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target1.id),
            SwitchCaseEdgeGroupDefault(target_id=target2.id),
        ],
    )
    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data=-1)
    message = Message(data=data, source_id=source.id)

    with patch("agent_framework._workflows._edge_runner.EdgeRunner._execute_on_target") as mock_send:
        success = await edge_runner.send_message(message, shared_state, ctx)

        assert success is True
        assert mock_send.call_count == 1

    # デフォルト条件は
    data = MockMessage(data=1)
    message = Message(data=data, source_id=source.id)
    with patch("agent_framework._workflows._edge_runner.EdgeRunner._execute_on_target") as mock_send:
        success = await edge_runner.send_message(message, shared_state, ctx)

        assert success is True
        assert mock_send.call_count == 1


async def test_switch_case_edge_group_send_message_with_invalid_target() -> None:
    """無効なターゲットを持つ switch case edge group を通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = SwitchCaseEdgeGroup(
        source_id=source.id,
        cases=[
            SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target1.id),
            SwitchCaseEdgeGroupDefault(target_id=target2.id),
        ],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data=-1)
    message = Message(data=data, source_id=source.id, target_id="invalid_target")

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


async def test_switch_case_edge_group_send_message_with_valid_target() -> None:
    """ターゲットを持つ switch case edge group を通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = SwitchCaseEdgeGroup(
        source_id=source.id,
        cases=[
            SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target1.id),
            SwitchCaseEdgeGroupDefault(target_id=target2.id),
        ],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = MockMessage(data=1)  # 条件は失敗する。
    message = Message(data=data, source_id=source.id, target_id=target1.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False

    data = MockMessage(data=-1)  # 条件は成功する。
    message = Message(data=data, source_id=source.id, target_id=target1.id)
    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is True


async def test_switch_case_edge_group_send_message_with_invalid_data() -> None:
    """無効なデータを持つ switch case edge group を通じてメッセージを送信するテスト。"""
    source = MockExecutor(id="source_executor")
    target1 = MockExecutor(id="target_executor_1")
    target2 = MockExecutor(id="target_executor_2")

    edge_group = SwitchCaseEdgeGroup(
        source_id=source.id,
        cases=[
            SwitchCaseEdgeGroupCase(condition=lambda x: x.data < 0, target_id=target1.id),
            SwitchCaseEdgeGroupDefault(target_id=target2.id),
        ],
    )

    executors: dict[str, Executor] = {source.id: source, target1.id: target1, target2.id: target2}
    edge_runner = create_edge_runner(edge_group, executors)

    shared_state = SharedState()
    ctx = InProcRunnerContext()

    data = "invalid_data"
    message = Message(data=data, source_id=source.id)

    success = await edge_runner.send_message(message, shared_state, ctx)
    assert success is False


# endregion SwitchCaseEdgeGroup
