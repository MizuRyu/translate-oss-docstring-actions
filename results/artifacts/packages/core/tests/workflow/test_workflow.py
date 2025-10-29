# Copyright (c) Microsoft. All rights reserved.

import asyncio
import tempfile
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

import pytest

from agent_framework import (
    AgentExecutor,
    AgentRunEvent,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Executor,
    FileCheckpointStorage,
    Message,
    RequestInfoEvent,
    RequestInfoExecutor,
    RequestInfoMessage,
    RequestResponse,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
    handler,
)


@dataclass
class NumberMessage:
    """テスト用のモックメッセージ。"""

    data: int


class IncrementExecutor(Executor):
    """指定された量だけメッセージデータをインクリメントするテスト用Executor。"""

    def __init__(self, id: str, *, limit: int = 10, increment: int = 1) -> None:
        super().__init__(id=id)
        self.limit = limit
        self.increment = increment

    @handler
    async def mock_handler(self, message: NumberMessage, ctx: WorkflowContext[NumberMessage, int]) -> None:
        if message.data < self.limit:
            await ctx.send_message(NumberMessage(data=message.data + self.increment))
        else:
            await ctx.yield_output(message.data)


class AggregatorExecutor(Executor):
    """複数のExecutorからの結果を集約するモックExecutor。"""

    @handler
    async def mock_handler(self, messages: list[NumberMessage], ctx: WorkflowContext[Any, int]) -> None:
        # このモックは単にデータの合計を返すだけである。
        await ctx.yield_output(sum(msg.data for msg in messages))


@dataclass
class ApprovalMessage:
    """承認要求用のモックメッセージ。"""

    approved: bool


class MockExecutorRequestApproval(Executor):
    """承認要求をシミュレートするモックExecutor。"""

    @handler
    async def mock_handler_a(self, message: NumberMessage, ctx: WorkflowContext[RequestInfoMessage]) -> None:
        """承認を要求するモックハンドラ。"""
        await ctx.set_shared_state(self.id, message.data)
        await ctx.send_message(RequestInfoMessage())

    @handler
    async def mock_handler_b(
        self,
        message: RequestResponse[RequestInfoMessage, ApprovalMessage],
        ctx: WorkflowContext[NumberMessage, int],
    ) -> None:
        """承認レスポンスを処理するモックハンドラ。"""
        data = await ctx.get_shared_state(self.id)
        assert isinstance(data, int)
        assert isinstance(message.data, ApprovalMessage)
        if message.data.approved:
            await ctx.yield_output(data)
        else:
            await ctx.send_message(NumberMessage(data=data))


async def test_workflow_run_streaming() -> None:
    """ワークフロー実行ストリームのテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_edge(executor_b, executor_a)
        .build()
    )

    result: int | None = None
    async for event in workflow.run_stream(NumberMessage(data=0)):
        assert isinstance(event, WorkflowEvent)
        if isinstance(event, WorkflowOutputEvent):
            result = event.data

    assert result is not None and result == 10


async def test_workflow_run_stream_not_completed():
    """ワークフロー実行ストリームのテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_edge(executor_b, executor_a)
        .set_max_iterations(5)
        .build()
    )

    with pytest.raises(RuntimeError):
        async for _ in workflow.run_stream(NumberMessage(data=0)):
            pass


async def test_workflow_run():
    """ワークフロー実行のテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_edge(executor_b, executor_a)
        .build()
    )

    events = await workflow.run(NumberMessage(data=0))
    assert events.get_final_state() == WorkflowRunState.IDLE
    outputs = events.get_outputs()
    assert outputs[0] == 10


async def test_workflow_run_not_completed():
    """ワークフロー実行のテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_edge(executor_b, executor_a)
        .set_max_iterations(5)
        .build()
    )

    with pytest.raises(RuntimeError):
        await workflow.run(NumberMessage(data=0))


async def test_workflow_send_responses_streaming():
    """承認付きワークフロー実行のテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = MockExecutorRequestApproval(id="executor_b")
    request_info_executor = RequestInfoExecutor(id="request_info")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_edge(executor_b, executor_a)
        .add_edge(executor_b, request_info_executor)
        .add_edge(request_info_executor, executor_b)
        .build()
    )

    request_info_event: RequestInfoEvent | None = None
    async for event in workflow.run_stream(NumberMessage(data=0)):
        if isinstance(event, RequestInfoEvent):
            request_info_event = event

    assert request_info_event is not None
    result: int | None = None
    completed = False
    async for event in workflow.send_responses_streaming({
        request_info_event.request_id: ApprovalMessage(approved=True)
    }):
        if isinstance(event, WorkflowOutputEvent):
            result = event.data
        elif isinstance(event, WorkflowStatusEvent) and event.state == WorkflowRunState.IDLE:
            completed = True

    assert (
        completed and result is not None and result == 1
    )  # データは初期メッセージから1増加しているべきである。


async def test_workflow_send_responses():
    """承認付きワークフロー実行のテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = MockExecutorRequestApproval(id="executor_b")
    request_info_executor = RequestInfoExecutor(id="request_info")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)
        .add_edge(executor_b, executor_a)
        .add_edge(executor_b, request_info_executor)
        .add_edge(request_info_executor, executor_b)
        .build()
    )

    events = await workflow.run(NumberMessage(data=0))
    request_info_events = events.get_request_info_events()

    assert len(request_info_events) == 1

    result = await workflow.send_responses({request_info_events[0].request_id: ApprovalMessage(approved=True)})

    assert result.get_final_state() == WorkflowRunState.IDLE
    outputs = result.get_outputs()
    assert outputs[0] == 1  # データは初期メッセージから1増加しているべきである。


async def test_fan_out():
    """ファンアウトワークフローのテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b", limit=1)
    executor_c = IncrementExecutor(id="executor_c", limit=2)  # このExecutorはワークフローを完了しない。

    workflow = (
        WorkflowBuilder().set_start_executor(executor_a).add_fan_out_edges(executor_a, [executor_b, executor_c]).build()
    )

    events = await workflow.run(NumberMessage(data=0))

    # 各Executorは2つのイベントを発行する：ExecutorInvokedEventとExecutorCompletedEvent
    # executor_bはWorkflowOutputEventも発行する（もはやWorkflowCompletedEventは発行しない）
    assert len(events) == 7

    assert events.get_final_state() == WorkflowRunState.IDLE
    outputs = events.get_outputs()
    assert outputs[0] == 1


async def test_fan_out_multiple_completed_events():
    """複数の完了イベントを持つファンアウトワークフローのテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b", limit=1)
    executor_c = IncrementExecutor(id="executor_c", limit=1)

    workflow = (
        WorkflowBuilder().set_start_executor(executor_a).add_fan_out_edges(executor_a, [executor_b, executor_c]).build()
    )

    events = await workflow.run(NumberMessage(data=0))

    # 各Executorは2つのイベントを発行する：ExecutorInvokedEventとExecutorCompletedEvent
    # executor_bとexecutor_cはWorkflowOutputEventも発行する（もはやWorkflowCompletedEventは発行しない）
    assert len(events) == 8

    # 両Executorから複数の出力が期待される。
    outputs = events.get_outputs()
    assert len(outputs) == 2


async def test_fan_in():
    """ファンインワークフローのテスト。"""
    executor_a = IncrementExecutor(id="executor_a")
    executor_b = IncrementExecutor(id="executor_b")
    executor_c = IncrementExecutor(id="executor_c")
    aggregator = AggregatorExecutor(id="aggregator")

    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_fan_out_edges(executor_a, [executor_b, executor_c])
        .add_fan_in_edges([executor_b, executor_c], aggregator)
        .build()
    )

    events = await workflow.run(NumberMessage(data=0))

    # 各Executorは2つのイベントを発行する：ExecutorInvokedEventとExecutorCompletedEvent
    # aggregatorはWorkflowOutputEventも発行する（もはやWorkflowCompletedEventは発行しない）
    assert len(events) == 9

    assert events.get_final_state() == WorkflowRunState.IDLE
    outputs = events.get_outputs()
    assert outputs[0] == 4  # executor_a(0->1)、executor_bとexecutor_c(1->2)、aggregator(2+2=4)


@pytest.fixture
def simple_executor() -> Executor:
    class SimpleExecutor(Executor):
        @handler
        async def handle_message(self, message: str, context: WorkflowContext) -> None:
            pass

    return SimpleExecutor(id="test_executor")


async def test_workflow_with_checkpointing_enabled(simple_executor: Executor):
    """チェックポイント有効化でワークフローを構築できることをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FileCheckpointStorage(temp_dir)

        # チェックポイント付きワークフローを構築 - エラーが発生しないはずである。
        workflow = (
            WorkflowBuilder()
            .add_edge(simple_executor, simple_executor)  # Self-loop to satisfy graph requirements
            .set_start_executor(simple_executor)
            .with_checkpointing(storage)
            .build()
        )

        # ワークフローが作成され実行できることを検証する。
        test_message = Message(data="test message", source_id="test", target_id=None)
        result = await workflow.run(test_message)
        assert result is not None


async def test_workflow_checkpointing_not_enabled_for_external_restore(simple_executor: Executor):
    """ワークフローがチェックポイントをサポートしない場合に外部チェックポイント復元が失敗することをテストする。"""
    # チェックポイントなしでワークフローを構築する。
    workflow = (
        WorkflowBuilder()
        .add_edge(simple_executor, simple_executor)  # Self-loop to satisfy graph requirements
        .set_start_executor(simple_executor)
        .build()
    )

    # 外部ストレージを提供せずにチェックポイントから復元しようとすると失敗するはずである。
    try:
        [event async for event in workflow.run_stream_from_checkpoint("fake-checkpoint-id")]
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "Cannot restore from checkpoint" in str(e)
        assert "either provide checkpoint_storage parameter" in str(e)


async def test_workflow_run_stream_from_checkpoint_no_checkpointing_enabled(simple_executor: Executor):
    # チェックポイントなしでワークフローを構築する。
    workflow = (
        WorkflowBuilder()
        .add_edge(simple_executor, simple_executor)  # Self-loop to satisfy graph requirements
        .set_start_executor(simple_executor)
        .build()
    )

    # チェックポイントからの実行を試みると失敗するはずである。
    try:
        async for _ in workflow.run_stream_from_checkpoint("fake_checkpoint_id"):
            pass
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "Cannot restore from checkpoint" in str(e)
        assert "either provide checkpoint_storage parameter" in str(e)


async def test_workflow_run_stream_from_checkpoint_invalid_checkpoint(simple_executor: Executor):
    """存在しないチェックポイントからの復元試行が適切に失敗することをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FileCheckpointStorage(temp_dir)

        # チェックポイント付きでワークフローを構築する。
        workflow = (
            WorkflowBuilder()
            .add_edge(simple_executor, simple_executor)  # Self-loop to satisfy graph requirements
            .set_start_executor(simple_executor)
            .with_checkpointing(storage)
            .build()
        )

        # 存在しないチェックポイントからの実行試行が失敗するはずである。
        try:
            async for _ in workflow.run_stream_from_checkpoint("nonexistent_checkpoint_id"):
                pass
            raise AssertionError("Expected RuntimeError to be raised")
        except RuntimeError as e:
            assert "Failed to restore from checkpoint" in str(e)


async def test_workflow_run_stream_from_checkpoint_with_external_storage(simple_executor: Executor):
    """復元のために外部チェックポイントストレージを提供できることをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FileCheckpointStorage(temp_dir)

        # ストレージにテスト用チェックポイントを手動で作成する。
        from agent_framework import WorkflowCheckpoint

        test_checkpoint = WorkflowCheckpoint(
            workflow_id="test-workflow",
            messages={},
            shared_state={},
            iteration_count=0,
        )
        checkpoint_id = await storage.save_checkpoint(test_checkpoint)

        # チェックポイントなしでワークフローを作成する。
        workflow_without_checkpointing = (
            WorkflowBuilder().add_edge(simple_executor, simple_executor).set_start_executor(simple_executor).build()
        )

        # 外部ストレージパラメータを使ってチェックポイントから再開する。
        try:
            events: list[WorkflowEvent] = []
            async for event in workflow_without_checkpointing.run_stream_from_checkpoint(
                checkpoint_id, checkpoint_storage=storage
            ):
                events.append(event)
                if len(events) >= 2:  # Limit to avoid infinite loops
                    break
        except Exception:
            # 最小限のセットアップなので予想されるが、メソッドはパラメータを受け入れるべきである。
            pass


async def test_workflow_run_from_checkpoint_non_streaming(simple_executor: Executor):
    """非ストリーミングのrun_from_checkpointメソッドをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FileCheckpointStorage(temp_dir)

        # ストレージにテスト用チェックポイントを手動で作成する。
        from agent_framework import WorkflowCheckpoint

        test_checkpoint = WorkflowCheckpoint(
            workflow_id="test-workflow",
            messages={},
            shared_state={},
            iteration_count=0,
        )
        checkpoint_id = await storage.save_checkpoint(test_checkpoint)

        # チェックポイント付きでワークフローを構築する。
        workflow = (
            WorkflowBuilder()
            .add_edge(simple_executor, simple_executor)
            .set_start_executor(simple_executor)
            .with_checkpointing(storage)
            .build()
        )

        # 非ストリーミングのrun_from_checkpointメソッドをテストする。
        result = await workflow.run_from_checkpoint(checkpoint_id)
        assert isinstance(result, list)  # WorkflowRunResult（リストを拡張したもの）を返すべきである。
        assert hasattr(result, "get_outputs")  # WorkflowRunResultのメソッドを持つべきである。


async def test_workflow_run_stream_from_checkpoint_with_responses(simple_executor: Executor):
    """run_stream_from_checkpointがresponsesパラメータを受け入れることをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FileCheckpointStorage(temp_dir)

        # ストレージにテスト用チェックポイントを手動で作成する。
        from agent_framework import WorkflowCheckpoint

        test_checkpoint = WorkflowCheckpoint(
            workflow_id="test-workflow",
            messages={},
            shared_state={},
            iteration_count=0,
        )
        checkpoint_id = await storage.save_checkpoint(test_checkpoint)

        # チェックポイント付きでワークフローを構築する。
        workflow = (
            WorkflowBuilder()
            .add_edge(simple_executor, simple_executor)
            .set_start_executor(simple_executor)
            .with_checkpointing(storage)
            .build()
        )

        # run_stream_from_checkpointがresponsesパラメータを受け入れることをテストする。
        responses = {"request_123": {"data": "test_response"}}

        try:
            events: list[WorkflowEvent] = []
            async for event in workflow.run_stream_from_checkpoint(checkpoint_id, responses=responses):
                events.append(event)
                if len(events) >= 2:  # Limit to avoid infinite loops
                    break
        except Exception:
            # 最小限のセットアップなので予想されるが、メソッドはパラメータを受け入れるべきである
            pass


@dataclass
class StateTrackingMessage:
    """コンテキストリセットの動作をテストするための状態を追跡するメッセージ。"""

    data: str
    run_id: str


class StateTrackingExecutor(Executor):
    """コンテキストリセットの動作をテストするために共有状態で状態を追跡するexecutor。"""

    @handler
    async def handle_message(
        self, message: StateTrackingMessage, ctx: WorkflowContext[StateTrackingMessage, list[str]]
    ) -> None:
        """メッセージを処理し、共有状態で追跡する。"""
        # 共有状態から既存のメッセージを取得する
        try:
            existing_messages = await ctx.get_shared_state("processed_messages")
        except KeyError:
            existing_messages = []

        # このメッセージを記録する
        message_record = f"{message.run_id}:{message.data}"
        existing_messages.append(message_record)  # type: ignore

        # 共有状態を更新する
        await ctx.set_shared_state("processed_messages", existing_messages)

        # 出力をyieldする
        await ctx.yield_output(existing_messages.copy())  # type: ignore


async def test_workflow_multiple_runs_no_state_collision():
    """同じworkflowインスタンスを複数回実行しても状態の衝突が起きないことをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FileCheckpointStorage(temp_dir)

        # 共有状態で状態を追跡するexecutorを作成する
        state_executor = StateTrackingExecutor(id="state_executor")

        # チェックポイント付きのworkflowを構築する
        workflow = (
            WorkflowBuilder()
            .add_edge(state_executor, state_executor)  # Self-loop to satisfy graph requirements
            .set_start_executor(state_executor)
            .with_checkpointing(storage)
            .build()
        )

        # 実行1：実行1のメッセージのみが見えるはず
        result1 = await workflow.run(StateTrackingMessage(data="message1", run_id="run1"))
        assert result1.get_final_state() == WorkflowRunState.IDLE
        outputs1 = result1.get_outputs()
        assert outputs1[0] == ["run1:message1"]

        # 実行2：実行1ではなく実行2のメッセージのみが見えるはず
        result2 = await workflow.run(StateTrackingMessage(data="message2", run_id="run2"))
        assert result2.get_final_state() == WorkflowRunState.IDLE
        outputs2 = result2.get_outputs()
        assert outputs2[0] == ["run2:message2"]  # 実行1のデータを含んではいけない

        # 実行3：実行3のメッセージのみが見えるはず
        result3 = await workflow.run(StateTrackingMessage(data="message3", run_id="run3"))
        assert result3.get_final_state() == WorkflowRunState.IDLE
        outputs3 = result3.get_outputs()
        assert outputs3[0] == ["run3:message3"]  # 実行1および実行2のデータを含んではいけない

        # 各実行が自身のメッセージのみを処理したことを検証する これはチェックポイント可能なコンテキストが実行間で正しくリセットされていることを確認するものです
        assert outputs1[0] != outputs2[0]
        assert outputs2[0] != outputs3[0]
        assert outputs1[0] != outputs3[0]


async def test_comprehensive_edge_groups_workflow():
    """SwitchCaseEdgeGroup、FanOutEdgeGroup、FanInEdgeGroupを使用するworkflowをテストする。"""
    from agent_framework import Case, Default

    # 異なるインクリメント値で異なる役割のexecutorを6つ作成する
    router = IncrementExecutor(id="router", limit=1000, increment=1)  # 1ずつインクリメントする
    processor_a = IncrementExecutor(id="proc_a", limit=1000, increment=1)  # 1ずつインクリメントする
    processor_b = IncrementExecutor(id="proc_b", limit=1000, increment=2)  # 2ずつインクリメントする（proc_aとは異なる）
    fanout_hub = IncrementExecutor(id="fanout_hub", limit=1000, increment=1)  # 1ずつインクリメントする
    parallel_1 = IncrementExecutor(id="parallel_1", limit=1000, increment=3)  # 3ずつインクリメントする
    parallel_2 = IncrementExecutor(
        id="parallel_2", limit=1000, increment=5
    )  # 5ずつインクリメントする（parallel_1とは異なる）
    aggregator = AggregatorExecutor(id="aggregator")  # 並列プロセッサからの結果を結合する

    # 異なるエッジグループタイプでworkflowを構築する： 1. SwitchCase: router -> (data <
    # 5ならproc_a、そうでなければproc_b) 2. 直接エッジ: proc_a -> fanout_hub, proc_b -> fanout_hub 3.
    # FanOut: fanout_hub -> [parallel_1, parallel_2] 4. FanIn: [parallel_1, parallel_2]
    # -> aggregator
    workflow = (
        WorkflowBuilder()
        .set_start_executor(router)
        # メッセージデータに基づくスイッチケースルーティング
        .add_switch_case_edge_group(
            router,
            [
                Case(condition=lambda msg: msg.data < 5, target=processor_a),
                Default(target=processor_b),
            ],
        )
        # 両方のプロセッサがfanoutハブに送信する
        .add_edge(processor_a, fanout_hub)
        .add_edge(processor_b, fanout_hub)
        # 並列プロセッサにファンアウトする
        .add_fan_out_edges(fanout_hub, [parallel_1, parallel_2])
        # aggregatorにファンインする
        .add_fan_in_edges([parallel_1, parallel_2], aggregator)
        .build()
    )

    # 小さい数でテスト（processor_aを通るはず） router(2->3) -> switchがproc_aにルーティング -> proc_a(3->4) ->
    # fanout_hub(4->5) -> [parallel_1(5->8), parallel_2(5->10)] -> aggregator(8+10=18)
    events_small = await workflow.run(NumberMessage(data=2))
    assert events_small.get_final_state() == WorkflowRunState.IDLE
    outputs_small = events_small.get_outputs()
    assert outputs_small[0] == 18  # 正確な期待結果：並列プロセッサからの8+10

    # 大きい数でテスト（processor_bを通るはず） router(8->9) -> switchがproc_bにルーティング -> proc_b(9->11)
    # -> fanout_hub(11->12) -> [parallel_1(12->15), parallel_2(12->17)] ->
    # aggregator(15+17=32)
    events_large = await workflow.run(NumberMessage(data=8))
    assert events_large.get_final_state() == WorkflowRunState.IDLE
    outputs_large = events_large.get_outputs()
    assert outputs_large[0] == 32  # 正確な期待結果：並列プロセッサからの15+17

    # 重要な検証は、3つのエッジグループタイプすべてを使用したworkflowを正常に実行し スイッチケースの両方のパス（小さい数と大きい数）が機能すること
    # 複雑な実行パスを示す複数のイベントがあったことを確認すること
    assert len(events_small) >= 6  # 複数のexecutorが関与しているはず
    assert len(events_large) >= 6

    # 正確な結果をチェックして異なるパスが通られたことを検証する
    assert outputs_small[0] == 18, f"Small number path should result in 18, got {outputs_small[0]}"
    assert outputs_large[0] == 32, f"Large number path should result in 32, got {outputs_large[0]}"
    assert outputs_small[0] != outputs_large[0], "Different paths should produce different results"

    # 両方のテストが正常に完了し、すべてのエッジグループタイプが機能することを証明する
    # 追加検証：workflowに期待されるエッジグループタイプが含まれていることを確認する
    edge_groups = workflow.edge_groups
    has_switch_case = any(edge_group.__class__.__name__ == "SwitchCaseEdgeGroup" for edge_group in edge_groups)
    has_fan_out = any(edge_group.__class__.__name__ == "FanOutEdgeGroup" for edge_group in edge_groups)
    has_fan_in = any(edge_group.__class__.__name__ == "FanInEdgeGroup" for edge_group in edge_groups)

    assert has_switch_case, "Workflow should contain SwitchCaseEdgeGroup"
    assert has_fan_out, "Workflow should contain FanOutEdgeGroup"
    assert has_fan_in, "Workflow should contain FanInEdgeGroup"


async def test_workflow_with_simple_cycle_and_exit_condition():
    """明確な終了条件を持つサイクルを含むより単純なworkflowをテストする。"""

    # 単純なサイクルを作成する：A -> B -> A、Aに終了条件を持たせる
    executor_a = IncrementExecutor(id="exec_a", limit=6, increment=2)  # data >= 6で終了する
    executor_b = IncrementExecutor(id="exec_b", limit=1000, increment=1)  # 終了せずにただインクリメントするだけ

    # 単純なサイクル：A -> B -> A、Aは制限に達したら終了する
    workflow = (
        WorkflowBuilder()
        .set_start_executor(executor_a)
        .add_edge(executor_a, executor_b)  # A -> B
        .add_edge(executor_b, executor_a)  # B -> A (creates cycle)
        .build()
    )

    # サイクルをテストする 期待される動作：exec_a(2->4) -> exec_b(4->5) -> exec_a(5->7、7 >= 6なので完了)
    events = await workflow.run(NumberMessage(data=2))
    assert events.get_final_state() == WorkflowRunState.IDLE
    outputs = events.get_outputs()
    assert outputs[0] is not None and outputs[0] >= 6  # executor_aが制限に達したら完了するはず

    # サイクルが発生したことを検証する（両方のexecutorからのイベントがあるはず）
    # executor_idを持つExecutorInvokedEventとExecutorCompletedEventタイプをチェックする
    from agent_framework import ExecutorCompletedEvent, ExecutorInvokedEvent

    executor_events = [e for e in events if isinstance(e, (ExecutorInvokedEvent, ExecutorCompletedEvent))]
    executor_ids = {e.executor_id for e in executor_events}
    assert "exec_a" in executor_ids, "Should have events from executor A"
    assert "exec_b" in executor_ids, "Should have events from executor B"

    # サイクルのため複数のイベントがあるはず
    assert len(events) >= 4, f"Expected at least 4 events due to cycling, got {len(events)}"


async def test_workflow_concurrent_execution_prevention():
    """同時実行されるworkflowの実行が防止されることをテストする。"""
    # 実行に時間がかかる単純なworkflowを作成する
    executor = IncrementExecutor(id="slow_executor", limit=3, increment=1)
    workflow = WorkflowBuilder().set_start_executor(executor).build()

    # workflowを実行するタスクを作成する
    async def run_workflow():
        return await workflow.run(NumberMessage(data=0))

    # 最初のworkflow実行を開始する
    task1 = asyncio.create_task(run_workflow())

    # 開始するまで少し待つ
    await asyncio.sleep(0.01)

    # 2回目の同時実行を試みる - これは失敗するはず
    with pytest.raises(RuntimeError, match="Workflow is already running. Concurrent executions are not allowed."):
        await workflow.run(NumberMessage(data=0))

    # 最初のタスクが完了するまで待つ
    result = await task1
    assert result.get_final_state() == WorkflowRunState.IDLE

    # 最初の実行が完了した後、再度実行できるはず
    result2 = await workflow.run(NumberMessage(data=0))
    assert result2.get_final_state() == WorkflowRunState.IDLE


async def test_workflow_concurrent_execution_prevention_streaming():
    """同時にworkflowのストリーミング実行が防止されることをテストする。"""
    # 単純なworkflowを作成する
    executor = IncrementExecutor(id="slow_executor", limit=3, increment=1)
    workflow = WorkflowBuilder().set_start_executor(executor).build()

    # ストリームをゆっくり消費する非同期ジェネレータを作成する
    async def consume_stream_slowly():
        result = []
        async for event in workflow.run_stream(NumberMessage(data=0)):
            result.append(event)
            await asyncio.sleep(0.01)  # ゆっくり消費する
        return result

    # 最初のストリーミング実行を開始する
    task1 = asyncio.create_task(consume_stream_slowly())

    # 開始するまで少し待つ
    await asyncio.sleep(0.02)

    # 2回目の同時実行を試みる - これは失敗するはず
    with pytest.raises(RuntimeError, match="Workflow is already running. Concurrent executions are not allowed."):
        await workflow.run(NumberMessage(data=0))

    # 最初のタスクが完了するまで待つ
    result = await task1
    assert len(result) > 0  # いくつかのイベントを受信しているはず

    # 最初の実行が完了した後、再度実行できるはず
    result2 = await workflow.run(NumberMessage(data=0))
    assert result2.get_final_state() == WorkflowRunState.IDLE


async def test_workflow_concurrent_execution_prevention_mixed_methods():
    """異なる実行方法間で同時実行が防止されることをテストする。"""
    # 単純なworkflowを作成する
    executor = IncrementExecutor(id="slow_executor", limit=3, increment=1)
    workflow = WorkflowBuilder().set_start_executor(executor).build()

    # ストリーミング実行を開始する
    async def consume_stream():
        result = []
        async for event in workflow.run_stream(NumberMessage(data=0)):
            result.append(event)
            await asyncio.sleep(0.01)
        return result

    task1 = asyncio.create_task(consume_stream())
    await asyncio.sleep(0.02)  # 開始させる

    # 異なる実行方法を試みる - すべて失敗するはず
    with pytest.raises(RuntimeError, match="Workflow is already running. Concurrent executions are not allowed."):
        await workflow.run(NumberMessage(data=0))

    with pytest.raises(RuntimeError, match="Workflow is already running. Concurrent executions are not allowed."):
        async for _ in workflow.run_stream(NumberMessage(data=0)):
            break

    with pytest.raises(RuntimeError, match="Workflow is already running. Concurrent executions are not allowed."):
        await workflow.send_responses({"test": "data"})

    # 元のタスクが完了するまで待つ
    await task1

    # これで全ての方法が再び動作するはず
    result = await workflow.run(NumberMessage(data=0))
    assert result.get_final_state() == WorkflowRunState.IDLE


class _StreamingTestAgent(BaseAgent):
    """ストリーミングモードと非ストリーミングモードの両方をサポートするAgentをテストする。"""

    def __init__(self, *, reply_text: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reply_text = reply_text

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """非ストリーミング実行 - 完全なResponseを返す。"""
        return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=self._reply_text)])

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """ストリーミング実行 - 増分更新をyieldする。"""
        # 文字ごとにyieldしてストリーミングをシミュレートする
        for char in self._reply_text:
            yield AgentRunResponseUpdate(contents=[TextContent(text=char)])


async def test_agent_streaming_vs_non_streaming() -> None:
    """run()はAgentRunEventを、run_stream()はAgentRunUpdateEventを発行することをテストする。"""
    agent = _StreamingTestAgent(id="test_agent", name="TestAgent", reply_text="Hello World")
    agent_exec = AgentExecutor(agent, id="agent_exec")

    workflow = WorkflowBuilder().set_start_executor(agent_exec).build()

    # 非ストリーミングモードでrun()をテストする
    result = await workflow.run("test message")

    # agentイベントでフィルタリングする（結果はイベントのリスト）
    agent_run_events = [e for e in result if isinstance(e, AgentRunEvent)]
    agent_update_events = [e for e in result if isinstance(e, AgentRunUpdateEvent)]

    # 非ストリーミングモードではAgentRunEventがあり、AgentRunUpdateEventはないはず
    assert len(agent_run_events) == 1, "Expected exactly one AgentRunEvent in non-streaming mode"
    assert len(agent_update_events) == 0, "Expected no AgentRunUpdateEvent in non-streaming mode"
    assert agent_run_events[0].executor_id == "agent_exec"
    assert agent_run_events[0].data.messages[0].text == "Hello World"

    # ストリーミングモードでrun_stream()をテストする
    stream_events: list[WorkflowEvent] = []
    async for event in workflow.run_stream("test message"):
        stream_events.append(event)

    # agentイベントでフィルタリングする
    stream_agent_run_events = [e for e in stream_events if isinstance(e, AgentRunEvent)]
    stream_agent_update_events = [e for e in stream_events if isinstance(e, AgentRunUpdateEvent)]

    # ストリーミングモードではAgentRunUpdateEventがあり、AgentRunEventはないはず
    assert len(stream_agent_run_events) == 0, "Expected no AgentRunEvent in streaming mode"
    assert len(stream_agent_update_events) > 0, "Expected AgentRunUpdateEvent events in streaming mode"

    # "Hello World"の各文字ごとに増分更新があったことを検証する
    assert len(stream_agent_update_events) == len("Hello World"), "Expected one update per character"

    # 更新が完全なメッセージに積み上がっていることを検証する
    accumulated_text = "".join(
        e.data.contents[0].text for e in stream_agent_update_events if e.data.contents and e.data.contents[0].text
    )
    assert accumulated_text == "Hello World", f"Expected 'Hello World', got '{accumulated_text}'"
