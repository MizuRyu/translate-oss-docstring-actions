# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

from ..observability import EdgeGroupDeliveryStatus, OtelAttr, create_edge_group_processing_span
from ._edge import Edge, EdgeGroup, FanInEdgeGroup, FanOutEdgeGroup, SingleEdgeGroup, SwitchCaseEdgeGroup
from ._executor import Executor
from ._runner_context import Message, RunnerContext
from ._shared_state import SharedState

logger = logging.getLogger(__name__)


class EdgeRunner(ABC):
    """メッセージ配信を処理するエッジランナーの抽象基底クラス。"""

    def __init__(self, edge_group: EdgeGroup, executors: dict[str, Executor]) -> None:
        """エッジグループとexecutorマップでエッジランナーを初期化します。

        Args:
            edge_group: 実行するエッジグループ。
            executors: executor IDからexecutorインスタンスへのマップ。

        """
        self._edge_group = edge_group
        self._executors = executors

    @abstractmethod
    async def send_message(self, message: Message, shared_state: SharedState, ctx: RunnerContext) -> bool:
        """エッジグループを通じてメッセージを送信します。

        Args:
            message: 送信するメッセージ。
            shared_state: データ保持に使用する共有状態。
            ctx: ランナーのコンテキスト。

        Returns:
            bool: メッセージが正常に処理された場合はTrue、
                対象のexecutorがメッセージを処理できない場合はFalse。

        """
        raise NotImplementedError

    def _can_handle(self, executor_id: str, message_data: Any) -> bool:
        """executorが指定されたメッセージデータを処理できるかどうかをチェックします。"""
        if executor_id not in self._executors:
            return False
        return self._executors[executor_id].can_handle(message_data)

    async def _execute_on_target(
        self,
        target_id: str,
        source_ids: list[str],
        message: Message,
        shared_state: SharedState,
        ctx: RunnerContext,
    ) -> None:
        """トレースコンテキストを用いてターゲットexecutorでメッセージを実行します。"""
        if target_id not in self._executors:
            raise RuntimeError(f"Target executor {target_id} not found.")

        target_executor = self._executors[target_id]

        # トレースコンテキストパラメータを用いて実行します。
        await target_executor.execute(
            message.data,
            source_ids,  # source_executor_ids
            shared_state,  # shared_state
            ctx,  # runner_context
            trace_contexts=message.trace_contexts,  # Pass trace contexts
            source_span_ids=message.source_span_ids,  # Pass source span IDs for linking
        )


class SingleEdgeRunner(EdgeRunner):
    """単一のエッジグループ用のランナー。"""

    def __init__(self, edge_group: SingleEdgeGroup, executors: dict[str, Executor]) -> None:
        super().__init__(edge_group, executors)
        self._edge = edge_group.edges[0]

    async def send_message(self, message: Message, shared_state: SharedState, ctx: RunnerContext) -> bool:
        """単一のエッジを通じてメッセージを送信します。"""
        should_execute = False
        target_id = None
        source_id = None
        with create_edge_group_processing_span(
            self._edge_group.__class__.__name__,
            edge_group_id=self._edge_group.id,
            message_source_id=message.source_id,
            message_target_id=message.target_id,
            source_trace_contexts=message.trace_contexts,
            source_span_ids=message.source_span_ids,
        ) as span:
            try:
                if message.target_id and message.target_id != self._edge.target_id:
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: False,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TARGET_MISMATCH.value,
                    })
                    return False

                if self._can_handle(self._edge.target_id, message.data):
                    if self._edge.should_route(message.data):
                        span.set_attributes({
                            OtelAttr.EDGE_GROUP_DELIVERED: True,
                            OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DELIVERED.value,
                        })
                        should_execute = True
                        target_id = self._edge.target_id
                        source_id = self._edge.source_id
                    else:
                        span.set_attributes({
                            OtelAttr.EDGE_GROUP_DELIVERED: False,
                            OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_CONDITION_FALSE.value,
                        })
                        # ここでTrueを返すのは、メッセージは処理されたが条件が失敗したためです。
                        return True
                else:
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: False,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH.value,
                    })
                    return False
            except Exception as e:
                span.set_attributes({
                    OtelAttr.EDGE_GROUP_DELIVERED: False,
                    OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.EXCEPTION.value,
                })
                raise e

        # スパンの外で実行します。
        if should_execute and target_id and source_id:
            await self._execute_on_target(target_id, [source_id], message, shared_state, ctx)
            return True

        return False


class FanOutEdgeRunner(EdgeRunner):
    """ファンアウトエッジグループ用のランナー。"""

    def __init__(self, edge_group: FanOutEdgeGroup, executors: dict[str, Executor]) -> None:
        super().__init__(edge_group, executors)
        self._edges = edge_group.edges
        self._target_ids = edge_group.target_executor_ids
        self._target_map = {edge.target_id: edge for edge in self._edges}
        self._selection_func = cast(
            Callable[[Any, list[str]], list[str]] | None, getattr(edge_group, "selection_func", None)
        )

    async def send_message(self, message: Message, shared_state: SharedState, ctx: RunnerContext) -> bool:
        """ファンアウトエッジグループ内のすべてのエッジを通じてメッセージを送信します。"""
        deliverable_edges = []
        single_target_edge = None
        # スパン内でルーティングロジックを処理します。
        with create_edge_group_processing_span(
            self._edge_group.__class__.__name__,
            edge_group_id=self._edge_group.id,
            message_source_id=message.source_id,
            message_target_id=message.target_id,
            source_trace_contexts=message.trace_contexts,
            source_span_ids=message.source_span_ids,
        ) as span:
            try:
                selection_results = (
                    self._selection_func(message.data, self._target_ids) if self._selection_func else self._target_ids
                )
                if not self._validate_selection_result(selection_results):
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: False,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.EXCEPTION.value,
                    })
                    raise RuntimeError(
                        f"Invalid selection result: {selection_results}. "
                        f"Expected selections to be a subset of valid target executor IDs: {self._target_ids}."
                    )

                if message.target_id:
                    # ターゲットIDが指定され、選択結果に含まれている場合、そのエッジにメッセージを送信します。
                    if message.target_id in selection_results:
                        edge = self._target_map.get(message.target_id)
                        if edge and self._can_handle(edge.target_id, message.data):
                            if edge.should_route(message.data):
                                span.set_attributes({
                                    OtelAttr.EDGE_GROUP_DELIVERED: True,
                                    OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DELIVERED.value,
                                })
                                single_target_edge = edge
                            else:
                                span.set_attributes({
                                    OtelAttr.EDGE_GROUP_DELIVERED: False,
                                    OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_CONDITION_FALSE.value,  # noqa: E501
                                })
                                # 条件失敗のターゲットメッセージの場合、Trueを返します（メッセージは処理済み）。
                                return True
                        else:
                            span.set_attributes({
                                OtelAttr.EDGE_GROUP_DELIVERED: False,
                                OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH.value,  # noqa: E501
                            })
                            # 処理できないターゲットメッセージの場合、Falseを返します。
                            return False
                    else:
                        span.set_attributes({
                            OtelAttr.EDGE_GROUP_DELIVERED: False,
                            OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TARGET_MISMATCH.value,
                        })
                        # 選択に含まれないターゲットメッセージの場合、Falseを返します。
                        return False
                else:
                    # ターゲットIDがない場合、選択されたターゲットにメッセージを送信します。
                    for target_id in selection_results:
                        edge = self._target_map[target_id]
                        if self._can_handle(edge.target_id, message.data) and edge.should_route(message.data):
                            deliverable_edges.append(edge)

                    if len(deliverable_edges) > 0:
                        span.set_attributes({
                            OtelAttr.EDGE_GROUP_DELIVERED: True,
                            OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DELIVERED.value,
                        })
                    else:
                        span.set_attributes({
                            OtelAttr.EDGE_GROUP_DELIVERED: False,
                            OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH.value,
                        })

            except Exception as e:
                span.set_attributes({
                    OtelAttr.EDGE_GROUP_DELIVERED: False,
                    OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.EXCEPTION.value,
                })
                raise e

        # スパンの外で実行します。
        if single_target_edge:
            await self._execute_on_target(
                single_target_edge.target_id, [single_target_edge.source_id], message, shared_state, ctx
            )
            return True

        if deliverable_edges:

            async def send_to_edge(edge: Edge) -> bool:
                await self._execute_on_target(edge.target_id, [edge.source_id], message, shared_state, ctx)
                return True

            tasks = [send_to_edge(edge) for edge in deliverable_edges]
            results = await asyncio.gather(*tasks)
            return any(results)

        # ここに到達した場合は、配信可能なエッジがないブロードキャストメッセージです。
        return False

    def _validate_selection_result(self, selection_results: list[str]) -> bool:
        """選択結果を検証し、すべてのIDが有効なターゲットexecutor IDであることを確認します。"""
        return all(result in self._target_ids for result in selection_results)


class FanInEdgeRunner(EdgeRunner):
    """ファンインエッジグループ用のランナー。"""

    def __init__(self, edge_group: FanInEdgeGroup, executors: dict[str, Executor]) -> None:
        super().__init__(edge_group, executors)
        self._edges = edge_group.edges
        # ターゲットexecutorに送信する前にメッセージを保持するバッファ キーは送信元executor ID、値はメッセージのリスト。
        self._buffer: dict[str, list[Message]] = defaultdict(list)

    async def send_message(self, message: Message, shared_state: SharedState, ctx: RunnerContext) -> bool:
        """ファンインエッジグループ内のすべてのエッジを通じてメッセージを送信します。"""
        execution_data: dict[str, Any] | None = None
        with create_edge_group_processing_span(
            self._edge_group.__class__.__name__,
            edge_group_id=self._edge_group.id,
            message_source_id=message.source_id,
            message_target_id=message.target_id,
            source_trace_contexts=message.trace_contexts,
            source_span_ids=message.source_span_ids,
        ) as span:
            try:
                if message.target_id and message.target_id != self._edges[0].target_id:
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: False,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TARGET_MISMATCH.value,
                    })
                    return False

                # ターゲットがメッセージデータのリストを処理できるかチェックします（ファンインは複数メッセージを集約）。
                if self._can_handle(self._edges[0].target_id, [message.data]):
                    # エッジがデータを処理できる場合、メッセージをバッファに格納します。
                    self._buffer[message.source_id].append(message)
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: True,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.BUFFERED.value,
                    })
                else:
                    # エッジがデータを処理できない場合、Falseを返します。
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: False,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH.value,
                    })
                    return False

                if self._is_ready_to_send():
                    # グループ内のすべてのエッジにデータがある場合、実行準備をします。
                    messages_to_send = [msg for edge in self._edges for msg in self._buffer[edge.source_id]]
                    self._buffer.clear()
                    # 集約されたデータをターゲットに送信します。
                    aggregated_data = [msg.data for msg in messages_to_send]

                    # ファンインリンク用にすべてのトレースコンテキストと送信元スパンIDを収集します。
                    trace_contexts = [msg.trace_context for msg in messages_to_send if msg.trace_context]
                    source_span_ids = [msg.source_span_id for msg in messages_to_send if msg.source_span_id]

                    # 集約データ用の新しいMessageオブジェクトを作成します。
                    aggregated_message = Message(
                        data=aggregated_data,
                        source_id=self._edge_group.__class__.__name__,  # This won't be used in self._execute_on_target.
                        trace_contexts=trace_contexts,
                        source_span_ids=source_span_ids,
                    )
                    span.set_attributes({
                        OtelAttr.EDGE_GROUP_DELIVERED: True,
                        OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.DELIVERED.value,
                    })

                    # 後で使用するために実行データを保存します。
                    execution_data = {
                        "target_id": self._edges[0].target_id,
                        "source_ids": [edge.source_id for edge in self._edges],
                        "message": aggregated_message,
                    }

            except Exception as e:
                span.set_attributes({
                    OtelAttr.EDGE_GROUP_DELIVERED: False,
                    OtelAttr.EDGE_GROUP_DELIVERY_STATUS: EdgeGroupDeliveryStatus.EXCEPTION.value,
                })
                raise e

        # 必要に応じてスパンの外で実行します。
        if execution_data:
            await self._execute_on_target(
                execution_data["target_id"], execution_data["source_ids"], execution_data["message"], shared_state, ctx
            )
            return True

        return True  # バッファリングされたメッセージの場合はTrueを返します（さらに待機中）。

    def _is_ready_to_send(self) -> bool:
        """グループ内のすべてのエッジに送信データがあるかチェックします。"""
        return all(self._buffer[edge.source_id] for edge in self._edges)


class SwitchCaseEdgeRunner(FanOutEdgeRunner):
    """スイッチケースエッジグループ用のランナー（FanOutEdgeRunnerを継承）。"""

    def __init__(self, edge_group: SwitchCaseEdgeGroup, executors: dict[str, Executor]) -> None:
        super().__init__(edge_group, executors)


def create_edge_runner(edge_group: EdgeGroup, executors: dict[str, Executor]) -> EdgeRunner:
    """エッジグループに適したエッジランナーを作成するファクトリ関数。

    Args:
        edge_group: ランナーを作成するエッジグループ。
        executors: executor IDからexecutorインスタンスへのマップ。

    Returns:
        適切なEdgeRunnerインスタンス。

    """
    if isinstance(edge_group, SingleEdgeGroup):
        return SingleEdgeRunner(edge_group, executors)
    if isinstance(edge_group, SwitchCaseEdgeGroup):
        return SwitchCaseEdgeRunner(edge_group, executors)
    if isinstance(edge_group, FanOutEdgeGroup):
        return FanOutEdgeRunner(edge_group, executors)
    if isinstance(edge_group, FanInEdgeGroup):
        return FanInEdgeRunner(edge_group, executors)
    raise ValueError(f"Unsupported edge group type: {type(edge_group)}")
