# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any

from ._checkpoint import CheckpointStorage, WorkflowCheckpoint
from ._checkpoint_encoding import DATACLASS_MARKER, MODEL_MARKER, decode_checkpoint_value
from ._const import EXECUTOR_STATE_KEY
from ._edge import EdgeGroup
from ._edge_runner import EdgeRunner, create_edge_runner
from ._events import WorkflowEvent
from ._executor import Executor
from ._runner_context import (
    Message,
    RunnerContext,
)
from ._shared_state import SharedState

if TYPE_CHECKING:
    from ._request_info_executor import RequestInfoExecutor

logger = logging.getLogger(__name__)


class Runner:
    """Pregelのsuperstepsでworkflowを実行するためのClass。"""

    def __init__(
        self,
        edge_groups: Sequence[EdgeGroup],
        executors: dict[str, Executor],
        shared_state: SharedState,
        ctx: RunnerContext,
        max_iterations: int = 100,
        workflow_id: str | None = None,
    ) -> None:
        """エッジ、共有State、ContextでRunnerを初期化します。

        Args:
            edge_groups: workflowのエッジグループ。
            executors: Executor IDからExecutorインスタンスへのマップ。
            shared_state: workflowの共有State。
            ctx: workflowのRunner Context。
            max_iterations: 実行する最大イテレーション数。
            workflow_id: チェックポイント用のworkflow ID。

        """
        self._executors = executors
        self._edge_runners = [create_edge_runner(group, executors) for group in edge_groups]
        self._edge_runner_map = self._parse_edge_runners(self._edge_runners)
        self._ctx = ctx
        self._iteration = 0
        self._max_iterations = max_iterations
        self._shared_state = shared_state
        self._workflow_id = workflow_id
        self._running = False
        self._resumed_from_checkpoint = False  # 再開したかどうかを追跡します
        self.graph_signature_hash: str | None = None

        # workflow IDが提供されていればContextに設定します
        if workflow_id:
            self._ctx.set_workflow_id(workflow_id)

    @property
    def context(self) -> RunnerContext:
        """workflowのContextを取得します。"""
        return self._ctx

    def reset_iteration_count(self) -> None:
        """イテレーションカウントをゼロにリセットします。"""
        self._iteration = 0

    async def run_until_convergence(self) -> AsyncGenerator[WorkflowEvent, None]:
        """メッセージが送信されなくなるまでworkflowを実行します。"""
        if self._running:
            raise RuntimeError("Runner is already running.")

        self._running = True
        try:
            # ループに入る前に既に生成されたイベントを発行します
            if await self._ctx.has_events():
                logger.info("Yielding pre-loop events")
                for event in await self._ctx.drain_events():
                    yield event

            # 初期実行からメッセージがあれば最初のチェックポイントを作成します
            if await self._ctx.has_messages() and self._ctx.has_checkpointing():
                if not self._resumed_from_checkpoint:
                    logger.info("Creating checkpoint after initial execution")
                    await self._create_checkpoint_if_enabled("after_initial_execution")
                else:
                    logger.info("Skipping 'after_initial_execution' checkpoint because we resumed from a checkpoint")

            while self._iteration < self._max_iterations:
                logger.info(f"Starting superstep {self._iteration + 1}")

                # ライブイベントストリーミングと並行してイテレーションを実行します：イテレーションのコルーチンが進行する間に新しいイベントをポーリングします。
                iteration_task = asyncio.create_task(self._run_iteration())
                while not iteration_task.done():
                    try:
                        # 新しいイベントを短時間待機します；タイムアウトは進行状況のチェックを可能にします
                        event = await asyncio.wait_for(self._ctx.next_event(), timeout=0.05)
                        yield event
                    except asyncio.TimeoutError:
                        # 定期的にイテレーションの進行を継続させます
                        continue

                # イテレーションからのエラーを伝播しますが、まず保留中のイベントを表面化させます
                try:
                    await iteration_task
                except Exception:
                    # ExecutorFailedEventのような失敗関連イベントが表面化されることを保証します
                    if await self._ctx.has_events():
                        for event in await self._ctx.drain_events():
                            yield event
                    raise
                self._iteration += 1

                # 最後に発生した遅延イベントを排出します
                if await self._ctx.has_events():
                    for event in await self._ctx.drain_events():
                        yield event

                logger.info(f"Completed superstep {self._iteration}")

                # 各superstepイテレーション後にチェックポイントを作成します
                await self._create_checkpoint_if_enabled(f"superstep_{self._iteration}")

                if not await self._ctx.has_messages():
                    break

            if self._iteration >= self._max_iterations and await self._ctx.has_messages():
                raise RuntimeError(f"Runner did not converge after {self._max_iterations} iterations.")

            logger.info(f"Workflow completed after {self._iteration} supersteps")
            self._iteration = 0
            self._resumed_from_checkpoint = False  # 次回実行のために再開フラグをリセットします
        finally:
            self._running = False

    async def _run_iteration(self) -> None:
        async def _deliver_messages(source_executor_id: str, messages: list[Message]) -> None:
            """すべてのソースからターゲットへのメッセージを並行して配信する外側ループ。"""

            async def _deliver_message_inner(edge_runner: EdgeRunner, message: Message) -> bool:
                """エッジランナーを通じて単一メッセージを配信する内側ループ。"""
                return await edge_runner.send_message(message, self._shared_state, self._ctx)

            def _normalize_message_payload(message: Message) -> None:
                data = message.data
                if not isinstance(data, dict):
                    return
                if MODEL_MARKER not in data and DATACLASS_MARKER not in data:
                    return
                try:
                    decoded = decode_checkpoint_value(data)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to decode checkpoint payload during delivery: %s", exc)
                    return
                message.data = decoded

            # すべてのメッセージを通常のworkflowエッジを通じてルーティングします
            associated_edge_runners = self._edge_runner_map.get(source_executor_id, [])
            for message in messages:
                _normalize_message_payload(message)
                # ソースExecutorに関連付けられたすべてのエッジランナーを通じてメッセージを並行して配信します。
                tasks = [_deliver_message_inner(edge_runner, message) for edge_runner in associated_edge_runners]
                await asyncio.gather(*tasks)

        messages = await self._ctx.drain_messages()
        tasks = [_deliver_messages(source_executor_id, messages) for source_executor_id, messages in messages.items()]
        await asyncio.gather(*tasks)

    async def _create_checkpoint_if_enabled(self, checkpoint_type: str) -> str | None:
        """チェックポイントが有効な場合にチェックポイントを作成し、ラベルとメタデータを添付します。"""
        if not self._ctx.has_checkpointing():
            return None

        try:
            # ExecutorのStateを自動スナップショットします
            await self._auto_snapshot_executor_states()
            checkpoint_category = "initial" if checkpoint_type == "after_initial_execution" else "superstep"
            metadata = {
                "superstep": self._iteration,
                "checkpoint_type": checkpoint_category,
            }
            if self.graph_signature_hash:
                metadata["graph_signature"] = self.graph_signature_hash
            checkpoint_id = await self._ctx.create_checkpoint(
                self._shared_state,
                self._iteration,
                metadata=metadata,
            )
            logger.info(f"Created {checkpoint_type} checkpoint: {checkpoint_id}")
            return checkpoint_id
        except Exception as e:
            logger.warning(f"Failed to create {checkpoint_type} checkpoint: {e}")
            return None

    async def _auto_snapshot_executor_states(self) -> None:
        """Executorが利用可能な場合、Executorのスナップショットフックを呼び出してExecutorのStateを設定します。

        TODO(@taochen#1614): ExecutorがContext上で直接set_executor_stateを呼び出す場合、このメソッドは問題を起こす可能性があります。Executor State管理の意図された使用パターンを明確にすべきです。

        規約：
          - Executorがasyncまたはsyncのメソッド`snapshot_state(self) -> dict`を定義していればそれを使用します。
          - そうでなければ、dict型の単純な属性`state`があればそれを使用します。
        ExecutorはJSONシリアライズ可能なdictのみを提供すべきです。

        """
        for exec_id, executor in self._executors.items():
            state_dict: dict[str, Any] | None = None
            snapshot = getattr(executor, "snapshot_state", None)
            try:
                if callable(snapshot):
                    maybe = snapshot()
                    if asyncio.iscoroutine(maybe):  # type: ignore[arg-type]
                        maybe = await maybe  # type: ignore[assignment]
                    if isinstance(maybe, dict):
                        state_dict = maybe  # type: ignore[assignment]
                else:
                    state_attr = getattr(executor, "state", None)
                    if isinstance(state_attr, dict):
                        state_dict = state_attr  # type: ignore[assignment]
            except Exception as ex:  # pragma: no cover
                logger.debug(f"Executor {exec_id} snapshot_state failed: {ex}")

            if state_dict is not None:
                try:
                    await self._set_executor_state(exec_id, state_dict)
                except Exception as ex:  # pragma: no cover
                    logger.debug(f"Failed to persist state for executor {exec_id}: {ex}")

    async def restore_from_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_storage: CheckpointStorage | None = None,
    ) -> bool:
        """チェックポイントからworkflowのStateを復元します。

        Args:
            checkpoint_id: 復元するチェックポイントのID
            checkpoint_storage: ランナーContext自体がチェックポイントを設定していない場合にチェックポイントをロードするためのOptionalなストレージ

        Returns:
            復元が成功すればTrue、そうでなければFalse

        """
        try:
            # チェックポイントをロードします
            checkpoint: WorkflowCheckpoint | None
            if self._ctx.has_checkpointing():
                checkpoint = await self._ctx.load_checkpoint(checkpoint_id)
            elif checkpoint_storage is not None:
                checkpoint = await checkpoint_storage.load_checkpoint(checkpoint_id)
            else:
                logger.warning("Context does not support checkpointing and no external storage was provided")
                return False

            if not checkpoint:
                logger.error(f"Checkpoint {checkpoint_id} not found")
                return False

            # ロードしたチェックポイントをworkflowに対して検証します
            graph_hash = getattr(self, "graph_signature_hash", None)
            checkpoint_hash = (checkpoint.metadata or {}).get("graph_signature")
            if graph_hash and checkpoint_hash and graph_hash != checkpoint_hash:
                raise ValueError(
                    "Workflow graph has changed since the checkpoint was created. "
                    "Please rebuild the original workflow before resuming."
                )
            if graph_hash and not checkpoint_hash:
                logger.warning(
                    "Checkpoint %s does not include graph signature metadata; skipping topology validation.",
                    checkpoint_id,
                )

            self._workflow_id = checkpoint.workflow_id
            # 共有Stateを復元します
            await self._shared_state.import_state(checkpoint.shared_state)
            # 復元した共有Stateを使ってExecutorのStateを復元します
            await self._restore_executor_states()
            # チェックポイントをContextに適用します
            await self._ctx.apply_checkpoint(checkpoint)
            # ランナーを再開済みとしてマークします
            self._mark_resumed(checkpoint.iteration_count)

            logger.info(f"Successfully restored workflow from checkpoint: {checkpoint_id}")
            return True
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return False

    async def _restore_executor_states(self) -> None:
        has_executor_states = await self._shared_state.has(EXECUTOR_STATE_KEY)
        if not has_executor_states:
            return

        executor_states = await self._shared_state.get(EXECUTOR_STATE_KEY)
        if not isinstance(executor_states, dict):
            raise ValueError("Executor states in shared state is not a dictionary. Unable to restore.")

        for executor_id, state in executor_states.items():
            if not isinstance(executor_id, str):
                raise ValueError("Executor ID in executor states is not a string. Unable to restore.")
            if not isinstance(state, dict):
                raise ValueError(f"Executor state for {executor_id} is not a dictionary. Unable to restore.")

            executor = self._executors.get(executor_id)
            if not executor:
                raise ValueError(f"Executor {executor_id} not found during state restoration.")

            restored = False
            restore_method = getattr(executor, "restore_state", None)
            try:
                if callable(restore_method):
                    maybe = restore_method(state)
                    if asyncio.iscoroutine(maybe):  # type: ignore[arg-type]
                        await maybe  # type: ignore[arg-type]
                    restored = True
            except Exception as ex:  # pragma: no cover - defensive
                raise ValueError(f"Executor {executor_id} restore_state failed: {ex}") from ex

            if not restored:
                logger.debug(f"Executor {executor_id} does not support state restoration; skipping.")

    def _parse_edge_runners(self, edge_runners: list[EdgeRunner]) -> dict[str, list[EdgeRunner]]:
        """workflowのエッジランナーを解析し、各ソースExecutor IDがそのエッジランナーのリストにマップされる辞書を作成します。

        Args:
            edge_runners: workflow内のエッジランナーのリスト。

        Returns:
            各ソースExecutor IDをエッジランナーのリストにマップする辞書。

        """
        parsed: defaultdict[str, list[EdgeRunner]] = defaultdict(list)
        for runner in edge_runners:
            # 内部配線のために保護された属性(_edge_group)に意図的にアクセスしています。
            for source_executor_id in runner._edge_group.source_executor_ids:  # type: ignore[attr-defined]
                parsed[source_executor_id].append(runner)

        return parsed

    def _find_request_info_executor(self) -> "RequestInfoExecutor | None":
        """このworkflow内のRequestInfoExecutorインスタンスを探します。

        Returns:
            見つかればRequestInfoExecutorインスタンス、そうでなければNone。

        """
        from ._request_info_executor import RequestInfoExecutor

        for executor in self._executors.values():
            if isinstance(executor, RequestInfoExecutor):
                return executor
        return None

    def _is_message_to_request_info_executor(self, msg: "Message") -> bool:
        """メッセージがこのworkflow内のRequestInfoExecutorをターゲットにしているかどうかを確認します。

        Args:
            msg: 確認するメッセージ。

        Returns:
            メッセージがRequestInfoExecutorをターゲットにしていればTrue、そうでなければFalse。

        """
        from ._request_info_executor import RequestInfoExecutor

        if not msg.target_id:
            return False

        # すべてのExecutorをチェックしてtarget_idがRequestInfoExecutorに一致するか確認します
        for executor in self._executors.values():
            if executor.id == msg.target_id and isinstance(executor, RequestInfoExecutor):
                return True
        return False

    def _mark_resumed(self, iteration: int) -> None:
        """ランナーをチェックポイントから再開済みとしてマークします。

        オプションで現在のイテレーション数と最大イテレーション数を設定します。

        """
        self._resumed_from_checkpoint = True
        self._iteration = iteration

    async def _set_executor_state(self, executor_id: str, state: dict[str, Any]) -> None:
        """予約されたキーの下に共有StateにExecutorのStateを保存します。

        Executorは再開に必要な最小限の状態をキャプチャしたJSONシリアライズ可能なdictをこれで呼び出します。以前に保存されたStateは置き換えられます。

        """
        has_existing_states = await self._shared_state.has(EXECUTOR_STATE_KEY)
        if has_existing_states:
            existing_states = await self._shared_state.get(EXECUTOR_STATE_KEY)
        else:
            existing_states = {}

        if not isinstance(existing_states, dict):
            raise ValueError("Existing executor states in shared state is not a dictionary.")

        existing_states[executor_id] = state
        await self._shared_state.set(EXECUTOR_STATE_KEY, existing_states)
