# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from typing import Any

from typing_extensions import Never

from agent_framework import AgentProtocol, ChatMessage, Role

from ._agent_executor import AgentExecutorRequest, AgentExecutorResponse
from ._checkpoint import CheckpointStorage
from ._executor import Executor, handler
from ._message_utils import normalize_messages_input
from ._workflow import Workflow
from ._workflow_builder import WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

"""Concurrent builder for agent-only fan-out/fan-in workflows.

This module provides a high-level, agent-focused API to quickly assemble a
parallel workflow with:
- a default dispatcher that broadcasts the input to all agent participants
- a default aggregator that combines all agent conversations and completes the workflow

Notes:
- Participants should be AgentProtocol instances or Executors.
- A custom aggregator can be provided as:
  - an Executor instance (it should handle list[AgentExecutorResponse],
    yield output), or
  - a callback function with signature:
        def cb(results: list[AgentExecutorResponse]) -> Any | None
        def cb(results: list[AgentExecutorResponse], ctx: WorkflowContext) -> Any | None
    The callback is wrapped in _CallbackAggregator.
    If the callback returns a non-None value, _CallbackAggregator yields that as output.
    If it returns None, the callback may have already yielded an output via ctx, so no further action is taken.
"""


class _DispatchToAllParticipants(Executor):
    """入力をすべての下流参加者にブロードキャストします（fan-outエッジ経由）。"""

    @handler
    async def from_request(self, request: AgentExecutorRequest, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        # 明示的なターゲットなし：エッジルーティングはすべての接続された参加者に配信します。
        await ctx.send_message(request)

    @handler
    async def from_str(self, prompt: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        request = AgentExecutorRequest(messages=normalize_messages_input(prompt), should_respond=True)
        await ctx.send_message(request)

    @handler
    async def from_message(self, message: ChatMessage, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        request = AgentExecutorRequest(messages=normalize_messages_input(message), should_respond=True)
        await ctx.send_message(request)

    @handler
    async def from_messages(
        self,
        messages: list[str | ChatMessage],
        ctx: WorkflowContext[AgentExecutorRequest],
    ) -> None:
        request = AgentExecutorRequest(messages=normalize_messages_input(messages), should_respond=True)
        await ctx.send_message(request)


class _AggregateAgentConversations(Executor):
    """Agentのレスポンスを集約し、結合されたChatMessagesで完了します。

    以下の形状のlist[ChatMessage]を出力します：
      [ 単一ユーザープロンプト?, agent1_final_assistant, agent2_final_assistant, ... ]

    - 単一のユーザープロンプト（結果全体で最初に見つかったユーザーメッセージ）を抽出します。
    - 各結果について、最終的なアシスタントメッセージを選択します（agent_run_response.messagesを優先）。
    - 各Agentごとに同じユーザーメッセージの重複を避けます。
    """

    @handler
    async def aggregate(
        self, results: list[AgentExecutorResponse], ctx: WorkflowContext[Never, list[ChatMessage]]
    ) -> None:
        if not results:
            logger.error("Concurrent aggregator received empty results list")
            raise ValueError("Aggregation failed: no results provided")

        def _is_role(msg: Any, role: Role) -> bool:
            r = getattr(msg, "role", None)
            if r is None:
                return False
            # 比較のためにrとroleの両方を小文字の文字列に正規化します
            r_str = str(r).lower() if isinstance(r, str) or hasattr(r, "__str__") else r
            role_str = getattr(role, "value", None)
            if role_str is None:
                role_str = str(role)
            role_str = role_str.lower()
            return r_str == role_str

        prompt_message: ChatMessage | None = None
        assistant_replies: list[ChatMessage] = []

        for r in results:
            resp_messages = list(getattr(r.agent_run_response, "messages", []) or [])
            conv = r.full_conversation if r.full_conversation is not None else resp_messages

            logger.debug(
                f"Aggregating executor {getattr(r, 'executor_id', '<unknown>')}: "
                f"{len(resp_messages)} response msgs, {len(conv)} conversation msgs"
            )

            # 単一のユーザープロンプトをキャプチャします（任意の会話で最初に出現したもの）
            if prompt_message is None:
                found_user = next((m for m in conv if _is_role(m, Role.USER)), None)
                if found_user is not None:
                    prompt_message = found_user

            # レスポンスから最終的なアシスタントメッセージを選択します。フォールバックは会話検索です。
            final_assistant = next((m for m in reversed(resp_messages) if _is_role(m, Role.ASSISTANT)), None)
            if final_assistant is None:
                final_assistant = next((m for m in reversed(conv) if _is_role(m, Role.ASSISTANT)), None)

            if final_assistant is not None:
                assistant_replies.append(final_assistant)
            else:
                logger.warning(
                    f"No assistant reply found for executor {getattr(r, 'executor_id', '<unknown>')}; skipping"
                )

        if not assistant_replies:
            logger.error(f"Aggregation failed: no assistant replies found across {len(results)} results")
            raise RuntimeError("Aggregation failed: no assistant replies found")

        output: list[ChatMessage] = []
        if prompt_message is not None:
            output.append(prompt_message)
        else:
            logger.warning("No user prompt found in any conversation; emitting assistants only")
        output.extend(assistant_replies)

        await ctx.yield_output(output)


class _CallbackAggregator(Executor):
    """PythonのコールバックをAggregatorとしてラップします。

    以下のいずれかのシグネチャの非同期または同期コールバックを受け入れます：
      - (results: list[AgentExecutorResponse]) -> Any | None
      - (results: list[AgentExecutorResponse], ctx: WorkflowContext[Any]) -> Any | None

    注意:
    - 非同期コールバックは直接awaitされます。
    - 同期コールバックはasyncio.to_thread経由で実行され、イベントループのブロックを回避します。
    - コールバックがNone以外の値を返した場合、それが出力としてyieldされます。
    """

    def __init__(self, callback: Callable[..., Any], id: str | None = None) -> None:
        derived_id = getattr(callback, "__name__", "") or ""
        if not derived_id or derived_id == "<lambda>":
            derived_id = f"{type(self).__name__}_unnamed"
        super().__init__(id or derived_id)
        self._callback = callback
        self._param_count = len(inspect.signature(callback).parameters)

    @handler
    async def aggregate(self, results: list[AgentExecutorResponse], ctx: WorkflowContext[Never, Any]) -> None:
        # 提供されたシグネチャに従って呼び出し、同期コールバックでも常にノンブロッキングにします
        if self._param_count >= 2:
            if inspect.iscoroutinefunction(self._callback):
                ret = await self._callback(results, ctx)  # type: ignore[misc]
            else:
                ret = await asyncio.to_thread(self._callback, results, ctx)
        else:
            if inspect.iscoroutinefunction(self._callback):
                ret = await self._callback(results)  # type: ignore[misc]
            else:
                ret = await asyncio.to_thread(self._callback, results)

        # コールバックが値を返した場合、それでワークフローを最終化します
        if ret is not None:
            await ctx.yield_output(ret)


class ConcurrentBuilder:
    r"""並行Agentワークフローの高レベルビルダー。

    - `participants([...])`はAgentProtocol（推奨）またはExecutorのリストを受け入れます。
    - `build()`はdispatcher -> fan-out -> participants -> fan-in -> aggregatorを配線します。
    - `with_custom_aggregator(...)`はデフォルトのaggregatorをExecutorまたはコールバックで上書きします。

    使用例:

    .. code-block:: python

        from agent_framework import ConcurrentBuilder

        # 最小限：デフォルトaggregatorを使用（list[ChatMessage]を返す）
        workflow = ConcurrentBuilder().participants([agent1, agent2, agent3]).build()


        # コールバックによるカスタムaggregator（同期または非同期）。コールバックは
        # list[AgentExecutorResponse]を受け取り、その戻り値がワークフローの出力になります。
        def summarize(results):
            return " | ".join(r.agent_run_response.messages[-1].text for r in results)


        workflow = ConcurrentBuilder().participants([agent1, agent2, agent3]).with_custom_aggregator(summarize).build()


        # チェックポイント永続化を有効にして実行を再開可能にする
        workflow = ConcurrentBuilder().participants([agent1, agent2, agent3]).with_checkpointing(storage).build()

    """

    def __init__(self) -> None:
        self._participants: list[AgentProtocol | Executor] = []
        self._aggregator: Executor | None = None
        self._checkpoint_storage: CheckpointStorage | None = None

    def participants(self, participants: Sequence[AgentProtocol | Executor]) -> "ConcurrentBuilder":
        r"""この並行ワークフローの並列参加者を定義します。

        AgentProtocolインスタンス（例：chat clientで作成）またはExecutorインスタンスを受け入れます。
        各参加者は内部dispatcherからのfan-outエッジを使って並列ブランチとして配線されます。

        例外:
            ValueError: `participants`が空または重複を含む場合
            TypeError: エントリがAgentProtocolまたはExecutorでない場合

        例:

        .. code-block:: python

            wf = ConcurrentBuilder().participants([researcher_agent, marketer_agent, legal_agent]).build()

            # AgentとExecutorの混在もサポート
            wf2 = ConcurrentBuilder().participants([researcher_agent, my_custom_executor]).build()

        """
        if not participants:
            raise ValueError("participants cannot be empty")

        # 重複検出の防御的処理
        seen_agent_ids: set[int] = set()
        seen_executor_ids: set[str] = set()
        for p in participants:
            if isinstance(p, Executor):
                if p.id in seen_executor_ids:
                    raise ValueError(f"Duplicate executor participant detected: id '{p.id}'")
                seen_executor_ids.add(p.id)
            elif isinstance(p, AgentProtocol):
                pid = id(p)
                if pid in seen_agent_ids:
                    raise ValueError("Duplicate agent participant detected (same agent instance provided twice)")
                seen_agent_ids.add(pid)
            else:
                raise TypeError(f"participants must be AgentProtocol or Executor instances; got {type(p).__name__}")

        self._participants = list(participants)
        return self

    def with_aggregator(self, aggregator: Executor | Callable[..., Any]) -> "ConcurrentBuilder":
        r"""デフォルトのaggregatorをExecutorまたはコールバックで上書きします。

        - Executor: `list[AgentExecutorResponse]`を処理し、`ctx.yield_output(...)`を使って出力をyieldします。
          出力がyieldされるとワークフローはアイドル状態になります。
        - コールバック: 同期または非同期のcallableで、以下のシグネチャのいずれかを持ちます：
          `(results: list[AgentExecutorResponse]) -> Any | None` または
          `(results: list[AgentExecutorResponse], ctx: WorkflowContext) -> Any | None`。
          コールバックがNone以外の値を返した場合、それがワークフローの出力になります。

        例:

        .. code-block:: python

            # コールバックベースのaggregator（文字列結果）
            async def summarize(results):
                return " | ".join(r.agent_run_response.messages[-1].text for r in results)


            wf = ConcurrentBuilder().participants([a1, a2, a3]).with_custom_aggregator(summarize).build()

        """
        if isinstance(aggregator, Executor):
            self._aggregator = aggregator
        elif callable(aggregator):
            self._aggregator = _CallbackAggregator(aggregator)
        else:
            raise TypeError("aggregator must be an Executor or a callable")
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "ConcurrentBuilder":
        """提供されたストレージバックエンドを使用してチェックポイント永続化を有効にします。"""
        self._checkpoint_storage = checkpoint_storage
        return self

    def build(self) -> Workflow:
        r"""並行ワークフローをビルドして検証します。

        配線パターン:
        - Dispatcher（内部）が入力をすべての`participants`にfan-outします
        - Fan-in aggregatorが`AgentExecutorResponse`オブジェクトを収集します
        - Aggregatorが出力をyieldし、ワークフローはアイドル状態になります。出力は以下のいずれかです：
          - list[ChatMessage]（デフォルトaggregator：ユーザー1人＋エージェントごとに1人のアシスタント）
          - 提供されたコールバック/Executorによるカスタムペイロード

        戻り値:
            Workflow: 実行準備が整ったワークフローインスタンス

        例外:
            ValueError: 参加者が定義されていない場合

        例:

        .. code-block:: python

            workflow = ConcurrentBuilder().participants([agent1, agent2]).build()

        """
        if not self._participants:
            raise ValueError("No participants provided. Call .participants([...]) first.")

        dispatcher = _DispatchToAllParticipants(id="dispatcher")
        aggregator = self._aggregator or _AggregateAgentConversations(id="aggregator")

        builder = WorkflowBuilder()
        builder.set_start_executor(dispatcher)
        builder.add_fan_out_edges(dispatcher, list(self._participants))
        builder.add_fan_in_edges(list(self._participants), aggregator)

        if self._checkpoint_storage is not None:
            builder = builder.with_checkpointing(self._checkpoint_storage)

        return builder.build()
