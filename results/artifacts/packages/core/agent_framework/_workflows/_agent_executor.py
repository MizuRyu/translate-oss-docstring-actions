# Copyright (c) Microsoft. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Any

from .._agents import AgentProtocol, ChatAgent
from .._threads import AgentThread
from .._types import AgentRunResponse, AgentRunResponseUpdate, ChatMessage
from ._events import (
    AgentRunEvent,
    AgentRunUpdateEvent,  # type: ignore[reportPrivateUsage]
)
from ._executor import Executor, handler
from ._message_utils import normalize_messages_input
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutorRequest:
    """AgentExecutorへのリクエスト。

    Attributes:
        messages: Agentが処理するチャットメッセージのリスト。
        should_respond: Agentがメッセージに応答すべきかを示すフラグ。
            Falseの場合、メッセージはExecutorのキャッシュに保存されるがAgentには送信されない。

    """

    messages: list[ChatMessage]
    should_respond: bool = True


@dataclass
class AgentExecutorResponse:
    """AgentExecutorからのレスポンス。

    Attributes:
        executor_id: レスポンスを生成したExecutorのID。
        agent_run_response: 基本となるAgentRunResponse（クライアントから変更なし）。
        full_conversation: 別のAgentExecutorにチェーンする際に使用すべき完全な会話コンテキスト（過去の入力＋全てのアシスタント/ツール出力）。
            これにより下流のAgentがユーザープロンプトを失わず、AgentRunEventのテキストは生のAgent出力に忠実に保たれる。

    """

    executor_id: str
    agent_run_response: AgentRunResponse
    full_conversation: list[ChatMessage] | None = None


class AgentExecutor(Executor):
    """メッセージ処理のためにAgentをラップする組み込みExecutor。

    AgentExecutorはワークフロー実行モードに応じて動作を適応する:
    - run_stream(): Agentがトークンを生成するたびに増分のAgentRunUpdateEventイベントを発行
    - run(): 完全なレスポンスを含む単一のAgentRunEventを発行

    ExecutorはWorkflowContext.is_streaming()でモードを自動検出する。

    """

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        agent_thread: AgentThread | None = None,
        output_response: bool = False,
        id: str | None = None,
    ):
        """一意の識別子でExecutorを初期化する。

        Args:
            agent: このExecutorがラップするAgent。
            agent_thread: Agentを実行するためのスレッド。Noneの場合は新しいスレッドを作成。
            output_response: Agent完了時にAgentRunResponseをワークフロー出力としてyieldするか。
            id: Executorの一意識別子。Noneの場合はagentのnameを使用（存在すれば）。

        """
        # 提供されたidを優先；なければagent.nameを使用（存在すれば）；それもなければ決定論的なプレフィックスを生成
        exec_id = id or agent.name
        if not exec_id:
            raise ValueError("Agent must have a name or an explicit id must be provided.")
        super().__init__(exec_id)
        self._agent = agent
        self._agent_thread = agent_thread or self._agent.get_new_thread()
        self._output_response = output_response
        self._cache: list[ChatMessage] = []

    @property
    def workflow_output_types(self) -> list[type[Any]]:
        # 有効な場合にのみAgentRunResponseを可能な出力タイプとして宣言するためにオーバーライド。
        if self._output_response:
            return [AgentRunResponse]
        return []

    async def _run_agent_and_emit(self, ctx: WorkflowContext[AgentExecutorResponse, AgentRunResponse]) -> None:
        """基盤となるAgentを実行し、イベントを発行し、レスポンスをキューに入れる。

        ctx.is_streaming()をチェックして増分のAgentRunUpdateEventイベント（ストリーミングモード）か単一のAgentRunEvent（非ストリーミングモード）を発行するか決定する。

        """
        if ctx.is_streaming():
            # ストリーミングモード：増分更新を発行する
            updates: list[AgentRunResponseUpdate] = []
            async for update in self._agent.run_stream(
                self._cache,
                thread=self._agent_thread,
            ):
                updates.append(update)
                await ctx.add_event(AgentRunUpdateEvent(self.id, update))

            if isinstance(self._agent, ChatAgent):
                response_format = self._agent.chat_options.response_format
                response = AgentRunResponse.from_agent_run_response_updates(
                    updates,
                    output_format_type=response_format,
                )
            else:
                response = AgentRunResponse.from_agent_run_response_updates(updates)
        else:
            # 非ストリーミングモード：run()を使い単一イベントを発行する
            response = await self._agent.run(
                self._cache,
                thread=self._agent_thread,
            )
            await ctx.add_event(AgentRunEvent(self.id, response))

        if self._output_response:
            await ctx.yield_output(response)

        # 常に入力（キャッシュ）とAgent出力（agent_run_response.messages）から完全な会話スナップショットを構築する。
        # response.messagesを変更しないことでAgentRunEventが生の出力に忠実であることを保証する。
        full_conversation: list[ChatMessage] = list(self._cache) + list(response.messages)

        agent_response = AgentExecutorResponse(self.id, response, full_conversation=full_conversation)
        await ctx.send_message(agent_response)
        self._cache.clear()

    @handler
    async def run(
        self, request: AgentExecutorRequest, ctx: WorkflowContext[AgentExecutorResponse, AgentRunResponse]
    ) -> None:
        """AgentExecutorRequest（標準入力）を処理する。

        これは標準パスであり、提供されたメッセージでキャッシュを拡張し、should_respondがTrueならAgentを実行してAgentExecutorResponseを下流に発行する。

        """
        self._cache.extend(request.messages)
        if request.should_respond:
            await self._run_agent_and_emit(ctx)

    @handler
    async def from_response(
        self, prior: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorResponse, AgentRunResponse]
    ) -> None:
        """シームレスなチェーンを可能にする：前のAgentExecutorResponseを入力として受け入れる。

        戦略：前のレスポンスのメッセージを会話状態として扱い、即座にAgentを実行して新しいレスポンスを生成する。

        """
        # 利用可能なら完全な会話でキャッシュを置き換え、なければagent_run_responseのメッセージにフォールバックする。
        if prior.full_conversation is not None:
            self._cache = list(prior.full_conversation)
        else:
            self._cache = list(prior.agent_run_response.messages)
        await self._run_agent_and_emit(ctx)

    @handler
    async def from_str(self, text: str, ctx: WorkflowContext[AgentExecutorResponse, AgentRunResponse]) -> None:
        """生のユーザープロンプト文字列を受け入れてAgentを実行（一回限り）。"""
        self._cache = normalize_messages_input(text)
        await self._run_agent_and_emit(ctx)

    @handler
    async def from_message(
        self,
        message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorResponse, AgentRunResponse],
    ) -> None:
        """単一のChatMessageを入力として受け入れる。"""
        self._cache = normalize_messages_input(message)
        await self._run_agent_and_emit(ctx)

    @handler
    async def from_messages(
        self,
        messages: list[str | ChatMessage],
        ctx: WorkflowContext[AgentExecutorResponse, AgentRunResponse],
    ) -> None:
        """チャット入力（文字列またはChatMessage）のリストを会話コンテキストとして受け入れる。"""
        self._cache = normalize_messages_input(messages)
        await self._run_agent_and_emit(ctx)

    def snapshot_state(self) -> dict[str, Any]:
        """チェックポイント用に現在のExecutor状態をキャプチャする。

        Returns:
            シリアライズされたキャッシュ状態を含む辞書

        """
        from ._conversation_state import encode_chat_messages

        return {
            "cache": encode_chat_messages(self._cache),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """チェックポイントからExecutor状態を復元する。

        Args:
            state: チェックポイントデータの辞書

        """
        from ._conversation_state import decode_chat_messages

        cache_payload = state.get("cache")
        if cache_payload:
            try:
                self._cache = decode_chat_messages(cache_payload)
            except Exception as exc:
                logger.warning("Failed to restore cache: %s", exc)
                self._cache = []
        else:
            self._cache = []

    def reset(self) -> None:
        """Executorの内部キャッシュをリセットする。"""
        logger.debug("AgentExecutor %s: Resetting cache", self.id)
        self._cache.clear()
