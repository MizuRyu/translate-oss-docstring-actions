# Copyright (c) Microsoft. All rights reserved.

"""会話のハンドオフワークフローのためのハイレベルビルダー。

ハンドオフパターンは、コーディネーターAgentがオプションで
スペシャリストAgentに制御をルーティングし、その後会話をユーザーに戻すモデルです。
フローは意図的に循環的です：

    user input -> coordinator -> optional specialist -> request user input -> ...

主な特徴:
- 会話全体が維持され、各ホップで再利用される
- コーディネーターはスペシャリストの名前を指定するツールコールを呼び出してハンドオフを示す
- スペシャリストが応答した後、ワークフローは即座に新しいユーザー入力を要求する
"""

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_framework import (
    AgentProtocol,
    AgentRunResponse,
    AIFunction,
    ChatMessage,
    FunctionApprovalRequestContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    ai_function,
)

from .._agents import ChatAgent
from .._middleware import FunctionInvocationContext, FunctionMiddleware
from ._agent_executor import AgentExecutor, AgentExecutorRequest, AgentExecutorResponse
from ._base_group_chat_orchestrator import BaseGroupChatOrchestrator
from ._checkpoint import CheckpointStorage
from ._executor import Executor, handler
from ._group_chat import (
    _default_participant_factory,  # type: ignore[reportPrivateUsage]
    _GroupChatConfig,  # type: ignore[reportPrivateUsage]
    assemble_group_chat_workflow,
)
from ._orchestrator_helpers import clean_conversation_for_handoff
from ._participant_utils import GroupChatParticipantSpec, prepare_participant_metadata, sanitize_identifier
from ._request_info_executor import RequestInfoExecutor, RequestInfoMessage, RequestResponse
from ._workflow import Workflow
from ._workflow_builder import WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


_HANDOFF_TOOL_PATTERN = re.compile(r"(?:handoff|transfer)[_\s-]*to[_\s-]*(?P<target>[\w-]+)", re.IGNORECASE)


def _create_handoff_tool(alias: str, description: str | None = None) -> AIFunction[Any, Any]:
    """`alias`へのルーティングを示す合成ハンドオフツールを構築します。"""
    sanitized = sanitize_identifier(alias)
    tool_name = f"handoff_to_{sanitized}"
    doc = description or f"Handoff to the {alias} agent."

    # 注意：approval_modeはハンドオフツールには意図的に設定されていません。
    # ハンドオフツールはフレームワーク内部の信号であり、ルーティングロジックをトリガーします。 実際の関数実行ではありません。自動的にインターセプトされ、
    # 実際には実行されないため、承認は不要であり、 会話のクリーンアップ時にtool_calls/responsesのペアリングに問題を引き起こします。
    @ai_function(name=tool_name, description=doc)
    def _handoff_tool(context: str | None = None) -> str:
        """ターゲットのaliasをエンコードした決定論的な承認を返します。"""
        return f"Handoff to {alias}"

    return _handoff_tool


def _clone_chat_agent(agent: ChatAgent) -> ChatAgent:
    """ChatAgentのランタイム設定を保持しつつディープコピーを作成します。"""
    options = agent.chat_options
    middleware = list(agent.middleware or [])

    return ChatAgent(
        chat_client=agent.chat_client,
        instructions=options.instructions,
        id=agent.id,
        name=agent.name,
        description=agent.description,
        chat_message_store_factory=agent.chat_message_store_factory,
        context_providers=agent.context_provider,
        middleware=middleware,
        frequency_penalty=options.frequency_penalty,
        logit_bias=dict(options.logit_bias) if options.logit_bias else None,
        max_tokens=options.max_tokens,
        metadata=dict(options.metadata) if options.metadata else None,
        model_id=options.model_id,
        presence_penalty=options.presence_penalty,
        response_format=options.response_format,
        seed=options.seed,
        stop=options.stop,
        store=options.store,
        temperature=options.temperature,
        tool_choice=options.tool_choice,  # type: ignore[arg-type]
        tools=list(options.tools) if options.tools else None,
        top_p=options.top_p,
        user=options.user,
        additional_chat_options=dict(options.additional_properties),
    )


@dataclass
class HandoffUserInputRequest(RequestInfoMessage):
    """ワークフローが新しいユーザー入力を必要とするときに発行されるリクエストメッセージ。"""

    conversation: list[ChatMessage] = field(default_factory=lambda: [])  # type: ignore[misc]
    awaiting_agent_id: str | None = None
    prompt: str | None = None


@dataclass
class _ConversationWithUserInput:
    """ゲートウェイからコーディネーターへ送られる、完全な会話と新しいユーザーメッセージを含む内部メッセージ。"""

    full_conversation: list[ChatMessage] = field(default_factory=lambda: [])  # type: ignore[misc]


class _AutoHandoffMiddleware(FunctionMiddleware):
    """ハンドオフツールの呼び出しをインターセプトし、合成結果で実行をショートサーキットします。"""

    def __init__(self, handoff_targets: Mapping[str, str]) -> None:
        """ツール名からスペシャリストIDへのマッピングでミドルウェアを初期化します。"""
        self._targets = {name.lower(): target for name, target in handoff_targets.items()}

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        """一致するハンドオフツール呼び出しをインターセプトし、合成結果を注入します。"""
        name = getattr(context.function, "name", "")
        normalized = name.lower() if name else ""
        target = self._targets.get(normalized)
        if target is None:
            await next(context)
            return

        # 実行をショートサーキットし、ツール呼び出しに対して決定論的な応答ペイロードを提供します。
        context.result = {"handoff_to": target}
        context.terminate = True


class _InputToConversation(Executor):
    """初期ワークフロー入力をlist[ChatMessage]に正規化します。"""

    @handler
    async def from_str(self, prompt: str, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        """生のユーザープロンプトを単一のユーザーメッセージを含む会話に変換します。"""
        await ctx.send_message([ChatMessage(Role.USER, text=prompt)])

    @handler
    async def from_message(self, message: ChatMessage, ctx: WorkflowContext[list[ChatMessage]]) -> None:  # type: ignore[name-defined]
        """既存のチャットメッセージを初期会話としてそのまま通過させます。"""
        await ctx.send_message([message])

    @handler
    async def from_messages(
        self,
        messages: list[ChatMessage],
        ctx: WorkflowContext[list[ChatMessage]],
    ) -> None:  # type: ignore[name-defined]
        """チャットメッセージのリストを開始会話履歴として転送します。"""
        await ctx.send_message(list(messages))


@dataclass
class _HandoffResolution:
    """ハンドオフ検出の結果で、ターゲットaliasと発信元呼び出しを含みます。"""

    target: str
    function_call: FunctionCallContent | None = None


def _resolve_handoff_target(agent_response: AgentRunResponse) -> _HandoffResolution | None:
    """ツールコールのメタデータからハンドオフ意図を検出します。"""
    for message in agent_response.messages:
        resolution = _resolution_from_message(message)
        if resolution:
            return resolution

    for request in agent_response.user_input_requests:
        if isinstance(request, FunctionApprovalRequestContent):
            resolution = _resolution_from_function_call(request.function_call)
            if resolution:
                return resolution

    return None


def _resolution_from_message(message: ChatMessage) -> _HandoffResolution | None:
    """アシスタントメッセージを検査し、埋め込まれたハンドオフツールのメタデータを抽出します。"""
    for content in getattr(message, "contents", ()):
        if isinstance(content, FunctionApprovalRequestContent):
            resolution = _resolution_from_function_call(content.function_call)
            if resolution:
                return resolution
        elif isinstance(content, FunctionCallContent):
            resolution = _resolution_from_function_call(content)
            if resolution:
                return resolution
    return None


def _resolution_from_function_call(function_call: FunctionCallContent | None) -> _HandoffResolution | None:
    """関数呼び出しから解決されたターゲットを`_HandoffResolution`でラップします。"""
    if function_call is None:
        return None
    target = _target_from_function_call(function_call)
    if not target:
        return None
    return _HandoffResolution(target=target, function_call=function_call)


def _target_from_function_call(function_call: FunctionCallContent) -> str | None:
    """ツール名または構造化引数からハンドオフターゲットを抽出します。"""
    name_candidate = _target_from_tool_name(function_call.name)
    if name_candidate:
        return name_candidate

    arguments = function_call.parse_arguments()
    if isinstance(arguments, Mapping):
        value = arguments.get("handoff_to")
        if isinstance(value, str) and value.strip():
            return value.strip()
    elif isinstance(arguments, str):
        stripped = arguments.strip()
        if stripped:
            name_candidate = _target_from_tool_name(stripped)
            if name_candidate:
                return name_candidate
            return stripped

    return None


def _target_from_tool_name(name: str | None) -> str | None:
    """ハンドオフツール名にエンコードされたスペシャリストのaliasを解析します。"""
    if not name:
        return None
    match = _HANDOFF_TOOL_PATTERN.search(name)
    if match:
        parsed = match.group("target").strip()
        if parsed:
            return parsed
    return None


class _HandoffCoordinator(BaseGroupChatOrchestrator):
    """Agent間の転送とユーザーターン要求を調整します。"""

    def __init__(
        self,
        *,
        starting_agent_id: str,
        specialist_ids: Mapping[str, str],
        input_gateway_id: str,
        termination_condition: Callable[[list[ChatMessage]], bool | Awaitable[bool]],
        id: str,
        handoff_tool_targets: Mapping[str, str] | None = None,
    ) -> None:
        """スペシャリストとユーザー間のルーティングを管理するコーディネーターを作成します。"""
        super().__init__(id)
        self._starting_agent_id = starting_agent_id
        self._specialist_by_alias = dict(specialist_ids)
        self._specialist_ids = set(specialist_ids.values())
        self._input_gateway_id = input_gateway_id
        self._termination_condition = termination_condition
        self._handoff_tool_targets = {k.lower(): v for k, v in (handoff_tool_targets or {}).items()}

    def _get_author_name(self) -> str:
        """オーケストレーター生成メッセージのためのコーディネーター名を取得します。"""
        return "handoff_coordinator"

    @handler
    async def handle_agent_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[AgentExecutorRequest | list[ChatMessage], list[ChatMessage]],
    ) -> None:
        """エージェントの応答を処理し、ルーティング、入力要求、または終了を決定します。"""
        # チェックポイント可能なエグゼキューター状態を使用してコーディネーター状態を復元（新しい実行を検出）します。
        state = await ctx.get_executor_state()
        if not state:
            self._clear_conversation()
        elif not self._get_conversation():
            restored = self._restore_conversation_from_state(state)
            if restored:
                self._conversation = list(restored)

        source = ctx.get_source_executor_id()
        is_starting_agent = source == self._starting_agent_id

        # 実行の最初のターンでは会話は空です 新しいメッセージのみを追跡し、権威ある履歴を段階的に構築します
        conversation_msgs = self._get_conversation()
        if not conversation_msgs:
            # 開始エージェントからの最初の応答 - 権威ある会話スナップショットで初期化 ツールコールを含む完全な会話を保持（OpenAI
            # SDKのデフォルト動作）
            full_conv = self._conversation_from_response(response)
            self._conversation = list(full_conv)
        else:
            # 以降の応答 - このエージェントからの新しいメッセージのみを追加 完全な履歴を維持するためにツールコールを含むすべてのメッセージを保持
            new_messages = response.agent_run_response.messages or []
            self._conversation.extend(new_messages)

        self._apply_response_metadata(self._conversation, response.agent_run_response)

        conversation = list(self._conversation)

        # 任意のエージェント（開始エージェントまたはスペシャリスト）からのハンドオフをチェックします。
        target = self._resolve_specialist(response.agent_run_response, conversation)
        if target is not None:
            await self._persist_state(ctx)
            # 次のエージェントに送信する前にツール関連の内容をクリーンアップします。
            cleaned = clean_conversation_for_handoff(conversation)
            request = AgentExecutorRequest(messages=cleaned, should_respond=True)
            await ctx.send_message(request, target_id=target)
            return

        # ハンドオフが検出されない場合 - 応答は開始エージェントまたは既知のスペシャリストからのものでなければなりません。
        if not is_starting_agent and source not in self._specialist_ids:
            raise RuntimeError(f"HandoffCoordinator received response from unknown executor '{source}'.")

        await self._persist_state(ctx)

        if await self._check_termination():
            logger.info("Handoff workflow termination condition met. Ending conversation.")
            await ctx.yield_output(list(conversation))
            return

        await ctx.send_message(list(conversation), target_id=self._input_gateway_id)

    @handler
    async def handle_user_input(
        self,
        message: _ConversationWithUserInput,
        ctx: WorkflowContext[AgentExecutorRequest, list[ChatMessage]],
    ) -> None:
        """ゲートウェイから新しいユーザー入力を含む完全な会話を受け取り、履歴を更新し、エージェント用にトリムします。"""
        # 権威ある会話を更新します。
        self._conversation = list(message.full_conversation)
        await self._persist_state(ctx)

        # エージェントに送信する前に終了をチェックします。
        if await self._check_termination():
            logger.info("Handoff workflow termination condition met. Ending conversation.")
            await ctx.yield_output(list(self._conversation))
            return

        # 開始エージェントに送信する前にクリーンアップします。
        cleaned = clean_conversation_for_handoff(self._conversation)
        request = AgentExecutorRequest(messages=cleaned, should_respond=True)
        await ctx.send_message(request, target_id=self._starting_agent_id)

    def _resolve_specialist(self, agent_response: AgentRunResponse, conversation: list[ChatMessage]) -> str | None:
        """エージェント応答で要求されたスペシャリストエグゼキューターIDを解決します（存在する場合）。"""
        resolution = _resolve_handoff_target(agent_response)
        if not resolution:
            return None

        candidate = resolution.target
        normalized = candidate.lower()
        resolved_id: str | None
        if normalized in self._handoff_tool_targets:
            resolved_id = self._handoff_tool_targets[normalized]
        else:
            resolved_id = self._specialist_by_alias.get(candidate)

        if resolved_id:
            if resolution.function_call:
                self._append_tool_acknowledgement(conversation, resolution.function_call, resolved_id)
            return resolved_id

        lowered = candidate.lower()
        for alias, exec_id in self._specialist_by_alias.items():
            if alias.lower() == lowered:
                if resolution.function_call:
                    self._append_tool_acknowledgement(conversation, resolution.function_call, exec_id)
                return exec_id

        logger.warning("Handoff requested unknown specialist '%s'.", candidate)
        return None

    def _append_tool_acknowledgement(
        self,
        conversation: list[ChatMessage],
        function_call: FunctionCallContent,
        resolved_id: str,
    ) -> None:
        """解決されたスペシャリストIDを承認する合成ツール結果を追加します。"""
        call_id = getattr(function_call, "call_id", None)
        if not call_id:
            return

        result_payload: Any = {"handoff_to": resolved_id}
        result_content = FunctionResultContent(call_id=call_id, result=result_payload)
        tool_message = ChatMessage(
            role=Role.TOOL,
            contents=[result_content],
            author_name=function_call.name,
        )
        # ツール承認を送信中の会話と完全な履歴の両方に追加します。
        conversation.extend((tool_message,))
        self._append_messages((tool_message,))

    def _conversation_from_response(self, response: AgentExecutorResponse) -> list[ChatMessage]:
        """エグゼキューター応答から権威ある会話スナップショットを返します。"""
        conversation = response.full_conversation
        if conversation is None:
            raise RuntimeError(
                "AgentExecutorResponse.full_conversation missing; AgentExecutor must populate it in handoff workflows."
            )
        return list(conversation)

    async def _persist_state(self, ctx: WorkflowContext[Any, Any]) -> None:
        """豊富なメタデータを失わずに権威ある会話スナップショットを保存します。"""
        state_payload = self.snapshot_state()
        await ctx.set_executor_state(state_payload)

    def _snapshot_pattern_metadata(self) -> dict[str, Any]:
        """パターン固有の状態をシリアライズします。

        ハンドオフは基本会話状態以外の追加メタデータを持ちません。

        Returns:
            空の辞書（パターン固有の状態なし）

        """
        return {}

    def _restore_pattern_metadata(self, metadata: dict[str, Any]) -> None:
        """パターン固有の状態を復元します。

        ハンドオフは基本会話状態以外の追加メタデータを持ちません。

        Args:
            metadata: パターン固有の状態辞書（無視されます）

        """
        pass

    def _restore_conversation_from_state(self, state: Mapping[str, Any]) -> list[ChatMessage]:
        """チェックポイントされた状態からコーディネーターの会話履歴を復元します。

        DEPRECATED: 代わりにrestore_state()を使用してください。後方互換性のために保持されています。

        """
        from ._orchestration_state import OrchestrationState

        orch_state_dict = {"conversation": state.get("full_conversation", state.get("conversation", []))}
        temp_state = OrchestrationState.from_dict(orch_state_dict)
        return list(temp_state.conversation)

    def _apply_response_metadata(self, conversation: list[ChatMessage], agent_response: AgentRunResponse) -> None:
        """トップレベルの応答メタデータを最新のアシスタントメッセージにマージします。"""
        if not agent_response.additional_properties:
            return

        # この応答によって寄与された最新のアシスタントメッセージを見つけます。
        for message in reversed(conversation):
            if message.role == Role.ASSISTANT:
                metadata = agent_response.additional_properties or {}
                if not metadata:
                    return
                # エージェント応答から共有辞書を変更せずにメタデータをマージします。
                merged = dict(message.additional_properties or {})
                for key, value in metadata.items():
                    merged.setdefault(key, value)
                message.additional_properties = merged
                break


class _UserInputGateway(Executor):
    """会話コンテキストをRequestInfoExecutorと橋渡しし、ループに再入します。"""

    def __init__(
        self,
        *,
        request_executor_id: str,
        starting_agent_id: str,
        prompt: str | None,
        id: str,
    ) -> None:
        """ユーザー入力を要求し応答を転送するゲートウェイを初期化します。"""
        super().__init__(id)
        self._request_executor_id = request_executor_id
        self._starting_agent_id = starting_agent_id
        self._prompt = prompt or "Provide your next input for the conversation."

    @handler
    async def request_input(
        self,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[HandoffUserInputRequest],
    ) -> None:
        """会話スナップショットをキャプチャする`HandoffUserInputRequest`を発行します。"""
        if not conversation:
            raise ValueError("Handoff workflow requires non-empty conversation before requesting user input.")
        request = HandoffUserInputRequest(
            conversation=list(conversation),
            awaiting_agent_id=self._starting_agent_id,
            prompt=self._prompt,
        )
        request.source_executor_id = self.id
        await ctx.send_message(request, target_id=self._request_executor_id)

    @handler
    async def resume_from_user(
        self,
        response: RequestResponse[HandoffUserInputRequest, Any],
        ctx: WorkflowContext[_ConversationWithUserInput],
    ) -> None:
        """ユーザー入力応答をチャットメッセージに変換し、ワークフローを再開します。"""
        # 新しいユーザー入力を含む完全な会話を再構築します。
        conversation = list(response.original_request.conversation)
        user_messages = _as_user_messages(response.data)
        conversation.extend(user_messages)

        # 完全な会話をコーディネーターに返送します（トリムされていません） コーディネーターは権威ある履歴を更新し、エージェント用にトリムします。
        message = _ConversationWithUserInput(full_conversation=conversation)
        # 重要：すべての接続されたエグゼキューターにブロードキャストしないようにターゲットを指定する必要があります
        # ゲートウェイはrequest_infoとcoordinatorの両方に接続されていますが、coordinatorのみを対象とします。
        await ctx.send_message(message, target_id="handoff-coordinator")


def _as_user_messages(payload: Any) -> list[ChatMessage]:
    """任意のペイロードをユーザー作成のチャットメッセージに正規化します。"""
    if isinstance(payload, ChatMessage):
        if payload.role == Role.USER:
            return [payload]
        return [ChatMessage(Role.USER, text=payload.text)]
    if isinstance(payload, list):
        # すべてのアイテムがChatMessageインスタンスかどうかをチェックします。
        all_chat_messages = all(isinstance(msg, ChatMessage) for msg in payload)  # type: ignore[arg-type]
        if all_chat_messages:
            messages: list[ChatMessage] = payload  # type: ignore[assignment]
            return [msg if msg.role == Role.USER else ChatMessage(Role.USER, text=msg.text) for msg in messages]
    if isinstance(payload, Mapping):  # User supplied structured data
        text = payload.get("text") or payload.get("content")  # type: ignore[union-attr]
        if isinstance(text, str) and text.strip():
            return [ChatMessage(Role.USER, text=text.strip())]
    return [ChatMessage(Role.USER, text=str(payload))]  # type: ignore[arg-type]


def _default_termination_condition(conversation: list[ChatMessage]) -> bool:
    """デフォルトの終了条件：無限ループ防止のため10件のユーザーメッセージ後に停止します。"""
    user_message_count = sum(1 for msg in conversation if msg.role == Role.USER)
    return user_message_count >= 10


class HandoffBuilder:
    r"""Fluent builder for conversational handoff workflows with coordinator and specialist agents.

    The handoff pattern enables a coordinator agent to route requests to specialist agents.
    A termination condition determines when the workflow should stop requesting input and complete.

    Routing Patterns:

    **Single-Tier (Default):** Only the coordinator can hand off to specialists. After any specialist
    responds, control returns to the user for more input. This creates a cyclical flow:
    user -> coordinator -> [optional specialist] -> user -> coordinator -> ...

    **Multi-Tier (Advanced):** Specialists can hand off to other specialists using `.add_handoff()`.
    This provides more flexibility for complex workflows but is less controllable than the single-tier
    pattern. Users lose real-time visibility into intermediate steps during specialist-to-specialist
    handoffs (though the full conversation history including all handoffs is preserved and can be
    inspected afterward).


    Key Features:
    - **Automatic handoff detection**: The coordinator invokes a handoff tool whose
      arguments (for example ``{"handoff_to": "shipping_agent"}``) identify the specialist to receive control.
    - **Auto-generated tools**: By default the builder synthesizes `handoff_to_<agent>` tools for the coordinator,
      so you don't manually define placeholder functions.
    - **Full conversation history**: The entire conversation (including any
      `ChatMessage.additional_properties`) is preserved and passed to each agent.
    - **Termination control**: By default, terminates after 10 user messages. Override with
      `.with_termination_condition(lambda conv: ...)` for custom logic (e.g., detect "goodbye").
    - **Checkpointing**: Optional persistence for resumable workflows.

    Usage (Single-Tier):

    .. code-block:: python

        from agent_framework import HandoffBuilder
        from agent_framework.openai import OpenAIChatClient

        chat_client = OpenAIChatClient()

        # Create coordinator and specialist agents
        coordinator = chat_client.create_agent(
            instructions=(
                "You are a frontline support agent. Assess the user's issue and decide "
                "whether to hand off to 'refund_agent' or 'shipping_agent'. When delegation is "
                "required, call the matching handoff tool (for example `handoff_to_refund_agent`)."
            ),
            name="coordinator_agent",
        )

        refund = chat_client.create_agent(
            instructions="You handle refund requests. Ask for order details and process refunds.",
            name="refund_agent",
        )

        shipping = chat_client.create_agent(
            instructions="You resolve shipping issues. Track packages and update delivery status.",
            name="shipping_agent",
        )

        # Build the handoff workflow - default single-tier routing
        workflow = (
            HandoffBuilder(
                name="customer_support",
                participants=[coordinator, refund, shipping],
            )
            .set_coordinator("coordinator_agent")
            .build()
        )

        # Run the workflow
        events = await workflow.run_stream("My package hasn't arrived yet")
        async for event in events:
            if isinstance(event, RequestInfoEvent):
                # Request user input
                user_response = input("You: ")
                await workflow.send_response(event.data.request_id, user_response)

    **Multi-Tier Routing with .add_handoff():**

    .. code-block:: python

        # Enable specialist-to-specialist handoffs with fluent API
        workflow = (
            HandoffBuilder(participants=[coordinator, replacement, delivery, billing])
            .set_coordinator("coordinator_agent")
            .add_handoff(coordinator, [replacement, delivery, billing])  # Coordinator routes to all
            .add_handoff(replacement, [delivery, billing])  # Replacement delegates to delivery/billing
            .add_handoff(delivery, billing)  # Delivery escalates to billing
            .build()
        )

        # Flow: User → Coordinator → Replacement → Delivery → Back to User
        # (Replacement hands off to Delivery without returning to user)

    **Custom Termination Condition:**

    .. code-block:: python

        # Terminate when user says goodbye or after 5 exchanges
        workflow = (
            HandoffBuilder(participants=[coordinator, refund, shipping])
            .set_coordinator("coordinator_agent")
            .with_termination_condition(
                lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 5
                or any("goodbye" in msg.text.lower() for msg in conv[-2:])
            )
            .build()
        )

    **Checkpointing:**

    .. code-block:: python

        from agent_framework import InMemoryCheckpointStorage

        storage = InMemoryCheckpointStorage()
        workflow = (
            HandoffBuilder(participants=[coordinator, refund, shipping])
            .set_coordinator("coordinator_agent")
            .with_checkpointing(storage)
            .build()
        )

    Args:
        name: Optional workflow name for identification and logging.
        participants: List of agents (AgentProtocol) or executors to participate in the handoff.
                     The first agent you specify as coordinator becomes the orchestrating agent.
        description: Optional human-readable description of the workflow.

    Raises:
        ValueError: If participants list is empty, contains duplicates, or coordinator not specified.
        TypeError: If participants are not AgentProtocol or Executor instances.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        participants: Sequence[AgentProtocol | Executor] | None = None,
        description: str | None = None,
    ) -> None:
        r"""Initialize a HandoffBuilder for creating conversational handoff workflows.

        The builder starts in an unconfigured state and requires you to call:
        1. `.participants([...])` - Register agents
        2. `.set_coordinator(...)` - Designate which agent receives initial user input
        3. `.build()` - Construct the final Workflow

        Optional configuration methods allow you to customize context management,
        termination logic, and persistence.

        Args:
            name: Optional workflow identifier used in logging and debugging.
                 If not provided, a default name will be generated.
            participants: Optional list of agents (AgentProtocol) or executors that will
                         participate in the handoff workflow. You can also call
                         `.participants([...])` later. Each participant must have a
                         unique identifier (name for agents, id for executors).
            description: Optional human-readable description explaining the workflow's
                        purpose. Useful for documentation and observability.

        Note:
            Participants must have stable names/ids because the workflow maps the
            handoff tool arguments to these identifiers. Agent names should match
            the strings emitted by the coordinator's handoff tool (e.g., a tool that
            outputs ``{\"handoff_to\": \"billing\"}`` requires an agent named ``billing``).
        """
        self._name = name
        self._description = description
        self._executors: dict[str, Executor] = {}
        self._aliases: dict[str, str] = {}
        self._starting_agent_id: str | None = None
        self._checkpoint_storage: CheckpointStorage | None = None
        self._request_prompt: str | None = None
        # Termination condition
        self._termination_condition: Callable[[list[ChatMessage]], bool | Awaitable[bool]] = (
            _default_termination_condition
        )
        self._auto_register_handoff_tools: bool = True
        self._handoff_config: dict[str, list[str]] = {}  # Maps agent_id -> [target_agent_ids]

        if participants:
            self.participants(participants)

    def participants(self, participants: Sequence[AgentProtocol | Executor]) -> "HandoffBuilder":
        """Register the agents or executors that will participate in the handoff workflow.

        Each participant must have a unique identifier (name for agents, id for executors).
        The workflow will automatically create an alias map so agents can be referenced by
        their name, display_name, or executor id when routing.

        Args:
            participants: Sequence of AgentProtocol or Executor instances. Each must have
                         a unique identifier. For agents, the name attribute is used as the
                         primary identifier and must match handoff target strings.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If participants is empty or contains duplicates.
            TypeError: If participants are not AgentProtocol or Executor instances.

        Example:

        .. code-block:: python

            from agent_framework import HandoffBuilder
            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            coordinator = client.create_agent(instructions="...", name="coordinator")
            refund = client.create_agent(instructions="...", name="refund_agent")
            billing = client.create_agent(instructions="...", name="billing_agent")

            builder = HandoffBuilder().participants([coordinator, refund, billing])
            # Now you can call .set_coordinator() to designate the entry point

        Note:
            This method resets any previously configured coordinator, so you must call
            `.set_coordinator(...)` again after changing participants.
        """
        if not participants:
            raise ValueError("participants cannot be empty")

        named: dict[str, AgentProtocol | Executor] = {}
        for participant in participants:
            identifier: str
            if isinstance(participant, Executor):
                identifier = participant.id
            elif isinstance(participant, AgentProtocol):
                name_attr = getattr(participant, "name", None)
                if not name_attr:
                    raise ValueError(
                        "Agents used in handoff workflows must have a stable name "
                        "so they can be addressed during routing."
                    )
                identifier = str(name_attr)
            else:
                raise TypeError(
                    f"Participants must be AgentProtocol or Executor instances. Got {type(participant).__name__}."
                )
            if identifier in named:
                raise ValueError(f"Duplicate participant name '{identifier}' detected")
            named[identifier] = participant

        metadata = prepare_participant_metadata(
            named,
            description_factory=lambda name, participant: getattr(participant, "description", None) or name,
        )

        wrapped = metadata["executors"]
        seen_ids: set[str] = set()
        for executor in wrapped.values():
            if executor.id in seen_ids:
                raise ValueError(f"Duplicate participant with id '{executor.id}' detected")
            seen_ids.add(executor.id)

        self._executors = {executor.id: executor for executor in wrapped.values()}
        self._aliases = metadata["aliases"]
        self._starting_agent_id = None
        return self

    def set_coordinator(self, agent: str | AgentProtocol | Executor) -> "HandoffBuilder":
        r"""Designate which agent receives initial user input and orchestrates specialist routing.

        The coordinator agent is responsible for analyzing user requests and deciding whether to:
        1. Handle the request directly and respond to the user, OR
        2. Hand off to a specialist agent by including handoff metadata in the response

        After a specialist responds, the workflow automatically returns control to the user,
        creating a cyclical flow: user -> coordinator -> [specialist] -> user -> ...

        Args:
            agent: The agent to use as the coordinator. Can be:
                  - Agent name (str): e.g., "coordinator_agent"
                  - AgentProtocol instance: The actual agent object
                  - Executor instance: A custom executor wrapping an agent

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If participants(...) hasn't been called yet, or if the specified
                       agent is not in the participants list.

        Example:

        .. code-block:: python

            # Use agent name
            builder = HandoffBuilder().participants([coordinator, refund, billing]).set_coordinator("coordinator")

            # Or pass the agent object directly
            builder = HandoffBuilder().participants([coordinator, refund, billing]).set_coordinator(coordinator)

        Note:
            The coordinator determines routing by invoking a handoff tool call whose
            arguments identify the target specialist (for example ``{\"handoff_to\": \"billing\"}``).
            Decorate the tool with ``approval_mode="always_require"`` to ensure the workflow
            intercepts the call before execution and can make the transition.
        """
        if not self._executors:
            raise ValueError("Call participants(...) before coordinator(...)")
        resolved = self._resolve_to_id(agent)
        if resolved not in self._executors:
            raise ValueError(f"coordinator '{resolved}' is not part of the participants list")
        self._starting_agent_id = resolved
        return self

    def add_handoff(
        self,
        source: str | AgentProtocol | Executor,
        targets: str | AgentProtocol | Executor | Sequence[str | AgentProtocol | Executor],
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> "HandoffBuilder":
        """ソースAgentから1つ以上のターゲットAgentへのhandoffルーティングを追加します。

        このメソッドは、どのAgentがどのAgentにhandoffできるかを設定することで、スペシャリスト間のhandoffを可能にします。完全なルーティンググラフを構築するために、このメソッドを複数回呼び出してください。デフォルトでは、開始Agentのみが他のすべての参加者にhandoffできます。追加のルーティングパスを有効にするには、このメソッドを使用します。

        Args:
            source: handoffを開始できるAgent。以下のいずれかです：
                   - Agent名（str）：例 "triage_agent"
                   - AgentProtocolインスタンス：実際のAgentオブジェクト
                   - Executorインスタンス：AgentをラップするカスタムExecutor
            targets: sourceがhandoffできる1つ以上のターゲットAgent。以下のいずれかです：
                    - 単一Agent："billing_agent" または agent_instance
                    - 複数Agent： ["billing_agent", "support_agent"] または [agent1, agent2]
            tool_name: handoffツールの任意のカスタム名。指定しない場合、単一ターゲットの場合は "handoff_to_<target>", 複数ターゲットの場合は "handoff_to_<target>_agent" をターゲット名に基づいて生成します。
            tool_description: handoffツールの任意のカスタム説明。指定しない場合、"Handoff to the <target> agent." を生成します。

        Returns:
            メソッドチェーンのためのself。

        Raises:
            ValueError: sourceまたはtargetsが参加者リストに存在しない場合、またはparticipants(...)がまだ呼ばれていない場合。

        Examples:
            単一ターゲット:

            .. code-block:: python

                builder.add_handoff("triage_agent", "billing_agent")

            複数ターゲット（Agent名を使用）:

            .. code-block:: python

                builder.add_handoff("triage_agent", ["billing_agent", "support_agent", "escalation_agent"])

            複数ターゲット（Agentインスタンスを使用）:

            .. code-block:: python

                builder.add_handoff(triage, [billing, support, escalation])

            複数の設定をチェーン:

            .. code-block:: python

                workflow = (
                    HandoffBuilder(participants=[triage, replacement, delivery, billing])
                    .set_coordinator(triage)
                    .add_handoff(triage, [replacement, delivery, billing])
                    .add_handoff(replacement, [delivery, billing])
                    .add_handoff(delivery, billing)
                    .build()
                )

            カスタムツール名と説明:

            .. code-block:: python

                builder.add_handoff(
                    "support_agent",
                    "escalation_agent",
                    tool_name="escalate_to_l2",
                    tool_description="Escalate this issue to Level 2 support",
                )

        Note:
            - handoffツールは各source Agentに自動的に登録されます
            - add_handoffでsource Agentが複数回設定された場合、targetsはマージされます

        """
        if not self._executors:
            raise ValueError("Call participants(...) before add_handoff(...)")

        # source Agent IDを解決します
        source_id = self._resolve_to_id(source)
        if source_id not in self._executors:
            raise ValueError(f"Source agent '{source}' is not in the participants list")

        # targetsをリストに正規化します
        target_list = [targets] if isinstance(targets, (str, AgentProtocol, Executor)) else list(targets)

        # すべてのtarget IDを解決します
        target_ids: list[str] = []
        for target in target_list:
            target_id = self._resolve_to_id(target)
            if target_id not in self._executors:
                raise ValueError(f"Target agent '{target}' is not in the participants list")
            target_ids.append(target_id)

        # このsourceの既存のhandoff設定とマージします
        if source_id in self._handoff_config:
            # 重複を避けて既存のリストに新しいtargetsを追加します
            existing = self._handoff_config[source_id]
            for target_id in target_ids:
                if target_id not in existing:
                    existing.append(target_id)
        else:
            self._handoff_config[source_id] = target_ids

        return self

    def auto_register_handoff_tools(self, enabled: bool) -> "HandoffBuilder":
        """builderがstarting Agentのためにhandoffツールを合成すべきかどうかを設定します。"""
        self._auto_register_handoff_tools = enabled
        return self

    def _apply_auto_tools(self, agent: ChatAgent, specialists: Mapping[str, Executor]) -> dict[str, str]:
        """チャットAgentに合成されたhandoffツールを添付し、targetのルックアップテーブルを返します。"""
        chat_options = agent.chat_options
        existing_tools = list(chat_options.tools or [])
        existing_names = {getattr(tool, "name", "") for tool in existing_tools if hasattr(tool, "name")}

        tool_targets: dict[str, str] = {}
        new_tools: list[Any] = []
        for exec_id in specialists:
            alias = exec_id
            sanitized = sanitize_identifier(alias)
            tool = _create_handoff_tool(alias)
            if tool.name not in existing_names:
                new_tools.append(tool)
            tool_targets[tool.name.lower()] = exec_id
            tool_targets[sanitized] = exec_id
            tool_targets[alias.lower()] = exec_id

        if new_tools:
            chat_options.tools = existing_tools + new_tools
        else:
            chat_options.tools = existing_tools

        return tool_targets

    def _resolve_agent_id(self, agent_identifier: str) -> str:
        """Agent識別子をexecutor IDに解決します。

        Args:
            agent_identifier: Agent名、表示名、またはexecutor IDのいずれか

        Returns:
            executor ID

        Raises:
            ValueError: 識別子が解決できない場合

        """
        # すでにexecutor IDかどうかをチェックします
        if agent_identifier in self._executors:
            return agent_identifier

        # エイリアスかどうかをチェックします
        if agent_identifier in self._aliases:
            return self._aliases[agent_identifier]

        # 見つかりませんでした
        raise ValueError(f"Agent identifier '{agent_identifier}' not found in participants")

    def _prepare_agent_with_handoffs(
        self,
        executor: AgentExecutor,
        target_agents: Mapping[str, Executor],
    ) -> tuple[AgentExecutor, dict[str, str]]:
        """指定されたtarget Agentのためにhandoffツールを追加してAgentを準備します。

        Args:
            executor: 準備するAgent executor
            target_agents: このAgentがhandoff可能なexecutor IDからターゲットexecutorへのマップ

        Returns:
            (更新されたexecutor, tool_targetsマップ)のタプル

        """
        agent = getattr(executor, "_agent", None)
        if not isinstance(agent, ChatAgent):
            return executor, {}

        cloned_agent = _clone_chat_agent(agent)
        tool_targets = self._apply_auto_tools(cloned_agent, target_agents)
        if tool_targets:
            middleware = _AutoHandoffMiddleware(tool_targets)
            existing_middleware = list(cloned_agent.middleware or [])
            existing_middleware.append(middleware)
            cloned_agent.middleware = existing_middleware

        new_executor = AgentExecutor(
            cloned_agent,
            agent_thread=getattr(executor, "_agent_thread", None),
            output_response=getattr(executor, "_output_response", False),
            id=executor.id,
        )
        return new_executor, tool_targets

    def request_prompt(self, prompt: str | None) -> "HandoffBuilder":
        """ユーザー入力を要求する際に表示されるカスタムプロンプトメッセージを設定します。

        デフォルトでは、ワークフローは一般的なプロンプト "Provide your next input for the conversation." を使用します。このメソッドを使って、ワークフローがユーザーの応答を必要とする際に表示されるメッセージをカスタマイズできます。

        Args:
            prompt: 表示するカスタムプロンプトテキスト、またはデフォルトプロンプトを使用する場合はNone。

        Returns:
            メソッドチェーンのためのself。

        Example:

        .. code-block:: python

            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .set_coordinator("triage")
                .request_prompt("How can we help you today?")
                .build()
            )

            # よりコンテキストに応じたプロンプトには、イベント処理ループ内でRequestInfoEvent.data.promptを参照できます

        Note:
            プロンプトは静的で、ワークフロー構築時に一度設定されます。会話の状態に基づく動的なプロンプトが必要な場合は、アプリケーションのイベント処理ロジックで対応してください。

        """
        self._request_prompt = prompt
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "HandoffBuilder":
        """再開可能な会話のためにワークフローの状態永続化を有効にします。

        チェックポイント機能により、ワークフローは重要なポイントで状態を保存でき、以下が可能になります：
        - アプリケーション再起動後に会話を再開
        - 複数セッションにまたがる長期サポートチケットの実装
        - 障害からの回復時に会話コンテキストを失わない
        - 会話履歴の監査および再生

        Args:
            checkpoint_storage: CheckpointStorageインターフェースを実装するストレージバックエンド。
                               一般的な実装例：InMemoryCheckpointStorage（テスト用）、
                               データベースベースのストレージ（本番用）。

        Returns:
            メソッドチェーンのためのself。

        Example (In-Memory):

        .. code-block:: python

            from agent_framework import InMemoryCheckpointStorage

            storage = InMemoryCheckpointStorage()
            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .set_coordinator("triage")
                .with_checkpointing(storage)
                .build()
            )

            # セッションIDを指定してワークフローを実行し再開可能に
            async for event in workflow.run_stream("Help me", session_id="user_123"):
                # イベント処理...
                pass

            # 後で同じ会話を再開
            async for event in workflow.run_stream("I need a refund", session_id="user_123"):
                # 会話は中断したところから続行
                pass

        Use Cases:
            - 永続的なチケット履歴を持つカスタマーサポートシステム
            - サーバー再起動をまたぐ数日にわたる会話
            - 会話監査のためのコンプライアンス要件
            - 同じ会話で異なるAgent構成のA/Bテスト

        Note:
            チェックポイントはシリアライズとストレージI/Oのオーバーヘッドを追加します。永続化が必要な場合に使用し、単純なステートレスなリクエスト-レスポンスパターンには使用しないでください。

        """
        self._checkpoint_storage = checkpoint_storage
        return self

    def with_termination_condition(
        self, condition: Callable[[list[ChatMessage]], bool | Awaitable[bool]]
    ) -> "HandoffBuilder":
        """handoffワークフローのカスタム終了条件を設定します。

        条件は同期または非同期のいずれかです。

        Args:
            condition: 会話全体を受け取り、ワークフローを終了すべき場合にTrue（またはawait可能なTrue）を返す関数。

        Returns:
            チェーン用のself。

        Example:

        .. code-block:: python

            # 同期条件
            builder.with_termination_condition(
                lambda conv: len(conv) > 20 or any("goodbye" in msg.text.lower() for msg in conv[-2:])
            )


            # 非同期条件
            async def check_termination(conv: list[ChatMessage]) -> bool:
                # 非同期処理が可能
                return len(conv) > 20


            builder.with_termination_condition(check_termination)

        """
        self._termination_condition = condition
        return self

    def build(self) -> Workflow:
        """設定されたbuilderから最終的なWorkflowインスタンスを構築します。

        このメソッドは設定を検証し、以下の内部コンポーネントを組み立てます：
        - 入力正規化executor
        - 開始Agent executor
        - Handoff coordinator
        - スペシャリストAgent executor
        - ユーザー入力ゲートウェイ
        - リクエスト/レスポンス処理

        Returns:
            `.run()`または`.run_stream()`で実行可能な完全に設定されたWorkflow。

        Raises:
            ValueError: 参加者またはcoordinatorが設定されていない場合、または必要な設定が無効な場合。

        Example (Minimal):

        .. code-block:: python

            workflow = (
                HandoffBuilder(participants=[coordinator, refund, billing]).set_coordinator("coordinator").build()
            )

            # ワークフローを実行
            async for event in workflow.run_stream("I need help"):
                # イベント処理...
                pass

        Example (Full Configuration):

        .. code-block:: python

            from agent_framework import InMemoryCheckpointStorage

            storage = InMemoryCheckpointStorage()
            workflow = (
                HandoffBuilder(
                    name="support_workflow",
                    participants=[coordinator, refund, billing],
                    description="Customer support with specialist routing",
                )
                .set_coordinator("coordinator")
                .with_termination_condition(lambda conv: len(conv) > 20)
                .request_prompt("How can we help?")
                .with_checkpointing(storage)
                .build()
            )

        Note:
            build()呼び出し後はbuilderインスタンスを再利用しないでください。異なる設定で別のワークフローを構築する場合は新しいbuilderを作成してください。

        """
        if not self._executors:
            raise ValueError("No participants provided. Call participants([...]) first.")
        if self._starting_agent_id is None:
            raise ValueError("coordinator must be defined before build().")

        starting_executor = self._executors[self._starting_agent_id]
        specialists = {
            exec_id: executor for exec_id, executor in self._executors.items() if exec_id != self._starting_agent_id
        }

        # handoffツールレジストリを必要なすべてのAgentのために構築します
        handoff_tool_targets: dict[str, str] = {}
        if self._auto_register_handoff_tools:
            # handoffツールを持つべきAgentを決定します
            if self._handoff_config:
                # add_handoff()呼び出しからの明示的なhandoff設定を使用します
                for source_exec_id, target_exec_ids in self._handoff_config.items():
                    executor = self._executors.get(source_exec_id)
                    if not executor:
                        raise ValueError(f"Handoff source agent '{source_exec_id}' not found in participants")

                    if isinstance(executor, AgentExecutor):
                        # このsource Agentのためのtargetsマップを構築します
                        targets_map: dict[str, Executor] = {}
                        for target_exec_id in target_exec_ids:
                            target_executor = self._executors.get(target_exec_id)
                            if not target_executor:
                                raise ValueError(f"Handoff target agent '{target_exec_id}' not found in participants")
                            targets_map[target_exec_id] = target_executor

                        # このAgentのためにhandoffツールを登録します
                        updated_executor, tool_targets = self._prepare_agent_with_handoffs(executor, targets_map)
                        self._executors[source_exec_id] = updated_executor
                        handoff_tool_targets.update(tool_targets)
        else:
            # デフォルト動作：coordinatorのみがすべてのスペシャリストへのhandoffツールを持ちます
            if isinstance(starting_executor, AgentExecutor) and specialists:
                starting_executor, tool_targets = self._prepare_agent_with_handoffs(starting_executor, specialists)
                self._executors[self._starting_agent_id] = starting_executor
                handoff_tool_targets.update(tool_targets)  # Agentの変更後に参照を更新します
        starting_executor = self._executors[self._starting_agent_id]
        specialists = {
            exec_id: executor for exec_id, executor in self._executors.items() if exec_id != self._starting_agent_id
        }

        if not specialists:
            logger.warning("Handoff workflow has no specialist agents; the coordinator will loop with the user.")

        descriptions = {
            exec_id: getattr(executor, "description", None) or exec_id for exec_id, executor in self._executors.items()
        }
        participant_specs = {
            exec_id: GroupChatParticipantSpec(name=exec_id, participant=executor, description=descriptions[exec_id])
            for exec_id, executor in self._executors.items()
        }

        input_node = _InputToConversation(id="input-conversation")
        request_info = RequestInfoExecutor(id=f"{starting_executor.id}_handoff_requests")
        user_gateway = _UserInputGateway(
            request_executor_id=request_info.id,
            starting_agent_id=starting_executor.id,
            prompt=self._request_prompt,
            id="handoff-user-input",
        )

        specialist_aliases = {alias: exec_id for alias, exec_id in self._aliases.items() if exec_id in specialists}

        def _handoff_orchestrator_factory(_: _GroupChatConfig) -> Executor:
            return _HandoffCoordinator(
                starting_agent_id=starting_executor.id,
                specialist_ids=specialist_aliases,
                input_gateway_id=user_gateway.id,
                termination_condition=self._termination_condition,
                id="handoff-coordinator",
                handoff_tool_targets=handoff_tool_targets,
            )

        wiring = _GroupChatConfig(
            manager=None,
            manager_name=self._starting_agent_id,
            participants=participant_specs,
            max_rounds=None,
            participant_aliases=self._aliases,
            participant_executors=self._executors,
        )

        result = assemble_group_chat_workflow(
            wiring=wiring,
            participant_factory=_default_participant_factory,
            orchestrator_factory=_handoff_orchestrator_factory,
            interceptors=(),
            checkpoint_storage=self._checkpoint_storage,
            builder=WorkflowBuilder(name=self._name, description=self._description),
            return_builder=True,
        )
        if not isinstance(result, tuple):
            raise TypeError("Expected tuple from assemble_group_chat_workflow with return_builder=True")
        builder, coordinator = result

        builder = builder.set_start_executor(input_node)
        builder = builder.add_edge(input_node, starting_executor)
        builder = builder.add_edge(coordinator, user_gateway)
        builder = builder.add_edge(user_gateway, request_info)
        builder = builder.add_edge(request_info, user_gateway)
        builder = builder.add_edge(user_gateway, coordinator)

        return builder.build()

    def _resolve_to_id(self, candidate: str | AgentProtocol | Executor) -> str:
        """参加者の参照を具体的なexecutor識別子に解決します。"""
        if isinstance(candidate, Executor):
            return candidate.id
        if isinstance(candidate, AgentProtocol):
            name: str | None = getattr(candidate, "name", None)
            if not name:
                raise ValueError("AgentProtocol without a name cannot be resolved to an executor id.")
            return self._aliases.get(name, name)
        if isinstance(candidate, str):
            if candidate in self._aliases:
                return self._aliases[candidate]
            return candidate
        raise TypeError(f"Invalid starting agent reference: {type(candidate).__name__}")
