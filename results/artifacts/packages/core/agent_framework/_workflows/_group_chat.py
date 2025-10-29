# Copyright (c) Microsoft. All rights reserved.

"""グループチャットのオーケストレーションプリミティブ。

このモジュールはマネージャー主導のマルチエージェント会話の再利用可能なオーケストレーションサーフェスを導入する。主なコンポーネントは:

- GroupChatRequestMessage / GroupChatResponseMessage: オーケストレーターと参加者間で使われる標準的なエンベロープ。
- グループチャットマネージャー: プラグ可能な調整ロジックのための最小限の非同期呼び出し可能。
- GroupChatOrchestratorExecutor: ランタイムステートマシンで、マネージャーに次の参加者選択やタスク完了を委譲。
- GroupChatBuilder: マネージャーと参加者をワークフローグラフに配線する高レベルビルダー。SequentialBuilderやConcurrentBuilderの使い勝手を模倣しつつ、Magenticが同じインフラを再利用可能にする。

デフォルトの配線はAgentExecutorを内部で使い、エージェント参加者の既存の可観測性やストリーミングセマンティクスを維持する。
"""

import inspect
import itertools
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field

from .._agents import AgentProtocol
from .._clients import ChatClientProtocol
from .._types import ChatMessage, Role
from ._agent_executor import AgentExecutorRequest, AgentExecutorResponse
from ._base_group_chat_orchestrator import BaseGroupChatOrchestrator
from ._checkpoint import CheckpointStorage
from ._conversation_history import ensure_author, latest_user_message
from ._executor import Executor, handler
from ._participant_utils import GroupChatParticipantSpec, prepare_participant_metadata, wrap_participant
from ._workflow import Workflow
from ._workflow_builder import WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


# region Message primitives


@dataclass
class _GroupChatRequestMessage:
    """内部: オーケストレーターから参加者に送信されるRequestエンベロープ。"""

    agent_name: str
    conversation: list[ChatMessage] = field(default_factory=list)  # type: ignore
    instruction: str = ""
    task: ChatMessage | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class _GroupChatResponseMessage:
    """内部: 参加者からオーケストレーターに返されるResponseエンベロープ。"""

    agent_name: str
    message: ChatMessage


@dataclass
class _GroupChatTurn:
    """内部: マネージャーと参加者の会話における単一ターンを表す。"""

    speaker: str
    role: str
    message: ChatMessage


@dataclass
class GroupChatDirective:
    """グループチャットマネージャー実装から発行される指示。"""

    agent_name: str | None = None
    instruction: str | None = None
    metadata: dict[str, Any] | None = None
    finish: bool = False
    final_message: ChatMessage | None = None


# endregion region Manager callable


GroupChatStateSnapshot = Mapping[str, Any]
_GroupChatManagerFn = Callable[[GroupChatStateSnapshot], Awaitable[GroupChatDirective]]


async def _maybe_await(value: Any) -> Any:
    """値がawaitableならawaitし、そうでなければそのまま返す。"""
    if inspect.isawaitable(value):
        return await value
    return value


_GroupChatParticipantPipeline: TypeAlias = Sequence[Executor]


@dataclass
class _GroupChatConfig:
    """内部: ワークフロー組み立て時にファクトリーに渡される設定。

    Attributes:
        manager: オーケストレーション決定を担当するマネージャーインスタンス（カスタムファクトリーが処理する場合はNone）
        manager_name: 会話履歴に表示されるマネージャー名
        participants: 参加者名から仕様へのマッピング
        max_rounds: 無限ループ防止のためのマネージャー選択ラウンドのオプション制限
        orchestrator: オーケストレーターExecutorインスタンス（ビルド時に設定）

    """

    manager: _GroupChatManagerFn | None
    manager_name: str
    participants: Mapping[str, GroupChatParticipantSpec]
    max_rounds: int | None = None
    orchestrator: Executor | None = None
    participant_aliases: dict[str, str] = field(default_factory=dict)  # type: ignore[type-arg]
    participant_executors: dict[str, Executor] = field(default_factory=dict)  # type: ignore[type-arg]


# endregion region Default participant factory

_GroupChatOrchestratorFactory: TypeAlias = Callable[[_GroupChatConfig], Executor]
_InterceptorSpec: TypeAlias = tuple[Callable[[_GroupChatConfig], Executor], Callable[[Any], bool]]


def _default_participant_factory(
    spec: GroupChatParticipantSpec,
    wiring: _GroupChatConfig,
) -> _GroupChatParticipantPipeline:
    """ワークフローグラフ内の参加者パイプラインノードを構築するデフォルトファクトリー。

    AgentProtocol参加者には単一のAgentExecutorノードを作成し、カスタム参加者にはパススルーExecutorを作成する。
    グループチャットのエンベロープとエージェントランタイム間の変換はオーケストレーター内で処理されるため、専用のイングレス/イーグレスアダプターは不要。

    Args:
        spec: 名前、インスタンス、説明を含む参加者仕様
        wiring: キャッシュされたExecutorにアクセスするためのGroupChatWiring設定

    Returns:
        実行順序で参加者パイプラインを表すExecutorのシーケンス

    Behavior:
        - AgentProtocol参加者は決定論的ID付きのAgentExecutorでラップされる
        - Executor参加者は追加アダプターなしで直接配線される

    """
    participant = spec.participant
    if isinstance(participant, Executor):
        return (participant,)

    cached = wiring.participant_executors.get(spec.name)
    if cached is not None:
        return (cached,)

    agent_executor = wrap_participant(participant, executor_id=f"groupchat_agent:{spec.name}")
    return (agent_executor,)


# endregion


# region Default orchestrator


class GroupChatOrchestratorExecutor(BaseGroupChatOrchestrator):
    """Executor that orchestrates a group chat between multiple participants using a manager.

    This is the central runtime state machine that drives multi-agent conversations. It
    maintains conversation state, delegates speaker selection to a manager, routes messages
    to participants, and collects responses in a loop until the manager signals completion.

    Core responsibilities:
    - Accept initial input as str, ChatMessage, or list[ChatMessage]
    - Maintain conversation history and turn tracking
    - Query manager for next action (select participant or finish)
    - Route requests to selected participants using AgentExecutorRequest or GroupChatRequestMessage
    - Collect participant responses and append to conversation
    - Enforce optional round limits to prevent infinite loops
    - Yield final completion message and transition to idle state

    State management:
    - _conversation: Growing list of all messages (user, manager, agents)
    - _history: Turn-by-turn record with speaker attribution and roles
    - _task_message: Original user task extracted from input
    - _pending_agent: Name of agent currently processing a request
    - _round_index: Count of manager selection rounds for limit enforcement

    Manager interaction:
    The orchestrator builds immutable state snapshots and passes them to the manager
    callable. The manager returns a GroupChatDirective indicating either:
    - Next participant to speak (with optional instruction)
    - Finish signal (with optional final message)

    Message flow topology:
        User input -> orchestrator -> manager -> orchestrator -> participant -> orchestrator
        (loops until manager returns finish directive)

    Why this design:
    - Separates orchestration logic (this class) from selection logic (manager)
    - Manager is stateless and testable in isolation
    - Orchestrator handles all state mutations and message routing
    - Broadcast routing to participants keeps graph structure simple

    Args:
        manager: Callable that selects the next participant or finishes based on state snapshot
        participants: Mapping of participant names to descriptions (for manager context)
        manager_name: Display name for manager in conversation history
        max_rounds: Optional limit on manager selection rounds (None = unlimited)
        executor_id: Optional custom ID for observability (auto-generated if not provided)
    """

    def __init__(
        self,
        manager: _GroupChatManagerFn,
        *,
        participants: Mapping[str, str],
        manager_name: str,
        max_rounds: int | None = None,
        executor_id: str | None = None,
    ) -> None:
        super().__init__(executor_id or f"groupchat_orchestrator_{uuid4().hex[:8]}")
        self._manager = manager
        self._participants = dict(participants)
        self._manager_name = manager_name
        self._max_rounds = max_rounds
        self._history: list[_GroupChatTurn] = []
        self._task_message: ChatMessage | None = None
        self._pending_agent: str | None = None
        # Stashes the initial conversation list until _handle_task_message normalizes it into _conversation.
        self._pending_initial_conversation: list[ChatMessage] | None = None

    def _get_author_name(self) -> str:
        """Get the manager name for orchestrator-generated messages."""
        return self._manager_name

    def _build_state(self) -> GroupChatStateSnapshot:
        """Build a snapshot of current orchestration state for the manager.

        Packages conversation history, participant metadata, and round tracking into
        an immutable mapping that the manager uses to make speaker selection decisions.

        Returns:
            Mapping containing all context needed for manager decision-making

        Raises:
            RuntimeError: If called before task message initialization (defensive check)

        When this is called:
            - After initial input is processed (first manager query)
            - After each participant response (subsequent manager queries)
        """
        if self._task_message is None:
            raise RuntimeError("GroupChatOrchestratorExecutor state not initialized with task message.")
        snapshot: dict[str, Any] = {
            "task": self._task_message,
            "participants": dict(self._participants),
            "conversation": tuple(self._conversation),
            "history": tuple(self._history),
            "pending_agent": self._pending_agent,
            "round_index": self._round_index,
        }
        return MappingProxyType(snapshot)

    def _snapshot_pattern_metadata(self) -> dict[str, Any]:
        """Serialize GroupChat-specific state for checkpointing.

        Returns:
            Dict with participants, manager name, history, and pending agent
        """
        return {
            "participants": dict(self._participants),
            "manager_name": self._manager_name,
            "pending_agent": self._pending_agent,
            "task_message": self._task_message.to_dict() if self._task_message else None,
            "history": [
                {"speaker": turn.speaker, "role": turn.role, "message": turn.message.to_dict()}
                for turn in self._history
            ],
        }

    def _restore_pattern_metadata(self, metadata: dict[str, Any]) -> None:
        """Restore GroupChat-specific state from checkpoint.

        Args:
            metadata: Pattern-specific state dict
        """
        if "participants" in metadata:
            self._participants = dict(metadata["participants"])
        if "manager_name" in metadata:
            self._manager_name = metadata["manager_name"]
        if "pending_agent" in metadata:
            self._pending_agent = metadata["pending_agent"]
        task_msg = metadata.get("task_message")
        if task_msg:
            self._task_message = ChatMessage.from_dict(task_msg)
        if "history" in metadata:
            self._history = [
                _GroupChatTurn(
                    speaker=turn["speaker"],
                    role=turn["role"],
                    message=ChatMessage.from_dict(turn["message"]),
                )
                for turn in metadata["history"]
            ]

    async def _apply_directive(
        self,
        directive: GroupChatDirective,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Execute a manager directive by either finishing the workflow or routing to a participant.

        This is the core routing logic that interprets manager decisions. It handles two cases:
        1. Finish directive: append final message, update state, yield output, become idle
        2. Agent selection: build request envelope, route to participant, increment round counter

        Args:
            directive: Manager's decision (finish or select next participant)
            ctx: Workflow context for sending messages and yielding output

        Behavior for finish directive:
            - Uses provided final_message or creates default completion message
            - Ensures author_name is set to manager for attribution
            - Appends to conversation and history for complete record
            - Yields message as workflow output
            - Orchestrator becomes idle (no further processing)

        Behavior for agent selection:
            - Validates agent_name exists in participants
            - Optionally appends manager instruction as USER message
            - Prepares full conversation context for the participant
            - Routes request directly to the participant entry executor
            - Increments round counter and enforces max_rounds if configured

        Round limit enforcement:
            If max_rounds is reached, recursively calls _apply_directive with a finish
            directive to gracefully terminate the conversation.

        Raises:
            ValueError: If directive lacks agent_name when finish=False, or if
                       agent_name doesn't match any participant
        """
        if directive.finish:
            final_message = directive.final_message
            if final_message is None:
                final_message = self._create_completion_message(
                    text="Completed without final summary.",
                    reason="no summary provided",
                )
            final_message = ensure_author(final_message, self._manager_name)

            self._conversation.extend((final_message,))
            self._history.append(_GroupChatTurn(self._manager_name, "manager", final_message))
            self._pending_agent = None
            await ctx.yield_output(final_message)
            return

        agent_name = directive.agent_name
        if not agent_name:
            raise ValueError("Directive must include agent_name when finish is False.")
        if agent_name not in self._participants:
            raise ValueError(f"Manager selected unknown participant '{agent_name}'.")

        instruction = directive.instruction or ""
        conversation = list(self._conversation)
        if instruction:
            manager_message = ensure_author(
                self._create_completion_message(text=instruction, reason="instruction"),
                self._manager_name,
            )
            conversation.extend((manager_message,))
            self._conversation.extend((manager_message,))
            self._history.append(_GroupChatTurn(self._manager_name, "manager", manager_message))

        self._pending_agent = agent_name
        self._increment_round()

        # Use inherited routing method from BaseGroupChatOrchestrator
        await self._route_to_participant(
            participant_name=agent_name,
            conversation=conversation,
            ctx=ctx,
            instruction=instruction,
            task=self._task_message,
            metadata=directive.metadata,
        )

        if self._check_round_limit():
            await self._apply_directive(
                GroupChatDirective(
                    finish=True,
                    final_message=self._create_completion_message(
                        text="Conversation halted after reaching manager round limit.",
                        reason="max_rounds reached",
                    ),
                ),
                ctx,
            )

    async def _ingest_participant_message(
        self,
        participant_name: str,
        message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Common response ingestion logic shared by agent and custom participants."""
        if participant_name not in self._participants:
            raise ValueError(f"Received response from unknown participant '{participant_name}'.")

        message = ensure_author(message, participant_name)
        self._conversation.extend((message,))
        self._history.append(_GroupChatTurn(participant_name, "agent", message))
        self._pending_agent = None

        if self._check_round_limit():
            await ctx.yield_output(
                self._create_completion_message(
                    text="Conversation halted after reaching manager round limit.",
                    reason="max_rounds reached after response",
                )
            )
            return

        directive = await self._manager(self._build_state())
        await self._apply_directive(directive, ctx)

    @staticmethod
    def _extract_agent_message(response: AgentExecutorResponse, participant_name: str) -> ChatMessage:
        """Select the final assistant message from an AgentExecutor response."""
        from ._orchestrator_helpers import create_completion_message

        final_message: ChatMessage | None = None
        candidate_sequences: tuple[Sequence[ChatMessage] | None, ...] = (
            response.agent_run_response.messages,
            response.full_conversation,
        )
        for sequence in candidate_sequences:
            if not sequence:
                continue
            for candidate in reversed(sequence):
                if candidate.role == Role.ASSISTANT:
                    final_message = candidate
                    break
            if final_message is not None:
                break

        if final_message is None:
            final_message = create_completion_message(
                text="",
                author_name=participant_name,
                reason="empty response",
            )
        return ensure_author(final_message, participant_name)

    async def _handle_task_message(
        self,
        task_message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Initialize orchestrator state and start the manager-directed conversation loop.

        This internal method is called by all public handlers (str, ChatMessage, list[ChatMessage])
        after normalizing their input. It initializes conversation state, queries the manager
        for the first action, and applies the resulting directive.

        Args:
            task_message: The primary user task message (extracted or provided directly)
            ctx: Workflow context for sending messages and yielding output

        Behavior:
            - Sets task_message for manager context
            - Initializes conversation from pending_initial_conversation if present
            - Otherwise starts fresh with just the task message
            - Builds turn history with speaker attribution
            - Resets pending_agent and round_index
            - Queries manager for first action
            - Applies directive to start the conversation loop

        State initialization:
            - _conversation: Full message list for context
            - _history: Turn-by-turn record with speaker names and roles
            - _pending_agent: None (no active request)
            - _round_index: 0 (first manager query)

        Why pending_initial_conversation exists:
            The handle_conversation handler supplies an explicit task (the first message in
            the list) but still forwards the entire conversation for context. The full list is
            stashed in _pending_initial_conversation to preserve all context when initializing state.
        """
        self._task_message = task_message
        if self._pending_initial_conversation:
            initial_conversation = list(self._pending_initial_conversation)
            self._pending_initial_conversation = None
            self._conversation = initial_conversation
            self._history = [
                _GroupChatTurn(
                    msg.author_name or msg.role.value,
                    msg.role.value,
                    msg,
                )
                for msg in initial_conversation
            ]
        else:
            self._conversation = [task_message]
            self._history = [_GroupChatTurn("user", "user", task_message)]
        self._pending_agent = None
        self._round_index = 0
        directive = await self._manager(self._build_state())
        await self._apply_directive(directive, ctx)

    @handler
    async def handle_str(
        self,
        task: str,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handler for string input as workflow entry point.

        Wraps the string in a USER role ChatMessage and delegates to _handle_task_message.

        Args:
            task: Plain text task description from user
            ctx: Workflow context

        Usage:
            workflow.run("Write a blog post about AI agents")
        """
        await self._handle_task_message(ChatMessage(role=Role.USER, text=task), ctx)

    @handler
    async def handle_chat_message(
        self,
        task_message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handler for ChatMessage input as workflow entry point.

        Directly delegates to _handle_task_message for state initialization.

        Args:
            task_message: Structured chat message from user (may include metadata, role, etc.)
            ctx: Workflow context

        Usage:
            workflow.run(ChatMessage(role=Role.USER, text="Analyze this data"))
        """
        await self._handle_task_message(task_message, ctx)

    @handler
    async def handle_conversation(
        self,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handler for conversation history as workflow entry point.

        Accepts a pre-existing conversation and uses the first message in the list as the task.
        Preserves the full conversation for state initialization.

        Args:
            conversation: List of chat messages (system, user, assistant)
            ctx: Workflow context

        Raises:
            ValueError: If conversation list is empty

        Behavior:
            - Validates conversation is non-empty
            - Clones conversation to avoid mutation
            - Extracts task message (most recent USER message)
            - Stashes full conversation in _pending_initial_conversation
            - Delegates to _handle_task_message for initialization

        Usage:
            existing_messages = [
                ChatMessage(role=Role.SYSTEM, text="You are an expert"),
                ChatMessage(role=Role.USER, text="Help me with this task")
            ]
            workflow.run(existing_messages)
        """
        if not conversation:
            raise ValueError("GroupChat workflow requires at least one chat message.")
        self._pending_initial_conversation = list(conversation)
        task_message = latest_user_message(conversation)
        await self._handle_task_message(task_message, ctx)

    @handler
    async def handle_agent_response(
        self,
        response: _GroupChatResponseMessage,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handle responses from custom participant executors."""
        await self._ingest_participant_message(response.agent_name, response.message, ctx)

    @handler
    async def handle_agent_executor_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[AgentExecutorRequest | _GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handle direct AgentExecutor responses."""
        participant_name = self._registry.get_participant_name(response.executor_id)
        if participant_name is None:
            logger.debug(
                "Ignoring response from unregistered agent executor '%s'.",
                response.executor_id,
            )
            return
        message = self._extract_agent_message(response, participant_name)
        await self._ingest_participant_message(participant_name, message, ctx)


def _default_orchestrator_factory(wiring: _GroupChatConfig) -> Executor:
    """Default factory for creating the GroupChatOrchestratorExecutor instance.

    This is the internal implementation used by GroupChatBuilder to instantiate the
    orchestrator. It extracts participant descriptions from the wiring configuration
    and passes them to the orchestrator for manager context.

    Args:
        wiring: Complete workflow configuration assembled by the builder

    Returns:
        Initialized GroupChatOrchestratorExecutor ready to coordinate the conversation

    Behavior:
        - Extracts participant names and descriptions for manager context
        - Forwards manager instance, manager name, and max_rounds settings
        - Allows orchestrator to auto-generate its executor ID

    Why descriptions are extracted:
        The manager needs participant descriptions (not full specs) to make informed
        selection decisions. The orchestrator doesn't need participant instances directly
        since routing is handled by the workflow graph.

    Raises:
        RuntimeError: If manager is None (should not happen when using default factory)
    """
    if wiring.manager is None:
        raise RuntimeError("Default orchestrator factory requires a manager to be set")

    return GroupChatOrchestratorExecutor(
        manager=wiring.manager,
        participants={name: spec.description for name, spec in wiring.participants.items()},
        manager_name=wiring.manager_name,
        max_rounds=wiring.max_rounds,
    )


def group_chat_orchestrator(factory: _GroupChatOrchestratorFactory | None = None) -> _GroupChatOrchestratorFactory:
    """Return a callable orchestrator factory, defaulting to the built-in implementation."""
    return factory or _default_orchestrator_factory


def assemble_group_chat_workflow(
    *,
    wiring: _GroupChatConfig,
    participant_factory: Callable[[GroupChatParticipantSpec, _GroupChatConfig], _GroupChatParticipantPipeline],
    orchestrator_factory: _GroupChatOrchestratorFactory = _default_orchestrator_factory,
    interceptors: Sequence[_InterceptorSpec] | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
    builder: WorkflowBuilder | None = None,
    return_builder: bool = False,
) -> Workflow | tuple[WorkflowBuilder, Executor]:
    """Build the workflow graph shared by group-chat style orchestrators."""
    interceptor_specs = interceptors or ()

    orchestrator = wiring.orchestrator or orchestrator_factory(wiring)
    wiring.orchestrator = orchestrator

    workflow_builder = builder or WorkflowBuilder()
    workflow_builder = workflow_builder.set_start_executor(orchestrator)

    for name, spec in wiring.participants.items():
        pipeline = list(participant_factory(spec, wiring))
        if not pipeline:
            raise ValueError(
                f"Participant factory returned an empty pipeline for '{name}'. "
                "Provide at least one executor per participant."
            )
        entry_executor = pipeline[0]
        exit_executor = pipeline[-1]

        register_entry = getattr(orchestrator, "register_participant_entry", None)
        if callable(register_entry):
            register_entry(
                name,
                entry_id=entry_executor.id,
                is_agent=not isinstance(spec.participant, Executor),
            )

        workflow_builder = workflow_builder.add_edge(orchestrator, entry_executor)
        for upstream, downstream in itertools.pairwise(pipeline):
            workflow_builder = workflow_builder.add_edge(upstream, downstream)
        if exit_executor is not orchestrator:
            workflow_builder = workflow_builder.add_edge(exit_executor, orchestrator)

    for factory, condition in interceptor_specs:
        interceptor_executor = factory(wiring)
        workflow_builder = workflow_builder.add_edge(orchestrator, interceptor_executor, condition=condition)
        workflow_builder = workflow_builder.add_edge(interceptor_executor, orchestrator)

    if checkpoint_storage is not None:
        workflow_builder = workflow_builder.with_checkpointing(checkpoint_storage)

    if return_builder:
        return workflow_builder, orchestrator
    return workflow_builder.build()


# endregion region Builder


class GroupChatBuilder:
    r"""マネージャーが指示するグループチャットワークフローのための高レベルビルダーで、動的なオーケストレーションを提供します。

    GroupChatは、次に話す参加者を選択するマネージャーを使ってマルチエージェントの会話を調整します。マネージャーは単純なPython関数（select_speakers）か、LLMベースのセレクター（set_prompt_based_manager）であり、これら二つの方法は相互排他的です。

    **Core Workflow:**
    1. 参加者を定義：エージェントのリスト（.nameを使用）または名前からエージェントへの辞書マッピング
    2. スピーカー選択の設定：select_speakers() または set_prompt_based_manager()（両方は不可）
    3. 任意：ラウンド制限、チェックポイント、終了条件の設定
    4. ワークフローの構築と実行

    **Speaker Selection Patterns:**

    *Pattern 1: シンプルな関数ベースの選択（推奨）*

    .. code-block:: python

    def select_next_speaker(state: GroupChatStateSnapshot) -> str | None:
        # stateには task, participants, conversation, history, round_index が含まれる
        if state["round_index"] >= 5:
            return None  # 終了
        last_speaker = state["history"][-1].speaker if state["history"] else None
        if last_speaker == "researcher":
            return "writer"
        return "researcher"


    workflow = (
        GroupChatBuilder()
        .select_speakers(select_next_speaker)
        .participants([researcher_agent, writer_agent])  # agent.nameを使用
        .build()
    )

    *Pattern 2: LLMベースの選択*

    .. code-block:: python

    from agent_framework.azure import AzureOpenAIChatClient

    workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(chat_client=AzureOpenAIChatClient(), display_name="Coordinator")
        .participants([researcher, writer])  # または辞書形式: researcher=r, writer=w
        .with_max_rounds(10)
        .build()
    )

    **Participant Specification:**

    参加者の指定方法は二通りあります：
    - リスト形式：``[agent1, agent2]`` - participant名は ``agent.name`` 属性を使用
    - 辞書形式：``{name1: agent1, name2: agent2}`` - 明示的な名前指定
    - キーワード形式：``participants(name1=agent1, name2=agent2)`` - 明示的な名前指定

    **State Snapshot Structure:**

    select_speakersに渡されるGroupChatStateSnapshotには以下が含まれます：
    - ``task``: ChatMessage - 元のユーザータスク
    - ``participants``: dict[str, str] - 参加者名から説明へのマッピング
    - ``conversation``: tuple[ChatMessage, ...] - 会話の全履歴
    - ``history``: tuple[GroupChatTurn, ...] - 発話者情報付きのターンごとの記録
    - ``round_index``: int - これまでのマネージャー選択ラウンド数
    - ``pending_agent``: str | None - 現在処理中のエージェント名（あれば）

    **Important Constraints:**
    - select_speakers() と set_prompt_based_manager() は併用不可。どちらか一方を選択
    - 参加者名はユニークであること
    - リスト形式使用時は、エージェントに空でない ``name`` 属性が必要

    """

    def __init__(
        self,
        *,
        _orchestrator_factory: _GroupChatOrchestratorFactory | None = None,
        _participant_factory: Callable[[GroupChatParticipantSpec, _GroupChatConfig], _GroupChatParticipantPipeline]
        | None = None,
    ) -> None:
        """GroupChatBuilderを初期化します。

        Args:
        _orchestrator_factory: カスタムオーケストレーター実装のための内部拡張ポイント。
        Magenticで使用。パブリックAPIの一部ではなく、変更される可能性があります。
        _participant_factory: カスタム参加者パイプラインのための内部拡張ポイント。
        Magenticで使用。パブリックAPIの一部ではなく、変更される可能性があります。

        """
        self._participants: dict[str, AgentProtocol | Executor] = {}
        self._participant_metadata: dict[str, Any] | None = None
        self._manager: _GroupChatManagerFn | None = None
        self._manager_name: str = "manager"
        self._checkpoint_storage: CheckpointStorage | None = None
        self._max_rounds: int | None = None
        self._interceptors: list[_InterceptorSpec] = []
        self._orchestrator_factory = group_chat_orchestrator(_orchestrator_factory)
        self._participant_factory = _participant_factory or _default_participant_factory

    def _set_manager_function(
        self,
        manager: _GroupChatManagerFn,
        display_name: str | None,
    ) -> "GroupChatBuilder":
        if self._manager is not None:
            raise ValueError(
                "GroupChatBuilder already has a manager configured. "
                "Call select_speakers(...) or set_prompt_based_manager(...) at most once."
            )
        resolved_name = display_name or getattr(manager, "name", None) or "manager"
        self._manager = manager
        self._manager_name = resolved_name
        return self

    def set_prompt_based_manager(
        self,
        chat_client: ChatClientProtocol,
        *,
        instructions: str | None = None,
        display_name: str | None = None,
    ) -> "GroupChatBuilder":
        r"""LLMチャットクライアントによって駆動されるデフォルトのプロンプトベースマネージャーを設定します。

        マネージャーは会話状態、タスク、参加者の説明に基づいて選択決定を行い、参加者を調整します。構造化出力（ManagerDirectiveModel）を使用して決定の信頼性の高い解析を保証します。

        Args:
        chat_client: コーディネーターLLMを実行するためのチャット完了クライアント。
        instructions: コーディネーターの意思決定を誘導するシステム指示。
        指定しない場合はDEFAULT_MANAGER_INSTRUCTIONSを使用。これらの指示はタスク説明、参加者リスト、構造化出力形式と組み合わされ、LLMが次のスピーカー選択や会話完了を行うのを導きます。
        display_name: マネージャーメッセージの会話表示名（任意）。

        Returns:
        Fluentチェーンのためのself。

        Note:
        このメソッドと :meth:`set_speaker_selector` を同時に呼び出すことはできません。どちらかを選択してください。

        Example:

        .. code-block:: python

        from agent_framework import GroupChatBuilder, DEFAULT_MANAGER_INSTRUCTIONS

        custom_instructions = (
        DEFAULT_MANAGER_INSTRUCTIONS + "\n\nPrioritize the researcher for data analysis tasks."
        )

        workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(chat_client, instructions=custom_instructions, display_name="Coordinator")
        .participants(researcher=researcher, writer=writer)
        .build()
        )

        """
        manager = _PromptBasedGroupChatManager(
            chat_client,
            instructions=instructions,
            name=display_name,
        )
        return self._set_manager_function(manager, display_name)

    def select_speakers(
        self,
        selector: (
            Callable[[GroupChatStateSnapshot], Awaitable[str | None]] | Callable[[GroupChatStateSnapshot], str | None]
        ),
        *,
        display_name: str | None = None,
        final_message: ChatMessage | str | Callable[[GroupChatStateSnapshot], Any] | None = None,
    ) -> "GroupChatBuilder":
        """グループチャットの状態を調べる純粋関数を使ってスピーカー選択を設定します。

        これはGroupChatのオーケストレーションフローを制御する主要な方法です。セレクター関数は現在の会話状態の不変スナップショットを受け取り、次に話す参加者の名前を返すか、会話を終了するためにNoneを返します。

        セレクター関数のシグネチャ:
        def select_next_speaker(state: GroupChatStateSnapshot) -> str | None:
        # stateには task, participants, conversation, history, round_index が含まれる
        # 続行する参加者名を返すか、終了する場合はNoneを返す
        ...

        Args:
        selector: GroupChatStateSnapshotを受け取り、次のスピーカーの名前（str）を返す関数。会話を終了する場合はNone。同期または非同期関数可。
        display_name: オーケストレーターのメッセージに表示される任意の名前（デフォルトは"manager"）。
        final_message: セレクターがNoneを返したときに出力される最終メッセージまたはファクトリー（デフォルトはマネージャー作成の"Conversation completed."）。

        Returns:
        Fluentチェーンのためのself。

        Example:

        .. code-block:: python

        def select_next_speaker(state: GroupChatStateSnapshot) -> str | None:
        if state["round_index"] >= 3:
            return None  # 3ラウンド後に終了
        last_speaker = state["history"][-1].speaker if state["history"] else None
        if last_speaker == "researcher":
            return "writer"
        return "researcher"


        workflow = (
        GroupChatBuilder()
        .select_speakers(select_next_speaker)
        .participants(researcher=researcher_agent, writer=writer_agent)
        .build()
        )

        Note:
        set_prompt_based_manager()と併用できません。どちらか一方のオーケストレーション戦略を選択してください。

        """
        manager_name = display_name or "manager"
        adapter = _SpeakerSelectorAdapter(
            selector,
            manager_name=manager_name,
            final_message=final_message,
        )
        return self._set_manager_function(adapter, display_name)

    def participants(
        self,
        participants: Mapping[str, AgentProtocol | Executor] | Sequence[AgentProtocol | Executor] | None = None,
        /,
        **named_participants: AgentProtocol | Executor,
    ) -> "GroupChatBuilder":
        """このグループチャットワークフローの参加者を定義します。

        AgentProtocolインスタンス（AgentExecutorとして自動ラップされる）またはExecutorインスタンスを受け入れます。
        名前→参加者のマッピングを提供して明示的に制御するか、シーケンスを渡してエージェントのname属性（またはexecutorのid）から名前を推測します。

        Args:
        participants: 参加者定義のマッピングまたはシーケンス（任意）
        **named_participants: 名前からエージェント/Executorインスタンスへのキーワード引数マッピング

        Returns:
        Fluentチェーンのためのself

        Raises:
        ValueError: 参加者が空、名前が重複、または名前が空文字列の場合

        Usage:

        .. code-block:: python

        from agent_framework import GroupChatBuilder

        workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(chat_client)
        .participants([writer_agent, reviewer_agent])
        .build()
        )

        """
        combined: dict[str, AgentProtocol | Executor] = {}

        def _add(name: str, participant: AgentProtocol | Executor) -> None:
            if not name:
                raise ValueError("participant names must be non-empty strings")
            if name in combined or name in self._participants:
                raise ValueError(f"Duplicate participant name '{name}' supplied.")
            combined[name] = participant

        if participants:
            if isinstance(participants, Mapping):
                for name, participant in participants.items():
                    _add(name, participant)
            else:
                for participant in participants:
                    inferred_name: str
                    if isinstance(participant, Executor):
                        inferred_name = participant.id
                    else:
                        name_attr = getattr(participant, "name", None)
                        if not name_attr:
                            raise ValueError(
                                "Agent participants supplied via sequence must define a non-empty 'name' attribute."
                            )
                        inferred_name = str(name_attr)
                    _add(inferred_name, participant)

        for name, participant in named_participants.items():
            _add(name, participant)

        if not combined:
            raise ValueError("participants cannot be empty")

        for name, participant in combined.items():
            self._participants[name] = participant
        self._participant_metadata = None
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "GroupChatBuilder":
        """提供されたストレージを使って構築されたワークフローのチェックポイント機能を有効にします。

        チェックポイントにより、ワークフローは状態を永続化し、中断ポイントから再開可能となり、長時間の会話や障害復旧を可能にします。

        Args:
        checkpoint_storage: ワークフロー状態を永続化するためのストレージ実装

        Returns:
        Fluentチェーンのためのself

        Usage:

        .. code-block:: python

        from agent_framework import GroupChatBuilder, MemoryCheckpointStorage

        storage = MemoryCheckpointStorage()
        workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(chat_client)
        .participants(agent1=agent1, agent2=agent2)
        .with_checkpointing(storage)
        .build()
        )

        """
        self._checkpoint_storage = checkpoint_storage
        return self

    def with_request_handler(
        self,
        handler: Callable[[_GroupChatConfig], Executor] | Executor,
        *,
        condition: Callable[[Any], bool],
    ) -> "GroupChatBuilder":
        """特別なリクエスト用のExecutorを作成するインターセプターファクトリーを登録します。

        Args:
        handler: wiringを受け取りExecutorを返すCallable、または事前構築済みのExecutor
        condition: インターセプターが処理すべきオーケストレーターのメッセージを判定するフィルター

        Returns:
        Fluentチェーンのためのself

        """
        factory: Callable[[_GroupChatConfig], Executor]
        if isinstance(handler, Executor):
            executor = handler

            def _factory(_: _GroupChatConfig) -> Executor:
                return executor

            factory = _factory
        else:
            factory = handler

        self._interceptors.append((factory, condition))
        return self

    def with_max_rounds(self, max_rounds: int | None) -> "GroupChatBuilder":
        """無限会話を防ぐためにマネージャーの最大ラウンド数を設定します。

        ラウンド制限に達すると、ワークフローはデフォルトの完了メッセージで自動的に完了します。Noneを設定すると無制限になります。

        Args:
        max_rounds: マネージャー選択ラウンドの最大数、または無制限の場合はNone

        Returns:
        Fluentチェーンのためのself

        Usage:

        .. code-block:: python

        from agent_framework import GroupChatBuilder

        # 15ラウンドに制限
        workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(chat_client)
        .participants(agent1=agent1, agent2=agent2)
        .with_max_rounds(15)
        .build()
        )

        # 無制限ラウンド
        workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(chat_client)
        .participants(agent1=agent1)
        .with_max_rounds(None)
        .build()
        )

        """
        self._max_rounds = max_rounds
        return self

    def _get_participant_metadata(self) -> dict[str, Any]:
        if self._participant_metadata is None:
            self._participant_metadata = prepare_participant_metadata(
                self._participants,
                executor_id_factory=lambda name, participant: (
                    participant.id if isinstance(participant, Executor) else f"groupchat_agent:{name}"
                ),
                description_factory=lambda name, participant: (
                    participant.id if isinstance(participant, Executor) else participant.__class__.__name__
                ),
            )
        return self._participant_metadata

    def _build_participant_specs(self) -> dict[str, GroupChatParticipantSpec]:
        metadata = self._get_participant_metadata()
        descriptions: Mapping[str, str] = metadata["descriptions"]
        specs: dict[str, GroupChatParticipantSpec] = {}
        for name, participant in self._participants.items():
            specs[name] = GroupChatParticipantSpec(
                name=name,
                participant=participant,
                description=descriptions[name],
            )
        return specs

    def build(self) -> Workflow:
        """グループチャットのワークフローを構築し検証します。

        オーケストレーター、参加者、およびそれらの相互接続を組み合わせて
        完全なワークフローグラフを作成します。オーケストレーターはスピーカー選択を
        マネージャーに委任し、リクエストを適切な参加者にルーティングし、
        その応答を収集して会話を継続または完了させます。

        Returns:
            実行準備が整った検証済みのWorkflowインスタンス

        Raises:
            ValueError: マネージャーまたは参加者が設定されていない場合（デフォルトファクトリ使用時）

        Wiring pattern:
        - Orchestratorは初期入力（str, ChatMessage, またはlist[ChatMessage]）を受け取る
        - Orchestratorは次のアクション（参加者選択または終了）についてマネージャーに問い合わせる
        - 参加者が選択された場合：リクエストは直接参加者のエントリノードにルーティングされる
        - 参加者パイプライン：AgentExecutor（エージェント用）またはカスタムエグゼキューターチェーン
        - 参加者の応答はオーケストレーターに戻る
        - オーケストレーターは状態を更新し再度マネージャーに問い合わせる
        - マネージャーが終了指示を返した場合：オーケストレーターは最終メッセージを返しアイドル状態になる

        Usage:

        .. code-block:: python

            from agent_framework import GroupChatBuilder

            # ワークフローを実行する
            workflow = (
                GroupChatBuilder()
                .set_prompt_based_manager(chat_client)
                .participants(agent1=agent1, agent2=agent2)
                .build()
            )
            async for message in workflow.run("Solve this problem collaboratively"):
                print(message.text)

        """
        # マネージャーはデフォルトのオーケストレーターファクトリを使用する場合にのみ必要です
        # カスタムファクトリ（例：MagenticBuilder）は、組み込みのマネージャーを持つ独自のオーケストレーターを提供します
        if self._manager is None and self._orchestrator_factory == _default_orchestrator_factory:
            raise ValueError("manager must be configured before build() when using default orchestrator")
        if not self._participants:
            raise ValueError("participants must be configured before build()")

        metadata = self._get_participant_metadata()
        participant_specs = self._build_participant_specs()
        wiring = _GroupChatConfig(
            manager=self._manager,
            manager_name=self._manager_name,
            participants=participant_specs,
            max_rounds=self._max_rounds,
            participant_aliases=metadata["aliases"],
            participant_executors=metadata["executors"],
        )

        result = assemble_group_chat_workflow(
            wiring=wiring,
            participant_factory=self._participant_factory,
            orchestrator_factory=self._orchestrator_factory,
            interceptors=self._interceptors,
            checkpoint_storage=self._checkpoint_storage,
        )
        if not isinstance(result, Workflow):
            raise TypeError("Expected Workflow from assemble_group_chat_workflow")
        return result


# endregion region Default manager implementation


DEFAULT_MANAGER_INSTRUCTIONS = """You are coordinating a team conversation to solve the user's task.
Your role is to orchestrate collaboration between multiple participants by selecting who speaks next.
Leverage each participant's unique expertise as described in their descriptions.
Have participants build on each other's contributions - earlier participants gather information,
later ones refine and synthesize.
Only finish the task after multiple relevant participants have contributed their expertise."""

DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT = """Return your decision using the following structure:
- next_agent: name of the participant who should act next (use null when finish is true)
- message: instruction for that participant (empty string if not needed)
- finish: boolean indicating if the task is complete
- final_response: when finish is true, provide the final answer to the user"""


class ManagerDirectiveModel(BaseModel):
    """構造化されたマネージャー指示出力のためのPydanticモデル。"""

    next_agent: str | None = Field(
        default=None,
        description="Name of the participant who should act next (null when finish is true)",
    )
    message: str = Field(
        default="",
        description="Instruction for the selected participant",
    )
    finish: bool = Field(
        default=False,
        description="Whether the task is complete",
    )
    final_response: str | None = Field(
        default=None,
        description="Final answer to the user when finish is true",
    )


class _PromptBasedGroupChatManager:
    """LLMを利用したマネージャーで、構造化出力を通じて指示を生成します。

    これはグループチャットワークフローのデフォルトのマネージャー実装です。会話状態、参加者の説明、
    カスタム指示に基づいてスピーカー選択の決定をLLMが行います。

    調整戦略:
    - 会話履歴全体を含む不変の状態スナップショットを受け取る
    - システムプロンプトを指示、タスク、参加者説明でフォーマットする
    - 会話コンテキストを追加し、構造化出力（Pydanticモデル）を使用して信頼性の高い解析を行う
    - LLMの応答をGroupChatDirectiveに変換する

    柔軟性:
    - カスタム指示によりドメイン固有の調整戦略が可能
    - 参加者の説明がLLMの選択ロジックを導く
    - 構造化出力により信頼性の高い解析を保証（正規表現や壊れやすいプロンプトを使用しない）

    例示的な調整パターン:
    - ラウンドロビン：「参加者を順番に回す」
    - タスクベース：「現在のサブタスクに最適な参加者を選択する」
    - 依存関係認識：「研究者がデータを提供した後にのみアナリストを呼び出す」

    Args:
        chat_client: LLM推論用のChatClientProtocol実装
        instructions: カスタムシステム指示（デフォルトはDEFAULT_MANAGER_INSTRUCTIONS）。
                     これらの指示はタスク、参加者リスト、
                     構造化出力フォーマット（ManagerDirectiveModel）と組み合わせて会話を調整します。
        name: 会話履歴に表示されるマネージャーの名前

    Raises:
        RuntimeError: LLMの応答が指示ペイロードに解析できない場合
                     finish=Falseでnext_agentが欠落している場合
                     選択されたエージェントが参加者に存在しない場合

    """

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        *,
        instructions: str | None = None,
        name: str | None = None,
    ) -> None:
        self._chat_client = chat_client
        self._instructions = instructions or DEFAULT_MANAGER_INSTRUCTIONS
        self._name = name or "GroupChatManager"

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, state: GroupChatStateSnapshot) -> GroupChatDirective:
        participants = state["participants"]
        task_message = state["task"]
        conversation = state["conversation"]

        participants_section = "\n".join(f"- {agent}: {description}" for agent, description in participants.items())

        system_message = ChatMessage(
            role=Role.SYSTEM,
            text=(
                f"{self._instructions}\n\n"
                f"Task:\n{task_message.text}\n\n"
                f"Participants:\n{participants_section}\n\n"
                f"{DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT}"
            ),
        )

        messages: list[ChatMessage] = [system_message, *conversation]

        response = await self._chat_client.get_response(messages, response_format=ManagerDirectiveModel)

        directive_model: ManagerDirectiveModel
        if response.value is not None:
            if isinstance(response.value, ManagerDirectiveModel):
                directive_model = response.value
            elif isinstance(response.value, str):
                directive_model = ManagerDirectiveModel.model_validate_json(response.value)
            elif isinstance(response.value, dict):
                directive_model = ManagerDirectiveModel.model_validate(response.value)  # type: ignore[arg-type]
            else:
                raise RuntimeError(f"Unexpected response.value type: {type(response.value)}")
        elif response.messages:
            text = response.messages[-1].text or "{}"
            directive_model = ManagerDirectiveModel.model_validate_json(text)
        else:
            raise RuntimeError("LLM response did not contain structured output.")

        if directive_model.finish:
            final_text = directive_model.final_response or ""
            return GroupChatDirective(
                finish=True,
                final_message=ChatMessage(
                    role=Role.ASSISTANT,
                    text=final_text,
                    author_name=self._name,
                ),
            )

        next_agent = directive_model.next_agent
        if not next_agent:
            raise RuntimeError("Manager directive missing next_agent while finish is False.")
        if next_agent not in participants:
            raise RuntimeError(f"Manager selected unknown participant '{next_agent}'.")

        return GroupChatDirective(
            agent_name=next_agent,
            instruction=directive_model.message or "",
        )


class _SpeakerSelectorAdapter:
    """単純なスピーカーセレクターを完全なマネージャー指示に変換するアダプター。"""

    def __init__(
        self,
        selector: Callable[[GroupChatStateSnapshot], Awaitable[Any]] | Callable[[GroupChatStateSnapshot], Any],
        *,
        manager_name: str,
        final_message: ChatMessage | str | Callable[[GroupChatStateSnapshot], Any] | None = None,
    ) -> None:
        self._selector = selector
        self._manager_name = manager_name
        self._final_message = final_message
        self.name = manager_name

    async def __call__(self, state: GroupChatStateSnapshot) -> GroupChatDirective:
        result = await _maybe_await(self._selector(state))
        if result is None:
            message = await self._resolve_final_message(state)
            return GroupChatDirective(finish=True, final_message=message)

        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            if not result:
                message = await self._resolve_final_message(state)
                return GroupChatDirective(finish=True, final_message=message)
            if len(result) != 1:  # type: ignore[arg-type]
                raise ValueError("Speaker selector must return a single participant name or None.")
            first_item = result[0]  # type: ignore[index]
            if not isinstance(first_item, str):
                raise TypeError("Speaker selector must return a participant name (str) or None.")
            result = first_item

        if not isinstance(result, str):
            raise TypeError("Speaker selector must return a participant name (str) or None.")

        return GroupChatDirective(agent_name=result)

    async def _resolve_final_message(self, state: GroupChatStateSnapshot) -> ChatMessage:
        final_message = self._final_message
        if callable(final_message):
            value = await _maybe_await(final_message(state))
        else:
            value = final_message

        if value is None:
            message = ChatMessage(
                role=Role.ASSISTANT,
                text="Conversation completed.",
                author_name=self._manager_name,
            )
        elif isinstance(value, ChatMessage):
            message = value
        else:
            message = ChatMessage(
                role=Role.ASSISTANT,
                text=str(value),
                author_name=self._manager_name,
            )

        if not message.author_name:
            patch = message.to_dict()
            patch["author_name"] = self._manager_name
            message = ChatMessage.from_dict(patch)
        return message


# endregion
