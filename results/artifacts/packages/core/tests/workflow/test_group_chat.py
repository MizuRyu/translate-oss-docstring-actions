# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable, Callable
from typing import Any

import pytest

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    GroupChatBuilder,
    GroupChatDirective,
    GroupChatStateSnapshot,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticContext,
    MagenticManagerBase,
    MagenticOrchestratorMessageEvent,
    Role,
    TextContent,
    Workflow,
    WorkflowOutputEvent,
)
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage
from agent_framework._workflows._group_chat import (
    GroupChatOrchestratorExecutor,
    _default_orchestrator_factory,  # type: ignore
    _GroupChatConfig,  # type: ignore
    _PromptBasedGroupChatManager,  # type: ignore
    _SpeakerSelectorAdapter,  # type: ignore
)
from agent_framework._workflows._magentic import (
    _MagenticProgressLedger,  # type: ignore
    _MagenticProgressLedgerItem,  # type: ignore
    _MagenticStartMessage,  # type: ignore
)


class StubAgent(BaseAgent):
    def __init__(self, agent_name: str, reply_text: str, **kwargs: Any) -> None:
        super().__init__(name=agent_name, description=f"Stub agent {agent_name}", **kwargs)
        self._reply_text = reply_text

    async def run(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        response = ChatMessage(role=Role.ASSISTANT, text=self._reply_text, author_name=self.name)
        return AgentRunResponse(messages=[response])

    def run_stream(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        async def _stream() -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(
                contents=[TextContent(text=self._reply_text)], role=Role.ASSISTANT, author_name=self.name
            )

        return _stream()


def make_sequence_selector() -> Callable[[GroupChatStateSnapshot], Any]:
    state_counter = {"value": 0}

    async def _selector(state: GroupChatStateSnapshot) -> str | None:
        participants = list(state["participants"].keys())
        step = state_counter["value"]
        if step == 0:
            state_counter["value"] = step + 1
            return participants[0]
        if step == 1 and len(participants) > 1:
            state_counter["value"] = step + 1
            return participants[1]
        return None

    _selector.name = "manager"  # type: ignore[attr-defined]
    return _selector


class StubMagenticManager(MagenticManagerBase):
    def __init__(self) -> None:
        super().__init__(max_stall_count=3, max_round_count=5)
        self._round = 0

    async def plan(self, magentic_context: MagenticContext) -> ChatMessage:
        return ChatMessage(role=Role.ASSISTANT, text="plan", author_name="magentic_manager")

    async def replan(self, magentic_context: MagenticContext) -> ChatMessage:
        return await self.plan(magentic_context)

    async def create_progress_ledger(self, magentic_context: MagenticContext) -> _MagenticProgressLedger:
        participants = list(magentic_context.participant_descriptions.keys())
        target = participants[0] if participants else "agent"
        if self._round == 0:
            self._round += 1
            return _MagenticProgressLedger(
                is_request_satisfied=_MagenticProgressLedgerItem(reason="", answer=False),
                is_in_loop=_MagenticProgressLedgerItem(reason="", answer=False),
                is_progress_being_made=_MagenticProgressLedgerItem(reason="", answer=True),
                next_speaker=_MagenticProgressLedgerItem(reason="", answer=target),
                instruction_or_question=_MagenticProgressLedgerItem(reason="", answer="respond"),
            )
        return _MagenticProgressLedger(
            is_request_satisfied=_MagenticProgressLedgerItem(reason="", answer=True),
            is_in_loop=_MagenticProgressLedgerItem(reason="", answer=False),
            is_progress_being_made=_MagenticProgressLedgerItem(reason="", answer=True),
            next_speaker=_MagenticProgressLedgerItem(reason="", answer=target),
            instruction_or_question=_MagenticProgressLedgerItem(reason="", answer=""),
        )

    async def prepare_final_answer(self, magentic_context: MagenticContext) -> ChatMessage:
        return ChatMessage(role=Role.ASSISTANT, text="final", author_name="magentic_manager")


async def test_group_chat_builder_basic_flow() -> None:
    selector = make_sequence_selector()
    alpha = StubAgent("alpha", "ack from alpha")
    beta = StubAgent("beta", "ack from beta")

    workflow = (
        GroupChatBuilder()
        .select_speakers(selector, display_name="manager", final_message="done")
        .participants(alpha=alpha, beta=beta)
        .build()
    )

    outputs: list[ChatMessage] = []
    async for event in workflow.run_stream("coordinate task"):
        if isinstance(event, WorkflowOutputEvent):
            data = event.data
            if isinstance(data, ChatMessage):
                outputs.append(data)

    assert len(outputs) == 1
    assert outputs[0].text == "done"
    assert outputs[0].author_name == "manager"


async def test_magentic_builder_returns_workflow_and_runs() -> None:
    manager = StubMagenticManager()
    agent = StubAgent("writer", "first draft")

    workflow = MagenticBuilder().participants(writer=agent).with_standard_manager(manager=manager).build()

    assert isinstance(workflow, Workflow)

    outputs: list[ChatMessage] = []
    orchestrator_events: list[MagenticOrchestratorMessageEvent] = []
    agent_events: list[MagenticAgentMessageEvent] = []
    start_message = _MagenticStartMessage.from_string("compose summary")
    async for event in workflow.run_stream(start_message):
        if isinstance(event, MagenticOrchestratorMessageEvent):
            orchestrator_events.append(event)
        if isinstance(event, MagenticAgentMessageEvent):
            agent_events.append(event)
        if isinstance(event, WorkflowOutputEvent):
            msg = event.data
            if isinstance(msg, ChatMessage):
                outputs.append(msg)

    assert outputs, "Expected a final output message"
    final = outputs[-1]
    assert final.text == "final"
    assert final.author_name == "magentic_manager"
    assert orchestrator_events, "Expected orchestrator events to be emitted"
    assert agent_events, "Expected agent message events to be emitted"


async def test_group_chat_as_agent_accepts_conversation() -> None:
    selector = make_sequence_selector()
    alpha = StubAgent("alpha", "ack from alpha")
    beta = StubAgent("beta", "ack from beta")

    workflow = (
        GroupChatBuilder()
        .select_speakers(selector, display_name="manager", final_message="done")
        .participants(alpha=alpha, beta=beta)
        .build()
    )

    agent = workflow.as_agent(name="group-chat-agent")
    conversation = [
        ChatMessage(role=Role.USER, text="kickoff", author_name="user"),
        ChatMessage(role=Role.ASSISTANT, text="noted", author_name="alpha"),
    ]
    response = await agent.run(conversation)

    assert response.messages, "Expected agent conversation output"


async def test_magentic_as_agent_accepts_conversation() -> None:
    manager = StubMagenticManager()
    writer = StubAgent("writer", "draft")

    workflow = MagenticBuilder().participants(writer=writer).with_standard_manager(manager=manager).build()

    agent = workflow.as_agent(name="magentic-agent")
    conversation = [
        ChatMessage(role=Role.SYSTEM, text="Guidelines", author_name="system"),
        ChatMessage(role=Role.USER, text="Summarize the findings", author_name="requester"),
    ]
    response = await agent.run(conversation)

    assert isinstance(response, AgentRunResponse)


# グループチャット機能の包括的なテスト。


class TestGroupChatBuilder:
    """GroupChatBuilderの検証と設定のテスト。"""

    def test_build_without_manager_raises_error(self) -> None:
        """マネージャなしでビルドするとValueErrorが発生することをテストします。"""
        agent = StubAgent("test", "response")

        builder = GroupChatBuilder().participants([agent])

        with pytest.raises(ValueError, match="manager must be configured before build"):
            builder.build()

    def test_build_without_participants_raises_error(self) -> None:
        """参加者なしでビルドするとValueErrorが発生することをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        builder = GroupChatBuilder().select_speakers(selector)

        with pytest.raises(ValueError, match="participants must be configured before build"):
            builder.build()

    def test_duplicate_manager_configuration_raises_error(self) -> None:
        """複数のマネージャを設定するとValueErrorが発生することをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        builder = GroupChatBuilder().select_speakers(selector)

        with pytest.raises(ValueError, match="already has a manager configured"):
            builder.select_speakers(selector)

    def test_empty_participants_raises_error(self) -> None:
        """空の参加者リストがValueErrorを発生させることをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        builder = GroupChatBuilder().select_speakers(selector)

        with pytest.raises(ValueError, match="participants cannot be empty"):
            builder.participants([])

    def test_duplicate_participant_names_raises_error(self) -> None:
        """重複する参加者名がValueErrorを発生させることをテストします。"""
        agent1 = StubAgent("test", "response1")
        agent2 = StubAgent("test", "response2")

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        builder = GroupChatBuilder().select_speakers(selector)

        with pytest.raises(ValueError, match="Duplicate participant name 'test'"):
            builder.participants([agent1, agent2])

    def test_agent_without_name_raises_error(self) -> None:
        """name属性のないagentがValueErrorを発生させることをテストします。"""

        class AgentWithoutName(BaseAgent):
            def __init__(self) -> None:
                super().__init__(name="", description="test")

            async def run(self, messages: Any = None, *, thread: Any = None, **kwargs: Any) -> AgentRunResponse:
                return AgentRunResponse(messages=[])

            def run_stream(
                self, messages: Any = None, *, thread: Any = None, **kwargs: Any
            ) -> AsyncIterable[AgentRunResponseUpdate]:
                async def _stream() -> AsyncIterable[AgentRunResponseUpdate]:
                    yield AgentRunResponseUpdate(contents=[])

                return _stream()

        agent = AgentWithoutName()

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        builder = GroupChatBuilder().select_speakers(selector)

        with pytest.raises(ValueError, match="must define a non-empty 'name' attribute"):
            builder.participants([agent])

    def test_empty_participant_name_raises_error(self) -> None:
        """空の参加者名がValueErrorを発生させることをテストします。"""
        agent = StubAgent("test", "response")

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        builder = GroupChatBuilder().select_speakers(selector)

        with pytest.raises(ValueError, match="participant names must be non-empty strings"):
            builder.participants({"": agent})


class TestGroupChatOrchestrator:
    """GroupChatOrchestratorExecutorのコア機能のテスト。"""

    async def test_max_rounds_enforcement(self) -> None:
        """max_roundsが会話のラウンド数を適切に制限することをテストします。"""
        call_count = {"value": 0}

        def selector(state: GroupChatStateSnapshot) -> str | None:
            call_count["value"] += 1
            # 無限に続けようとするために常にagent名を返します。
            return "agent"

        agent = StubAgent("agent", "response")

        workflow = (
            GroupChatBuilder()
            .select_speakers(selector)
            .participants([agent])
            .with_max_rounds(2)  # Limit to 2 rounds
            .build()
        )

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream("test task"):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        # max_roundsにより終了しているはずで、少なくとも1つの出力を期待します。
        assert len(outputs) >= 1
        # 最終メッセージはラウンド制限に関するものであるべきです。
        final_output = outputs[-1]
        assert "round limit" in final_output.text.lower()

    async def test_unknown_participant_error(self) -> None:
        """_apply_directiveが不明な参加者に対してエラーを発生させることをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return "unknown_agent"  # 存在しない参加者を返します。

        agent = StubAgent("agent", "response")

        workflow = GroupChatBuilder().select_speakers(selector).participants([agent]).build()

        with pytest.raises(ValueError, match="Manager selected unknown participant 'unknown_agent'"):
            async for _ in workflow.run_stream("test task"):
                pass

    async def test_directive_without_agent_name_raises_error(self) -> None:
        """agent_nameなしのdirectiveがfinish=Falseの場合にエラーを発生させることをテストします。"""

        def bad_selector(state: GroupChatStateSnapshot) -> GroupChatDirective:
            # エラーを引き起こすために文字列ではなくGroupChatDirectiveオブジェクトを返します。
            return GroupChatDirective(finish=False, agent_name=None)  # type: ignore

        agent = StubAgent("agent", "response")

        # _SpeakerSelectorAdapterがこれをキャッチしてTypeErrorを発生させます。
        workflow = GroupChatBuilder().select_speakers(bad_selector).participants([agent]).build()  # type: ignore

        # selectorがstrまたはNoneを返さないためTypeErrorが発生するはずです。
        with pytest.raises(TypeError, match="must return a participant name \\(str\\) or None"):
            async for _ in workflow.run_stream("test"):
                pass

    async def test_handle_empty_conversation_raises_error(self) -> None:
        """空の会話リストがValueErrorを発生させることをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        agent = StubAgent("agent", "response")

        workflow = GroupChatBuilder().select_speakers(selector).participants([agent]).build()

        with pytest.raises(ValueError, match="requires at least one chat message"):
            async for _ in workflow.run_stream([]):
                pass

    async def test_unknown_participant_response_raises_error(self) -> None:
        """不明な参加者からの応答がエラーを発生させることをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return "agent"

        # _orchestrator_participant_messageを直接テストするためにオーケストレーターを作成します。
        orchestrator = GroupChatOrchestratorExecutor(
            manager=selector,  # type: ignore
            participants={"agent": "test agent"},
            manager_name="test_manager",  # type: ignore
        )

        # ワークフローコンテキストをモックします。
        class MockContext:
            async def yield_output(self, message: ChatMessage) -> None:
                pass

        ctx = MockContext()

        # オーケストレーターの状態を初期化します。
        orchestrator._task_message = ChatMessage(role=Role.USER, text="test")  # type: ignore
        orchestrator._conversation = [orchestrator._task_message]  # type: ignore
        orchestrator._history = []  # type: ignore
        orchestrator._pending_agent = None  # type: ignore
        orchestrator._round_index = 0  # type: ignore

        # 不明な参加者でテストします。
        message = ChatMessage(role=Role.ASSISTANT, text="response")

        with pytest.raises(ValueError, match="Received response from unknown participant 'unknown'"):
            await orchestrator._ingest_participant_message("unknown", message, ctx)  # type: ignore

    async def test_state_build_before_initialization_raises_error(self) -> None:
        """タスクメッセージの初期化前に_build_stateがエラーを発生させることをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            return None

        orchestrator = GroupChatOrchestratorExecutor(
            manager=selector,  # type: ignore
            participants={"agent": "test agent"},
            manager_name="test_manager",  # type: ignore
        )

        with pytest.raises(RuntimeError, match="state not initialized with task message"):
            orchestrator._build_state()  # type: ignore


class TestSpeakerSelectorAdapter:
    """_SpeakerSelectorAdapterの機能のテスト。"""

    async def test_selector_returning_list_with_multiple_items_raises_error(self) -> None:
        """複数のアイテムを返すselectorがエラーを発生させることをテストします。"""

        def bad_selector(state: GroupChatStateSnapshot) -> list[str]:
            return ["agent1", "agent2"]  # 複数のアイテム。

        adapter = _SpeakerSelectorAdapter(bad_selector, manager_name="manager")

        state = {
            "participants": {"agent1": "desc1", "agent2": "desc2"},
            "task": ChatMessage(role=Role.USER, text="test"),
            "conversation": (),
            "history": (),
            "round_index": 0,
            "pending_agent": None,
        }

        with pytest.raises(ValueError, match="must return a single participant name"):
            await adapter(state)

    async def test_selector_returning_non_string_raises_error(self) -> None:
        """文字列以外を返すselectorがTypeErrorを発生させることをテストします。"""

        def bad_selector(state: GroupChatStateSnapshot) -> int:
            return 42  # 文字列ではありません。

        adapter = _SpeakerSelectorAdapter(bad_selector, manager_name="manager")

        state = {
            "participants": {"agent": "desc"},
            "task": ChatMessage(role=Role.USER, text="test"),
            "conversation": (),
            "history": (),
            "round_index": 0,
            "pending_agent": None,
        }

        with pytest.raises(TypeError, match="must return a participant name \\(str\\) or None"):
            await adapter(state)

    async def test_selector_returning_empty_list_finishes(self) -> None:
        """空のリストを返すselectorが会話を終了させることをテストします。"""

        def empty_selector(state: GroupChatStateSnapshot) -> list[str]:
            return []  # 空リストは会話を終了させるべきです。

        adapter = _SpeakerSelectorAdapter(empty_selector, manager_name="manager")

        state = {
            "participants": {"agent": "desc"},
            "task": ChatMessage(role=Role.USER, text="test"),
            "conversation": (),
            "history": (),
            "round_index": 0,
            "pending_agent": None,
        }

        directive = await adapter(state)
        assert directive.finish is True
        assert directive.final_message is not None


class TestCheckpointing:
    """チェックポイント機能のテスト。"""

    async def test_workflow_with_checkpointing(self) -> None:
        """チェックポイントが有効な状態でワークフローが動作することをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            if state["round_index"] >= 1:
                return None
            return "agent"

        agent = StubAgent("agent", "response")
        storage = InMemoryCheckpointStorage()

        workflow = (
            GroupChatBuilder().select_speakers(selector).participants([agent]).with_checkpointing(storage).build()
        )

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream("test task"):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        assert len(outputs) == 1  # 正常に完了するはずです。


class TestPromptBasedManager:
    """_PromptBasedGroupChatManagerのテスト。"""

    async def test_manager_with_missing_next_agent_raises_error(self) -> None:
        """next_agentなしのマネージャディレクティブがRuntimeErrorを発生させることをテストします。"""

        class MockChatClient:
            async def get_response(self, messages: Any, response_format: Any = None) -> Any:
                # finish=Falseだがnext_agentがない応答を返します。
                class MockResponse:
                    def __init__(self) -> None:
                        self.value = {"finish": False, "next_agent": None}
                        self.messages: list[Any] = []

                return MockResponse()

        manager = _PromptBasedGroupChatManager(MockChatClient())  # type: ignore

        state = {
            "participants": {"agent": "desc"},
            "task": ChatMessage(role=Role.USER, text="test"),
            "conversation": (),
        }

        with pytest.raises(RuntimeError, match="missing next_agent while finish is False"):
            await manager(state)

    async def test_manager_with_unknown_participant_raises_error(self) -> None:
        """不明な参加者を選択するマネージャがRuntimeErrorを発生させることをテストします。"""

        class MockChatClient:
            async def get_response(self, messages: Any, response_format: Any = None) -> Any:
                # 不明な参加者を選択する応答を返します。
                class MockResponse:
                    def __init__(self) -> None:
                        self.value = {"finish": False, "next_agent": "unknown"}
                        self.messages: list[Any] = []

                return MockResponse()

        manager = _PromptBasedGroupChatManager(MockChatClient())  # type: ignore

        state = {
            "participants": {"agent": "desc"},
            "task": ChatMessage(role=Role.USER, text="test"),
            "conversation": (),
        }

        with pytest.raises(RuntimeError, match="Manager selected unknown participant 'unknown'"):
            await manager(state)


class TestFactoryFunctions:
    """ファクトリ関数のテスト。"""

    def test_default_orchestrator_factory_without_manager_raises_error(self) -> None:
        """デフォルトファクトリが manager の設定を必要とすることをテストします。"""
        config = _GroupChatConfig(manager=None, manager_name="test", participants={})

        with pytest.raises(RuntimeError, match="requires a manager to be set"):
            _default_orchestrator_factory(config)


class TestConversationHandling:
    """異なる会話入力タイプのテスト。"""

    async def test_handle_string_input(self) -> None:
        """文字列入力の処理が適切な ChatMessage を作成することをテストします。"""

        def selector(state: GroupChatStateSnapshot) -> str | None:
            # タスクが正しく変換されたことを検証します。
            assert state["task"].role == Role.USER
            assert state["task"].text == "test string"
            return None

        agent = StubAgent("agent", "response")

        workflow = GroupChatBuilder().select_speakers(selector).participants([agent]).build()

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream("test string"):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        assert len(outputs) == 1

    async def test_handle_chat_message_input(self) -> None:
        """ChatMessage 入力を直接処理することをテストします。"""
        task_message = ChatMessage(role=Role.USER, text="test message")

        def selector(state: GroupChatStateSnapshot) -> str | None:
            # タスクメッセージが保持されていることを検証します。
            assert state["task"] == task_message
            return None

        agent = StubAgent("agent", "response")

        workflow = GroupChatBuilder().select_speakers(selector).participants([agent]).build()

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream(task_message):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        assert len(outputs) == 1

    async def test_handle_conversation_list_input(self) -> None:
        """会話リストの処理がコンテキストを保持することをテストします。"""
        conversation = [
            ChatMessage(role=Role.SYSTEM, text="system message"),
            ChatMessage(role=Role.USER, text="user message"),
        ]

        def selector(state: GroupChatStateSnapshot) -> str | None:
            # 会話コンテキストが保持されていることを検証します。
            assert len(state["conversation"]) == 2
            assert state["task"].text == "user message"
            return None

        agent = StubAgent("agent", "response")

        workflow = GroupChatBuilder().select_speakers(selector).participants([agent]).build()

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream(conversation):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        assert len(outputs) == 1


class TestRoundLimitEnforcement:
    """ラウンド制限チェック機能のテスト。"""

    async def test_round_limit_in_apply_directive(self) -> None:
        """_apply_directive におけるラウンド制限の適用をテストします。"""
        rounds_called = {"count": 0}

        def selector(state: GroupChatStateSnapshot) -> str | None:
            rounds_called["count"] += 1
            # 制限適用をテストするために Agent の選択を繰り返します。
            return "agent"

        agent = StubAgent("agent", "response")

        workflow = (
            GroupChatBuilder()
            .select_speakers(selector)
            .participants([agent])
            .with_max_rounds(1)  # Very low limit
            .build()
        )

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream("test"):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        # 少なくとも1つの出力（ラウンド制限メッセージ）があるはずです。
        assert len(outputs) >= 1
        # 最後のメッセージはラウンド制限に関するものであるべきです。
        final_output = outputs[-1]
        assert "round limit" in final_output.text.lower()

    async def test_round_limit_in_ingest_participant_message(self) -> None:
        """参加者の応答後のラウンド制限適用をテストします。"""
        responses_received = {"count": 0}

        def selector(state: GroupChatStateSnapshot) -> str | None:
            responses_received["count"] += 1
            if responses_received["count"] == 1:
                return "agent"  # 最初の呼び出しで Agent を選択します。
            return "agent"  # 続行を試みますが、制限に達するはずです。

        agent = StubAgent("agent", "response from agent")

        workflow = (
            GroupChatBuilder()
            .select_speakers(selector)
            .participants([agent])
            .with_max_rounds(1)  # Hit limit after first response
            .build()
        )

        outputs: list[ChatMessage] = []
        async for event in workflow.run_stream("test"):
            if isinstance(event, WorkflowOutputEvent):
                data = event.data
                if isinstance(data, ChatMessage):
                    outputs.append(data)

        # 少なくとも1つの出力（ラウンド制限メッセージ）があるはずです。
        assert len(outputs) >= 1
        # 最後のメッセージはラウンド制限に関するものであるべきです。
        final_output = outputs[-1]
        assert "round limit" in final_output.text.lower()
