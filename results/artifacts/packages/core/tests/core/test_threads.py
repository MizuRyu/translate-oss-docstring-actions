# Copyright (c) Microsoft. All rights reserved.

from collections.abc import Sequence
from typing import Any

import pytest

from agent_framework import AgentThread, ChatMessage, ChatMessageStore, Role
from agent_framework._threads import AgentThreadState, ChatMessageStoreState
from agent_framework.exceptions import AgentThreadException


class MockChatMessageStore:
    """テスト用の ChatMessageStoreProtocol のモック実装。"""

    def __init__(self, messages: list[ChatMessage] | None = None) -> None:
        self._messages = messages or []
        self._serialize_calls = 0
        self._deserialize_calls = 0

    async def list_messages(self) -> list[ChatMessage]:
        return self._messages

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        self._messages.extend(messages)

    async def serialize(self, **kwargs: Any) -> Any:
        self._serialize_calls += 1
        return {"messages": [msg.__dict__ for msg in self._messages], "kwargs": kwargs}

    async def update_from_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        self._deserialize_calls += 1
        if serialized_store_state and "messages" in serialized_store_state:
            self._messages = serialized_store_state["messages"]

    @classmethod
    async def deserialize(cls, serialized_store_state: Any, **kwargs: Any) -> "MockChatMessageStore":
        instance = cls()
        await instance.update_from_state(serialized_store_state, **kwargs)
        return instance


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """テスト用のサンプルチャットメッセージを提供するフィクスチャ。"""
    return [
        ChatMessage(role=Role.USER, text="Hello", message_id="msg1"),
        ChatMessage(role=Role.ASSISTANT, text="Hi there!", message_id="msg2"),
        ChatMessage(role=Role.USER, text="How are you?", message_id="msg3"),
    ]


@pytest.fixture
def sample_message() -> ChatMessage:
    """テスト用の単一のサンプルチャットメッセージを提供するフィクスチャ。"""
    return ChatMessage(role=Role.USER, text="Test message", message_id="test1")


class TestAgentThread:
    """AgentThread クラスのテストケース。"""

    def test_init_with_no_parameters(self) -> None:
        """パラメータなしでの AgentThread 初期化をテストします。"""
        thread = AgentThread()
        assert thread.service_thread_id is None
        assert thread.message_store is None

    def test_init_with_service_thread_id(self) -> None:
        """service_thread_id を指定した AgentThread 初期化をテストします。"""
        service_thread_id = "test-conversation-123"
        thread = AgentThread(service_thread_id=service_thread_id)
        assert thread.service_thread_id == service_thread_id
        assert thread.message_store is None

    def test_init_with_message_store(self) -> None:
        """message_store を指定した AgentThread 初期化をテストします。"""
        store = ChatMessageStore()
        thread = AgentThread(message_store=store)
        assert thread.service_thread_id is None
        assert thread.message_store is store

    def test_service_thread_id_property_setter(self) -> None:
        """service_thread_id プロパティのセッターをテストします。"""
        thread = AgentThread()
        service_thread_id = "test-conversation-456"

        thread.service_thread_id = service_thread_id
        assert thread.service_thread_id == service_thread_id

    def test_service_thread_id_setter_with_existing_message_store_raises_error(self) -> None:
        """message_store が存在する場合に service_thread_id を設定すると AgentThreadException が発生することをテストします。"""
        store = ChatMessageStore()
        thread = AgentThread(message_store=store)

        with pytest.raises(AgentThreadException, match="Only the service_thread_id or message_store may be set"):
            thread.service_thread_id = "test-conversation-789"

    def test_service_thread_id_setter_with_none_values(self) -> None:
        """None 値での service_thread_id セッターは何もしないことをテストします。"""
        thread = AgentThread()
        thread.service_thread_id = None  # エラーを発生させるべきではありません。
        assert thread.service_thread_id is None

    def test_message_store_property_setter(self) -> None:
        """message_store プロパティのセッターをテストします。"""
        thread = AgentThread()
        store = ChatMessageStore()

        thread.message_store = store
        assert thread.message_store is store

    def test_message_store_setter_with_existing_service_thread_id_raises_error(self) -> None:
        """service_thread_id が存在する場合に message_store を設定すると AgentThreadException が発生することをテストします。"""
        service_thread_id = "test-conversation-999"
        thread = AgentThread(service_thread_id=service_thread_id)
        store = ChatMessageStore()

        with pytest.raises(AgentThreadException, match="Only the service_thread_id or message_store may be set"):
            thread.message_store = store

    def test_message_store_setter_with_none_values(self) -> None:
        """None 値での message_store セッターは何もしないことをテストします。"""
        thread = AgentThread()
        thread.message_store = None  # エラーを発生させるべきではありません。
        assert thread.message_store is None

    async def test_get_messages_with_message_store(self, sample_messages: list[ChatMessage]) -> None:
        """message_store が設定されている場合の get_messages をテストします。"""
        store = ChatMessageStore(sample_messages)
        thread = AgentThread(message_store=store)

        assert thread.message_store is not None

        messages: list[ChatMessage] = await thread.message_store.list_messages()

        assert messages is not None
        assert len(messages) == 3
        assert messages[0].text == "Hello"
        assert messages[1].text == "Hi there!"
        assert messages[2].text == "How are you?"

    async def test_get_messages_with_no_message_store(self) -> None:
        """message_store が設定されていない場合の get_messages をテストします。"""
        thread = AgentThread()

        assert thread.message_store is None

    async def test_on_new_messages_with_service_thread_id(self, sample_message: ChatMessage) -> None:
        """service_thread_id が設定されている場合の _on_new_messages は何もしないことをテストします。"""
        thread = AgentThread(service_thread_id="test-conv")

        await thread.on_new_messages(sample_message)

        # メッセージストアを作成すべきではありません。
        assert thread.message_store is None

    async def test_on_new_messages_single_message_creates_store(self, sample_message: ChatMessage) -> None:
        """単一メッセージで _on_new_messages が ChatMessageStore を作成することをテストします。"""
        thread = AgentThread()

        await thread.on_new_messages(sample_message)

        assert thread.message_store is not None
        assert isinstance(thread.message_store, ChatMessageStore)
        messages = await thread.message_store.list_messages()
        assert len(messages) == 1
        assert messages[0].text == "Test message"

    async def test_on_new_messages_multiple_messages(self, sample_messages: list[ChatMessage]) -> None:
        """複数メッセージでの _on_new_messages をテストします。"""
        thread = AgentThread()

        await thread.on_new_messages(sample_messages)

        assert thread.message_store is not None
        messages = await thread.message_store.list_messages()
        assert len(messages) == 3

    async def test_on_new_messages_with_existing_store(self, sample_message: ChatMessage) -> None:
        """既存のメッセージストアに _on_new_messages が追加することをテストします。"""
        initial_messages = [ChatMessage(role=Role.USER, text="Initial", message_id="init1")]
        store = ChatMessageStore(initial_messages)
        thread = AgentThread(message_store=store)

        await thread.on_new_messages(sample_message)

        assert thread.message_store is not None
        messages = await thread.message_store.list_messages()
        assert len(messages) == 2
        assert messages[0].text == "Initial"
        assert messages[1].text == "Test message"

    async def test_deserialize_with_service_thread_id(self) -> None:
        """service_thread_id を用いた _deserialize をテストします。"""
        serialized_data = {"service_thread_id": "test-conv-123", "chat_message_store_state": None}

        thread = await AgentThread.deserialize(serialized_data)

        assert thread.service_thread_id == "test-conv-123"
        assert thread.message_store is None

    async def test_deserialize_with_store_state(self, sample_messages: list[ChatMessage]) -> None:
        """chat_message_store_state を用いた _deserialize をテストします。"""
        store_state = {"messages": sample_messages}
        serialized_data = {"service_thread_id": None, "chat_message_store_state": store_state}

        thread = await AgentThread.deserialize(serialized_data)

        assert thread.service_thread_id is None
        assert thread.message_store is not None
        assert isinstance(thread.message_store, ChatMessageStore)

    async def test_deserialize_with_no_state(self) -> None:
        """状態なしでの _deserialize をテストします。"""
        thread = AgentThread()
        serialized_data = {"service_thread_id": None, "chat_message_store_state": None}

        await thread.deserialize(serialized_data)

        assert thread.service_thread_id is None
        assert thread.message_store is None

    async def test_deserialize_with_existing_store(self) -> None:
        """既存のメッセージストアを用いた _deserialize をテストします。"""
        store = MockChatMessageStore()
        thread = AgentThread(message_store=store)
        serialized_data: dict[str, Any] = {
            "service_thread_id": None,
            "chat_message_store_state": {"messages": [ChatMessage(role="user", text="test")]},
        }

        await thread.update_from_thread_state(serialized_data)

        assert store._messages
        assert store._messages[0].text == "test"

    async def test_serialize_with_service_thread_id(self) -> None:
        """service_thread_id を用いたシリアライズをテストします。"""
        thread = AgentThread(service_thread_id="test-conv-456")

        result = await thread.serialize()

        assert result["service_thread_id"] == "test-conv-456"
        assert result["chat_message_store_state"] is None

    async def test_serialize_with_message_store(self) -> None:
        """message_store を用いたシリアライズをテストします。"""
        store = MockChatMessageStore()
        thread = AgentThread(message_store=store)

        result = await thread.serialize()

        assert result["service_thread_id"] is None
        assert result["chat_message_store_state"] is not None
        assert store._serialize_calls == 1  # pyright: ignore[reportPrivateUsage]

    async def test_serialize_with_no_state(self) -> None:
        """状態なしでのシリアライズをテストします。"""
        thread = AgentThread()

        result = await thread.serialize()

        assert result["service_thread_id"] is None
        assert result["chat_message_store_state"] is None

    async def test_serialize_with_kwargs(self) -> None:
        """シリアライズが kwargs を message store に渡すことをテストします。"""
        store = MockChatMessageStore()
        thread = AgentThread(message_store=store)

        await thread.serialize(custom_param="test_value")

        assert store._serialize_calls == 1  # pyright: ignore[reportPrivateUsage]

    async def test_serialize_round_trip_messages(self, sample_messages: list[ChatMessage]) -> None:
        """シリアライズの往復をテストします。"""
        store = ChatMessageStore(sample_messages)
        thread = AgentThread(message_store=store)
        new_thread = await AgentThread.deserialize(await thread.serialize())
        assert new_thread.message_store is not None
        new_messages = await new_thread.message_store.list_messages()
        assert len(new_messages) == len(sample_messages)
        assert {new.text for new in new_messages} == {orig.text for orig in sample_messages}

    async def test_serialize_round_trip_thread_id(self) -> None:
        """シリアライズの往復をテストします。"""
        thread = AgentThread(service_thread_id="test-1234")
        new_thread = await AgentThread.deserialize(await thread.serialize())
        assert new_thread.message_store is None
        assert new_thread.service_thread_id == "test-1234"


class TestChatMessageList:
    """ChatMessageStore クラスのテストケース。"""

    def test_init_empty(self) -> None:
        """メッセージなしでの ChatMessageStore 初期化をテストします。"""
        store = ChatMessageStore()
        assert len(store.messages) == 0

    def test_init_with_messages(self, sample_messages: list[ChatMessage]) -> None:
        """メッセージありでの ChatMessageStore 初期化をテストします。"""
        store = ChatMessageStore(sample_messages)
        assert len(store.messages) == 3

    async def test_add_messages(self, sample_messages: list[ChatMessage]) -> None:
        """ストアへのメッセージ追加をテストします。"""
        store = ChatMessageStore()

        await store.add_messages(sample_messages)

        assert len(store.messages) == 3
        messages = await store.list_messages()
        assert messages[0].text == "Hello"

    async def test_get_messages(self, sample_messages: list[ChatMessage]) -> None:
        """ストアからのメッセージ取得をテストします。"""
        store = ChatMessageStore(sample_messages)

        messages = await store.list_messages()

        assert len(messages) == 3
        assert messages[0].message_id == "msg1"

    async def test_serialize_state(self, sample_messages: list[ChatMessage]) -> None:
        """ストア状態のシリアライズをテストします。"""
        store = ChatMessageStore(sample_messages)

        result = await store.serialize()

        assert "messages" in result
        assert len(result["messages"]) == 3

    async def test_serialize_state_empty(self) -> None:
        """空のストア状態のシリアライズをテストします。"""
        store = ChatMessageStore()

        result = await store.serialize()

        assert "messages" in result
        assert len(result["messages"]) == 0

    async def test_deserialize_state(self, sample_messages: list[ChatMessage]) -> None:
        """ストア状態のデシリアライズをテストします。"""
        store = ChatMessageStore()
        state_data = {"messages": sample_messages}

        await store.update_from_state(state_data)

        messages = await store.list_messages()
        assert len(messages) == 3
        assert messages[0].text == "Hello"

    async def test_deserialize_state_none(self) -> None:
        """None 状態のデシリアライズをテストします。"""
        store = ChatMessageStore()

        await store.update_from_state(None)

        assert len(store.messages) == 0

    async def test_deserialize_state_empty(self) -> None:
        """空の状態のデシリアライズをテストします。"""
        store = ChatMessageStore()

        await store.update_from_state({})

        assert len(store.messages) == 0


class TestStoreState:
    """ChatMessageStoreState クラスのテストケース。"""

    def test_init(self, sample_messages: list[ChatMessage]) -> None:
        """ChatMessageStoreState の初期化をテストします。"""
        state = ChatMessageStoreState(messages=sample_messages)

        assert len(state.messages) == 3
        assert state.messages[0].text == "Hello"

    def test_init_empty(self) -> None:
        """空のメッセージでの ChatMessageStoreState 初期化をテストします。"""
        state = ChatMessageStoreState(messages=[])

        assert len(state.messages) == 0


class TestThreadState:
    """AgentThreadState クラスのテストケース。"""

    def test_init_with_service_thread_id(self) -> None:
        """service_thread_id を用いた AgentThreadState 初期化をテストします。"""
        state = AgentThreadState(service_thread_id="test-conv-123")

        assert state.service_thread_id == "test-conv-123"
        assert state.chat_message_store_state is None

    def test_init_with_chat_message_store_state(self) -> None:
        """chat_message_store_state を用いた AgentThreadState 初期化をテストします。"""
        store_data: dict[str, Any] = {"messages": []}
        state = AgentThreadState.from_dict({"chat_message_store_state": store_data})

        assert state.service_thread_id is None
        assert state.chat_message_store_state.messages == []

    def test_init_with_both(self) -> None:
        """両方のパラメータを用いた AgentThreadState 初期化をテストします。"""
        store_data: dict[str, Any] = {"messages": []}
        with pytest.raises(AgentThreadException):
            AgentThreadState(service_thread_id="test-conv-123", chat_message_store_state=store_data)

    def test_init_defaults(self) -> None:
        """デフォルト値での AgentThreadState 初期化をテストします。"""
        state = AgentThreadState()

        assert state.service_thread_id is None
        assert state.chat_message_store_state is None
