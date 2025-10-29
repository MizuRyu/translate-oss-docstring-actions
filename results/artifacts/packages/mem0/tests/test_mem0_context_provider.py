# Copyright (c) Microsoft. All rights reserved. pyright: reportPrivateUsage=false

from unittest.mock import AsyncMock, patch

import pytest
from agent_framework import ChatMessage, Context, Role
from agent_framework.exceptions import ServiceInitializationError
from agent_framework.mem0 import Mem0Provider


def test_mem0_provider_import() -> None:
    """Mem0Providerがインポート可能であることをテストします。"""
    assert Mem0Provider is not None


@pytest.fixture
def mock_mem0_client() -> AsyncMock:
    """モックのMem0 AsyncMemoryClientを作成する。"""
    from mem0 import AsyncMemoryClient

    mock_client = AsyncMock(spec=AsyncMemoryClient)
    mock_client.add = AsyncMock()
    mock_client.search = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    mock_client.async_client = AsyncMock()
    mock_client.async_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """テスト用のサンプルチャットメッセージを作成する。"""
    return [
        ChatMessage(role=Role.USER, text="Hello, how are you?"),
        ChatMessage(role=Role.ASSISTANT, text="I'm doing well, thank you!"),
        ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant"),
    ]


class TestMem0ProviderInitialization:
    """Mem0Providerの初期化と設定をテストします。"""

    def test_init_with_all_ids(self, mock_mem0_client: AsyncMock) -> None:
        """すべてのIDが提供された場合の初期化をテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            agent_id="agent123",
            application_id="app123",
            thread_id="thread123",
            mem0_client=mock_mem0_client,
        )
        assert provider.user_id == "user123"
        assert provider.agent_id == "agent123"
        assert provider.application_id == "app123"
        assert provider.thread_id == "thread123"

    def test_init_without_filters_succeeds(self, mock_mem0_client: AsyncMock) -> None:
        """フィルターなしでも初期化が成功することをテストします（検証は呼び出し時に行われます）。"""
        provider = Mem0Provider(mem0_client=mock_mem0_client)
        assert provider.user_id is None
        assert provider.agent_id is None
        assert provider.application_id is None
        assert provider.thread_id is None

    def test_init_with_custom_context_prompt(self, mock_mem0_client: AsyncMock) -> None:
        """カスタムcontext promptを使った初期化をテストします。"""
        custom_prompt = "## Custom Memories\nConsider these memories:"
        provider = Mem0Provider(user_id="user123", context_prompt=custom_prompt, mem0_client=mock_mem0_client)
        assert provider.context_prompt == custom_prompt

    def test_init_with_scope_to_per_operation_thread_id(self, mock_mem0_client: AsyncMock) -> None:
        """scope_to_per_operation_thread_idが有効な場合の初期化をテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            scope_to_per_operation_thread_id=True,
            mem0_client=mock_mem0_client,
        )
        assert provider.scope_to_per_operation_thread_id is True

    @patch("agent_framework_mem0._provider.AsyncMemoryClient")
    def test_init_creates_default_client_when_none_provided(self, mock_memory_client_class: AsyncMock) -> None:
        """クライアントが提供されない場合にデフォルトクライアントが作成されることをテストします。"""
        from mem0 import AsyncMemoryClient

        mock_client = AsyncMock(spec=AsyncMemoryClient)
        mock_memory_client_class.return_value = mock_client

        provider = Mem0Provider(user_id="user123", api_key="test_api_key")

        mock_memory_client_class.assert_called_once_with(api_key="test_api_key")
        assert provider.mem0_client == mock_client
        assert provider._should_close_client is True

    def test_init_with_provided_client_should_not_close(self, mock_mem0_client: AsyncMock) -> None:
        """提供されたクライアントはproviderによって閉じられないことをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        assert provider._should_close_client is False


class TestMem0ProviderAsyncContextManager:
    """非同期コンテキストマネージャの動作をテストします。"""

    async def test_async_context_manager_entry(self, mock_mem0_client: AsyncMock) -> None:
        """非同期コンテキストマネージャのエントリがselfを返すことをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        async with provider as ctx:
            assert ctx is provider

    async def test_async_context_manager_exit_closes_client_when_should_close(self) -> None:
        """非同期コンテキストマネージャがクライアントを閉じるべき場合に閉じることをテストします。"""
        from mem0 import AsyncMemoryClient

        mock_client = AsyncMock(spec=AsyncMemoryClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.async_client = AsyncMock()
        mock_client.async_client.aclose = AsyncMock()

        with patch("agent_framework_mem0._provider.AsyncMemoryClient", return_value=mock_client):
            provider = Mem0Provider(user_id="user123", api_key="test_key")
            assert provider._should_close_client is True

            async with provider:
                pass

            mock_client.__aexit__.assert_called_once()

    async def test_async_context_manager_exit_does_not_close_provided_client(self, mock_mem0_client: AsyncMock) -> None:
        """非同期コンテキストマネージャが提供されたクライアントを閉じないことをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        assert provider._should_close_client is False

        async with provider:
            pass

        mock_mem0_client.__aexit__.assert_not_called()


class TestMem0ProviderThreadMethods:
    """スレッドのライフサイクルメソッドをテストします。"""

    async def test_thread_created_sets_per_operation_thread_id(self, mock_mem0_client: AsyncMock) -> None:
        """thread_createdが操作ごとのスレッドIDを設定することをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)

        await provider.thread_created("thread123")

        assert provider._per_operation_thread_id == "thread123"

    async def test_thread_created_with_existing_thread_id(self, mock_mem0_client: AsyncMock) -> None:
        """thread_createdが既にスレッドIDが存在する場合の動作をテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        provider._per_operation_thread_id = "existing_thread"

        await provider.thread_created("thread123")

        # 既存のスレッドIDを上書きしないはずです
        assert provider._per_operation_thread_id == "existing_thread"

    async def test_thread_created_validation_with_scope_enabled(self, mock_mem0_client: AsyncMock) -> None:
        """scope_to_per_operation_thread_idが有効な場合のthread_createdの検証をテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            scope_to_per_operation_thread_id=True,
            mem0_client=mock_mem0_client,
        )
        provider._per_operation_thread_id = "existing_thread"

        with pytest.raises(ValueError) as exc_info:
            await provider.thread_created("different_thread")

        assert "can only be used with one thread at a time" in str(exc_info.value)

    async def test_messages_adding_sets_per_operation_thread_id(self, mock_mem0_client: AsyncMock) -> None:
        """invokedが操作ごとのスレッドIDを設定することをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)

        await provider.thread_created("thread123")

        assert provider._per_operation_thread_id == "thread123"


class TestMem0ProviderMessagesAdding:
    """invokedメソッドをテストします。"""

    async def test_messages_adding_fails_without_filters(self, mock_mem0_client: AsyncMock) -> None:
        """フィルターが提供されていない場合にinvokedが失敗することをテストします。"""
        provider = Mem0Provider(mem0_client=mock_mem0_client)
        message = ChatMessage(role=Role.USER, text="Hello!")

        with pytest.raises(ServiceInitializationError) as exc_info:
            await provider.invoked(message)

        assert "At least one of the filters" in str(exc_info.value)

    async def test_messages_adding_single_message(self, mock_mem0_client: AsyncMock) -> None:
        """単一メッセージの追加をテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        message = ChatMessage(role=Role.USER, text="Hello!")

        await provider.invoked(message)

        mock_mem0_client.add.assert_called_once()
        call_args = mock_mem0_client.add.call_args
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello!"}]
        assert call_args.kwargs["user_id"] == "user123"

    async def test_messages_adding_multiple_messages(
        self, mock_mem0_client: AsyncMock, sample_messages: list[ChatMessage]
    ) -> None:
        """複数メッセージの追加をテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)

        await provider.invoked(sample_messages)

        mock_mem0_client.add.assert_called_once()
        call_args = mock_mem0_client.add.call_args
        expected_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "system", "content": "You are a helpful assistant"},
        ]
        assert call_args.kwargs["messages"] == expected_messages

    async def test_messages_adding_with_agent_id(
        self, mock_mem0_client: AsyncMock, sample_messages: list[ChatMessage]
    ) -> None:
        """agent_idを含むメッセージの追加をテストします。"""
        provider = Mem0Provider(agent_id="agent123", mem0_client=mock_mem0_client)

        await provider.invoked(sample_messages)

        call_args = mock_mem0_client.add.call_args
        assert call_args.kwargs["agent_id"] == "agent123"
        assert call_args.kwargs["user_id"] is None

    async def test_messages_adding_with_application_id(
        self, mock_mem0_client: AsyncMock, sample_messages: list[ChatMessage]
    ) -> None:
        """metadataにapplication_idを含むメッセージの追加をテストします。"""
        provider = Mem0Provider(user_id="user123", application_id="app123", mem0_client=mock_mem0_client)

        await provider.invoked(sample_messages)

        call_args = mock_mem0_client.add.call_args
        assert call_args.kwargs["metadata"] == {"application_id": "app123"}

    async def test_messages_adding_with_scope_to_per_operation_thread_id(
        self, mock_mem0_client: AsyncMock, sample_messages: list[ChatMessage]
    ) -> None:
        """scope_to_per_operation_thread_idが有効な場合のメッセージ追加をテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            thread_id="base_thread",
            scope_to_per_operation_thread_id=True,
            mem0_client=mock_mem0_client,
        )
        provider._per_operation_thread_id = "operation_thread"

        await provider.thread_created(thread_id="operation_thread")
        await provider.invoked(sample_messages)

        call_args = mock_mem0_client.add.call_args
        assert call_args.kwargs["run_id"] == "operation_thread"

    async def test_messages_adding_without_scope_uses_base_thread_id(
        self, mock_mem0_client: AsyncMock, sample_messages: list[ChatMessage]
    ) -> None:
        """スコープなしでのメッセージ追加はベースのthread_idを使用することをテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            thread_id="base_thread",
            scope_to_per_operation_thread_id=False,
            mem0_client=mock_mem0_client,
        )

        await provider.invoked(sample_messages)

        call_args = mock_mem0_client.add.call_args
        assert call_args.kwargs["run_id"] == "base_thread"

    async def test_messages_adding_filters_empty_messages(self, mock_mem0_client: AsyncMock) -> None:
        """空または無効なメッセージがフィルタリングされることをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        messages = [
            ChatMessage(role=Role.USER, text=""),  # Empty text
            ChatMessage(role=Role.USER, text="   "),  # Whitespace only
            ChatMessage(role=Role.USER, text="Valid message"),
        ]

        await provider.invoked(messages)

        call_args = mock_mem0_client.add.call_args
        # 有効なメッセージのみを含むはずです
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Valid message"}]

    async def test_messages_adding_skips_when_no_valid_messages(self, mock_mem0_client: AsyncMock) -> None:
        """有効なメッセージが存在しない場合にmem0クライアントが呼び出されないことをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        messages = [
            ChatMessage(role=Role.USER, text=""),
            ChatMessage(role=Role.USER, text="   "),
        ]

        await provider.invoked(messages)

        mock_mem0_client.add.assert_not_called()


class TestMem0ProviderModelInvoking:
    """invokingメソッドをテストします。"""

    async def test_model_invoking_fails_without_filters(self, mock_mem0_client: AsyncMock) -> None:
        """フィルターが提供されていない場合にinvokingが失敗することをテストします。"""
        provider = Mem0Provider(mem0_client=mock_mem0_client)
        message = ChatMessage(role=Role.USER, text="What's the weather?")

        with pytest.raises(ServiceInitializationError) as exc_info:
            await provider.invoking(message)

        assert "At least one of the filters" in str(exc_info.value)

    async def test_model_invoking_single_message(self, mock_mem0_client: AsyncMock) -> None:
        """単一メッセージでのinvokingをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        message = ChatMessage(role=Role.USER, text="What's the weather?")

        # モックの検索結果
        mock_mem0_client.search.return_value = [
            {"memory": "User likes outdoor activities"},
            {"memory": "User lives in Seattle"},
        ]

        context = await provider.invoking(message)

        mock_mem0_client.search.assert_called_once()
        call_args = mock_mem0_client.search.call_args
        assert call_args.kwargs["query"] == "What's the weather?"
        assert call_args.kwargs["user_id"] == "user123"

        assert isinstance(context, Context)
        expected_instructions = (
            "## Memories\nConsider the following memories when answering user questions:\n"
            "User likes outdoor activities\nUser lives in Seattle"
        )

        assert context.messages
        assert context.messages[0].text == expected_instructions

    async def test_model_invoking_multiple_messages(
        self, mock_mem0_client: AsyncMock, sample_messages: list[ChatMessage]
    ) -> None:
        """複数メッセージでのinvokingをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)

        mock_mem0_client.search.return_value = [{"memory": "Previous conversation context"}]

        await provider.invoking(sample_messages)

        call_args = mock_mem0_client.search.call_args
        expected_query = "Hello, how are you?\nI'm doing well, thank you!\nYou are a helpful assistant"
        assert call_args.kwargs["query"] == expected_query

    async def test_model_invoking_with_agent_id(self, mock_mem0_client: AsyncMock) -> None:
        """agent_idを含むinvokingをテストします。"""
        provider = Mem0Provider(agent_id="agent123", mem0_client=mock_mem0_client)
        message = ChatMessage(role=Role.USER, text="Hello")

        mock_mem0_client.search.return_value = []

        await provider.invoking(message)

        call_args = mock_mem0_client.search.call_args
        assert call_args.kwargs["agent_id"] == "agent123"
        assert call_args.kwargs["user_id"] is None

    async def test_model_invoking_with_scope_to_per_operation_thread_id(self, mock_mem0_client: AsyncMock) -> None:
        """scope_to_per_operation_thread_idが有効な場合のinvokingをテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            thread_id="base_thread",
            scope_to_per_operation_thread_id=True,
            mem0_client=mock_mem0_client,
        )
        provider._per_operation_thread_id = "operation_thread"
        message = ChatMessage(role=Role.USER, text="Hello")

        mock_mem0_client.search.return_value = []

        await provider.invoking(message)

        call_args = mock_mem0_client.search.call_args
        assert call_args.kwargs["run_id"] == "operation_thread"

    async def test_model_invoking_no_memories_returns_none_instructions(self, mock_mem0_client: AsyncMock) -> None:
        """メモリがない場合はNoneの指示を含むコンテキストを返すことをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        message = ChatMessage(role=Role.USER, text="Hello")

        mock_mem0_client.search.return_value = []

        context = await provider.invoking(message)

        assert isinstance(context, Context)
        assert not context.messages

    async def test_model_invoking_filters_empty_message_text(self, mock_mem0_client: AsyncMock) -> None:
        """空のメッセージテキストがクエリからフィルタリングされることをテストします。"""
        provider = Mem0Provider(user_id="user123", mem0_client=mock_mem0_client)
        messages = [
            ChatMessage(role=Role.USER, text=""),
            ChatMessage(role=Role.USER, text="Valid message"),
            ChatMessage(role=Role.USER, text="   "),
        ]

        mock_mem0_client.search.return_value = []

        await provider.invoking(messages)

        call_args = mock_mem0_client.search.call_args
        assert call_args.kwargs["query"] == "Valid message"

    async def test_model_invoking_custom_context_prompt(self, mock_mem0_client: AsyncMock) -> None:
        """カスタムcontext promptを使ったinvokingをテストします。"""
        custom_prompt = "## Custom Context\nRemember these details:"
        provider = Mem0Provider(
            user_id="user123",
            context_prompt=custom_prompt,
            mem0_client=mock_mem0_client,
        )
        message = ChatMessage(role=Role.USER, text="Hello")

        mock_mem0_client.search.return_value = [{"memory": "Test memory"}]

        context = await provider.invoking(message)

        expected_instructions = "## Custom Context\nRemember these details:\nTest memory"
        assert context.messages
        assert context.messages[0].text == expected_instructions


class TestMem0ProviderValidation:
    """検証メソッドをテストします。"""

    def test_validate_per_operation_thread_id_success(self, mock_mem0_client: AsyncMock) -> None:
        """操作ごとのスレッドIDの検証が成功することをテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            scope_to_per_operation_thread_id=True,
            mem0_client=mock_mem0_client,
        )
        provider._per_operation_thread_id = "thread123"

        # 同じスレッドIDの場合は例外を発生させないはずです
        provider._validate_per_operation_thread_id("thread123")

        # Noneの場合は例外を発生させないはずです
        provider._validate_per_operation_thread_id(None)

    def test_validate_per_operation_thread_id_failure(self, mock_mem0_client: AsyncMock) -> None:
        """競合するスレッドIDの検証失敗をテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            scope_to_per_operation_thread_id=True,
            mem0_client=mock_mem0_client,
        )
        provider._per_operation_thread_id = "thread123"

        with pytest.raises(ValueError) as exc_info:
            provider._validate_per_operation_thread_id("different_thread")

        assert "can only be used with one thread at a time" in str(exc_info.value)

    def test_validate_per_operation_thread_id_disabled_scope(self, mock_mem0_client: AsyncMock) -> None:
        """スコープが無効な場合は検証がスキップされることをテストします。"""
        provider = Mem0Provider(
            user_id="user123",
            scope_to_per_operation_thread_id=False,
            mem0_client=mock_mem0_client,
        )
        provider._per_operation_thread_id = "thread123"

        # 異なるスレッドIDでも例外を発生させないはずです
        provider._validate_per_operation_thread_id("different_thread")
