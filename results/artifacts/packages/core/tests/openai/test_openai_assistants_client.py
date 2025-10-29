# Copyright (c) Microsoft. All rights reserved.

import json
import os
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.beta.threads import MessageDeltaEvent, Run, TextDeltaBlock
from openai.types.beta.threads.runs import RunStep
from pydantic import Field

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    ChatAgent,
    ChatClientProtocol,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    FunctionCallContent,
    FunctionResultContent,
    HostedCodeInterpreterTool,
    HostedFileSearchTool,
    HostedVectorStoreContent,
    Role,
    TextContent,
    ToolMode,
    UriContent,
    UsageContent,
    ai_function,
)
from agent_framework.exceptions import ServiceInitializationError
from agent_framework.openai import OpenAIAssistantsClient

skip_if_openai_integration_tests_disabled = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
    or os.getenv("OPENAI_API_KEY", "") in ("", "test-dummy-key"),
    reason="No real OPENAI_API_KEY provided; skipping integration tests."
    if os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    else "Integration tests are disabled.",
)


def create_test_openai_assistants_client(
    mock_async_openai: MagicMock,
    model_id: str | None = None,
    assistant_id: str | None = None,
    assistant_name: str | None = None,
    thread_id: str | None = None,
    should_delete_assistant: bool = False,
) -> OpenAIAssistantsClient:
    """テスト用にOpenAIAssistantsClientインスタンスを作成するヘルパー関数。"""
    client = OpenAIAssistantsClient(
        model_id=model_id or "gpt-4",
        assistant_id=assistant_id,
        assistant_name=assistant_name,
        thread_id=thread_id,
        api_key="test-api-key",
        org_id="test-org-id",
        async_client=mock_async_openai,
    )
    # 必要に応じて_should_delete_assistantフラグを直接設定
    if should_delete_assistant:
        object.__setattr__(client, "_should_delete_assistant", True)
    return client


async def create_vector_store(client: OpenAIAssistantsClient) -> tuple[str, HostedVectorStoreContent]:
    """テスト用のサンプルドキュメントを持つベクターストアを作成。"""
    file = await client.client.files.create(
        file=("todays_weather.txt", b"The weather today is sunny with a high of 25C."), purpose="user_data"
    )
    vector_store = await client.client.vector_stores.create(
        name="knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    result = await client.client.vector_stores.files.create_and_poll(vector_store_id=vector_store.id, file_id=file.id)
    if result.last_error is not None:
        raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")

    return file.id, HostedVectorStoreContent(vector_store_id=vector_store.id)


async def delete_vector_store(client: OpenAIAssistantsClient, file_id: str, vector_store_id: str) -> None:
    """テスト後にベクターストアを削除。"""

    await client.client.vector_stores.delete(vector_store_id=vector_store_id)
    await client.client.files.delete(file_id=file_id)


@pytest.fixture
def mock_async_openai() -> MagicMock:
    """AsyncOpenAIクライアントのモック。"""
    mock_client = MagicMock()

    # beta.assistantsのモック
    mock_client.beta.assistants.create = AsyncMock(return_value=MagicMock(id="test-assistant-id"))
    mock_client.beta.assistants.delete = AsyncMock()

    # beta.threadsのモック
    mock_client.beta.threads.create = AsyncMock(return_value=MagicMock(id="test-thread-id"))
    mock_client.beta.threads.delete = AsyncMock()

    # beta.threads.runsのモック
    mock_client.beta.threads.runs.create = AsyncMock(return_value=MagicMock(id="test-run-id"))
    mock_client.beta.threads.runs.retrieve = AsyncMock()
    mock_client.beta.threads.runs.submit_tool_outputs = AsyncMock()
    mock_client.beta.threads.runs.cancel = AsyncMock()

    # Mock beta.threads.messages
    mock_client.beta.threads.messages.create = AsyncMock()
    mock_client.beta.threads.messages.list = AsyncMock(return_value=MagicMock(data=[]))

    return mock_client


def test_openai_assistants_client_init_with_client(mock_async_openai: MagicMock) -> None:
    """既存のclientを使ったOpenAIAssistantsClientの初期化テスト。"""
    chat_client = create_test_openai_assistants_client(
        mock_async_openai, model_id="gpt-4", assistant_id="existing-assistant-id", thread_id="test-thread-id"
    )

    assert chat_client.client is mock_async_openai
    assert chat_client.model_id == "gpt-4"
    assert chat_client.assistant_id == "existing-assistant-id"
    assert chat_client.thread_id == "test-thread-id"
    assert not chat_client._should_delete_assistant  # type: ignore
    assert isinstance(chat_client, ChatClientProtocol)


def test_openai_assistants_client_init_auto_create_client(
    openai_unit_test_env: dict[str, str],
    mock_async_openai: MagicMock,
) -> None:
    """自動生成されたclientを使ったOpenAIAssistantsClientの初期化テスト。"""
    chat_client = OpenAIAssistantsClient(
        model_id=openai_unit_test_env["OPENAI_CHAT_MODEL_ID"],
        assistant_name="TestAssistant",
        api_key=openai_unit_test_env["OPENAI_API_KEY"],
        org_id=openai_unit_test_env["OPENAI_ORG_ID"],
        async_client=mock_async_openai,
    )

    assert chat_client.client is mock_async_openai
    assert chat_client.model_id == openai_unit_test_env["OPENAI_CHAT_MODEL_ID"]
    assert chat_client.assistant_id is None
    assert chat_client.assistant_name == "TestAssistant"
    assert not chat_client._should_delete_assistant  # type: ignore


def test_openai_assistants_client_init_validation_fail() -> None:
    """バリデーション失敗時のOpenAIAssistantsClient初期化テスト。"""
    with pytest.raises(ServiceInitializationError):
        # 無効なmodel IDタイプを提供して強制的に失敗させる - これによりバリデーションが失敗するはず。
        OpenAIAssistantsClient(model_id=123, api_key="valid-key")  # type: ignore


@pytest.mark.parametrize("exclude_list", [["OPENAI_CHAT_MODEL_ID"]], indirect=True)
def test_openai_assistants_client_init_missing_model_id(openai_unit_test_env: dict[str, str]) -> None:
    """model IDが欠落している場合のOpenAIAssistantsClient初期化テスト。"""
    with pytest.raises(ServiceInitializationError):
        OpenAIAssistantsClient(
            api_key=openai_unit_test_env.get("OPENAI_API_KEY", "test-key"), env_file_path="nonexistent.env"
        )


@pytest.mark.parametrize("exclude_list", [["OPENAI_API_KEY"]], indirect=True)
def test_openai_assistants_client_init_missing_api_key(openai_unit_test_env: dict[str, str]) -> None:
    """APIキーが欠落している場合のOpenAIAssistantsClient初期化テスト。"""
    with pytest.raises(ServiceInitializationError):
        OpenAIAssistantsClient(model_id="gpt-4", env_file_path="nonexistent.env")


def test_openai_assistants_client_init_with_default_headers(openai_unit_test_env: dict[str, str]) -> None:
    """デフォルトヘッダーを使ったOpenAIAssistantsClient初期化テスト。"""
    default_headers = {"X-Unit-Test": "test-guid"}

    chat_client = OpenAIAssistantsClient(
        model_id="gpt-4",
        api_key=openai_unit_test_env["OPENAI_API_KEY"],
        default_headers=default_headers,
    )

    assert chat_client.model_id == "gpt-4"
    assert isinstance(chat_client, ChatClientProtocol)

    # 追加したデフォルトヘッダーがclientのデフォルトヘッダーに存在することをアサート。
    for key, value in default_headers.items():
        assert key in chat_client.client.default_headers
        assert chat_client.client.default_headers[key] == value


def test_openai_assistants_client_instructions_sent_once(mock_async_openai: MagicMock) -> None:
    """OpenAI AssistantsのRequestでinstructionsが一度だけ含まれることを保証。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)
    instructions = "You are a helpful assistant."
    chat_options = ChatOptions(instructions=instructions)

    prepared_messages = chat_client.prepare_messages([ChatMessage(role=Role.USER, text="Hello")], chat_options)
    run_options, _ = chat_client._prepare_options(prepared_messages, chat_options)  # type: ignore[reportPrivateUsage]

    assert run_options.get("instructions") == instructions


async def test_openai_assistants_client_get_assistant_id_or_create_existing_assistant(
    mock_async_openai: MagicMock,
) -> None:
    """assistant_idが既に提供されている場合の_get_assistant_id_or_createのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai, assistant_id="existing-assistant-id")

    assistant_id = await chat_client._get_assistant_id_or_create()  # type: ignore

    assert assistant_id == "existing-assistant-id"
    assert not chat_client._should_delete_assistant  # type: ignore
    mock_async_openai.beta.assistants.create.assert_not_called()


async def test_openai_assistants_client_get_assistant_id_or_create_create_new(
    mock_async_openai: MagicMock,
) -> None:
    """新しいassistantを作成する場合の_get_assistant_id_or_createのテスト。"""
    chat_client = create_test_openai_assistants_client(
        mock_async_openai, model_id="gpt-4", assistant_name="TestAssistant"
    )

    assistant_id = await chat_client._get_assistant_id_or_create()  # type: ignore

    assert assistant_id == "test-assistant-id"
    assert chat_client._should_delete_assistant  # type: ignore
    mock_async_openai.beta.assistants.create.assert_called_once()


async def test_openai_assistants_client_aclose_should_not_delete(
    mock_async_openai: MagicMock,
) -> None:
    """assistantを削除しない場合のcloseのテスト。"""
    chat_client = create_test_openai_assistants_client(
        mock_async_openai, assistant_id="assistant-to-keep", should_delete_assistant=False
    )

    await chat_client.close()  # type: ignore

    # assistantの削除が呼ばれていないことを検証。
    mock_async_openai.beta.assistants.delete.assert_not_called()
    assert not chat_client._should_delete_assistant  # type: ignore


async def test_openai_assistants_client_aclose_should_delete(mock_async_openai: MagicMock) -> None:
    """closeメソッドがcleanupを呼び出すことのテスト。"""
    chat_client = create_test_openai_assistants_client(
        mock_async_openai, assistant_id="assistant-to-delete", should_delete_assistant=True
    )

    await chat_client.close()

    # assistantの削除が呼ばれたことを検証。
    mock_async_openai.beta.assistants.delete.assert_called_once_with("assistant-to-delete")
    assert not chat_client._should_delete_assistant  # type: ignore


async def test_openai_assistants_client_async_context_manager(mock_async_openai: MagicMock) -> None:
    """非同期コンテキストマネージャの機能テスト。"""
    chat_client = create_test_openai_assistants_client(
        mock_async_openai, assistant_id="assistant-to-delete", should_delete_assistant=True
    )

    # コンテキストマネージャのテスト。
    async with chat_client:
        pass  # 入退出ができることだけをテスト。

    # exit時にcleanupが呼ばれたことを検証。
    mock_async_openai.beta.assistants.delete.assert_called_once_with("assistant-to-delete")


def test_openai_assistants_client_serialize(openai_unit_test_env: dict[str, str]) -> None:
    """OpenAIAssistantsClientのシリアライズテスト。"""
    default_headers = {"X-Unit-Test": "test-guid"}

    # 基本的な初期化とto_dictのテスト。
    chat_client = OpenAIAssistantsClient(
        model_id="gpt-4",
        assistant_id="test-assistant-id",
        assistant_name="TestAssistant",
        thread_id="test-thread-id",
        api_key=openai_unit_test_env["OPENAI_API_KEY"],
        org_id=openai_unit_test_env["OPENAI_ORG_ID"],
        default_headers=default_headers,
    )

    dumped_settings = chat_client.to_dict()

    assert dumped_settings["model_id"] == "gpt-4"
    assert dumped_settings["assistant_id"] == "test-assistant-id"
    assert dumped_settings["assistant_name"] == "TestAssistant"
    assert dumped_settings["thread_id"] == "test-thread-id"
    assert dumped_settings["org_id"] == openai_unit_test_env["OPENAI_ORG_ID"]

    # 追加したデフォルトヘッダーがdumped_settingsのデフォルトヘッダーに存在することをアサート。
    for key, value in default_headers.items():
        assert key in dumped_settings["default_headers"]
        assert dumped_settings["default_headers"][key] == value
    # 'User-Agent'ヘッダーがdumped_settingsのデフォルトヘッダーに存在しないことをアサート。
    assert "User-Agent" not in dumped_settings["default_headers"]


async def test_openai_assistants_client_get_active_thread_run_none_thread_id(mock_async_openai: MagicMock) -> None:
    """thread_idがNoneの場合、_get_active_thread_runはNoneを返すことのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    result = await chat_client._get_active_thread_run(None)  # type: ignore

    assert result is None
    # thread_idがNoneの場合、APIを呼び出さないこと。
    mock_async_openai.beta.threads.runs.list.assert_not_called()


async def test_openai_assistants_client_get_active_thread_run_with_active_run(mock_async_openai: MagicMock) -> None:
    """_get_active_thread_runがアクティブなrunを見つけるテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # アクティブなrunをモック（ステータスが完了状態にない）。
    mock_run = MagicMock()
    mock_run.status = "in_progress"  # アクティブなステータス。

    # runs.listの非同期イテレータをモック。
    async def mock_runs_list(*args: Any, **kwargs: Any) -> Any:
        yield mock_run

    mock_async_openai.beta.threads.runs.list.return_value.__aiter__ = mock_runs_list

    result = await chat_client._get_active_thread_run("thread-123")  # type: ignore

    assert result == mock_run
    mock_async_openai.beta.threads.runs.list.assert_called_once_with(thread_id="thread-123", limit=1, order="desc")


async def test_openai_assistants_client_prepare_thread_create_new(mock_async_openai: MagicMock) -> None:
    """thread_idがNoneの場合、新しいthreadを作成する_prepare_threadのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # thread作成のモック。
    mock_thread = MagicMock()
    mock_thread.id = "new-thread-123"
    mock_async_openai.beta.threads.create.return_value = mock_thread

    # 追加メッセージを含むrunオプションの準備。
    run_options: dict[str, Any] = {
        "additional_messages": [{"role": "user", "content": "Hello"}],
        "tool_resources": {"code_interpreter": {}},
        "metadata": {"test": "true"},
    }

    result = await chat_client._prepare_thread(None, None, run_options)  # type: ignore

    assert result == "new-thread-123"
    assert run_options["additional_messages"] == []  # クリアされるべき。
    mock_async_openai.beta.threads.create.assert_called_once_with(
        messages=[{"role": "user", "content": "Hello"}],
        tool_resources={"code_interpreter": {}},
        metadata={"test": "true"},
    )


async def test_openai_assistants_client_prepare_thread_cancel_existing_run(mock_async_openai: MagicMock) -> None:
    """既存のrunが提供された場合、_prepare_threadがキャンセルするテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 既存のthread runをモック。
    mock_thread_run = MagicMock()
    mock_thread_run.id = "run-456"

    run_options: dict[str, Any] = {"additional_messages": []}

    result = await chat_client._prepare_thread("thread-123", mock_thread_run, run_options)  # type: ignore

    assert result == "thread-123"
    mock_async_openai.beta.threads.runs.cancel.assert_called_once_with(run_id="run-456", thread_id="thread-123")


async def test_openai_assistants_client_prepare_thread_existing_no_run(mock_async_openai: MagicMock) -> None:
    """既存のthread_idがあるがアクティブなrunがない場合の_prepare_threadのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    run_options: dict[str, list[dict[str, str]]] = {"additional_messages": []}

    result = await chat_client._prepare_thread("thread-123", None, run_options)  # type: ignore

    assert result == "thread-123"
    # thread_runが提供されていないためcancelを呼ばないべき。
    mock_async_openai.beta.threads.runs.cancel.assert_not_called()


async def test_openai_assistants_client_process_stream_events_thread_run_created(mock_async_openai: MagicMock) -> None:
    """thread.run.createdイベントでの_process_stream_eventsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # thread.run.createdのモックストリームレスポンスを作成。
    mock_response = MagicMock()
    mock_response.event = "thread.run.created"
    mock_response.data = MagicMock()

    # 適切な非同期イテレータを作成。
    async def async_iterator() -> Any:
        yield mock_response

    # レスポンスをyieldするモックストリームを作成。
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=async_iterator())
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    thread_id = "thread-123"
    updates: list[ChatResponseUpdate] = []
    async for update in chat_client._process_stream_events(mock_stream, thread_id):  # type: ignore
        updates.append(update)

    # thread.run.createdに対して1つのChatResponseUpdateをyieldすべき。
    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, ChatResponseUpdate)
    assert update.conversation_id == thread_id
    assert update.role == Role.ASSISTANT
    assert update.contents == []
    assert update.raw_representation == mock_response.data


async def test_openai_assistants_client_process_stream_events_message_delta_text(mock_async_openai: MagicMock) -> None:
    """thread.message.deltaイベントでテキストを含む_process_stream_eventsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 適切な仕様のTextDeltaBlockをモック作成。
    mock_delta_block = MagicMock(spec=TextDeltaBlock)
    mock_delta_block.text = MagicMock()
    mock_delta_block.text.value = "Hello from assistant"

    mock_delta = MagicMock()
    mock_delta.role = "assistant"
    mock_delta.content = [mock_delta_block]

    mock_message_delta = MagicMock(spec=MessageDeltaEvent)
    mock_message_delta.delta = mock_delta

    mock_response = MagicMock()
    mock_response.event = "thread.message.delta"
    mock_response.data = mock_message_delta

    # 適切な非同期イテレータを作成。
    async def async_iterator() -> Any:
        yield mock_response

    # モックストリームを作成。
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=async_iterator())
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    thread_id = "thread-456"
    updates: list[ChatResponseUpdate] = []
    async for update in chat_client._process_stream_events(mock_stream, thread_id):  # type: ignore
        updates.append(update)

    # 1つのテキスト更新をyieldすべき。
    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, ChatResponseUpdate)
    assert update.conversation_id == thread_id
    assert update.role == Role.ASSISTANT
    assert update.text == "Hello from assistant"
    assert update.raw_representation == mock_message_delta


async def test_openai_assistants_client_process_stream_events_requires_action(mock_async_openai: MagicMock) -> None:
    """thread.run.requires_actionイベントでの_process_stream_eventsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # _test_create_function_call_contentsメソッドをモックしてテスト用コンテンツを返す。
    test_function_content = FunctionCallContent(call_id="call-123", name="test_func", arguments={"arg": "value"})
    chat_client._create_function_call_contents = MagicMock(return_value=[test_function_content])  # type: ignore

    # モックRunオブジェクトを作成。
    mock_run = MagicMock(spec=Run)

    mock_response = MagicMock()
    mock_response.event = "thread.run.requires_action"
    mock_response.data = mock_run

    # 適切な非同期イテレータを作成。
    async def async_iterator() -> Any:
        yield mock_response

    # モックストリームを作成。
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=async_iterator())
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    thread_id = "thread-789"
    updates: list[ChatResponseUpdate] = []
    async for update in chat_client._process_stream_events(mock_stream, thread_id):  # type: ignore
        updates.append(update)

    # 1つのfunction call updateをyieldすべき。
    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, ChatResponseUpdate)
    assert update.conversation_id == thread_id
    assert update.role == Role.ASSISTANT
    assert len(update.contents) == 1
    assert update.contents[0] == test_function_content
    assert update.raw_representation == mock_run

    # _create_function_call_contentsが正しく呼ばれたことを検証。
    chat_client._create_function_call_contents.assert_called_once_with(mock_run, None)  # type: ignore


async def test_openai_assistants_client_process_stream_events_run_step_created(mock_async_openai: MagicMock) -> None:
    """thread.run.step.createdイベントでの_process_stream_eventsのテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # モックRunStepオブジェクトを作成。
    mock_run_step = MagicMock(spec=RunStep)
    mock_run_step.run_id = "run-456"

    mock_response = MagicMock()
    mock_response.event = "thread.run.step.created"
    mock_response.data = mock_run_step

    # 適切な非同期イテレータを作成。
    async def async_iterator() -> Any:
        yield mock_response

    # モックストリームを作成。
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=async_iterator())
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    thread_id = "thread-789"
    updates: list[ChatResponseUpdate] = []
    async for update in chat_client._process_stream_events(mock_stream, thread_id):  # type: ignore
        updates.append(update)

    # run stepの作成自体は更新をyieldしないが、後続イベントのためにresponse_idを設定すべき。
    assert len(updates) == 0


async def test_openai_assistants_client_process_stream_events_run_completed_with_usage(
    mock_async_openai: MagicMock,
) -> None:
    """usageを含むthread.run.completedイベントでの_process_stream_eventsのテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # usage情報を含むモックRunオブジェクトを作成。
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_run = MagicMock(spec=Run)
    mock_run.usage = mock_usage

    mock_response = MagicMock()
    mock_response.event = "thread.run.completed"
    mock_response.data = mock_run

    # 適切な非同期イテレータを作成。
    async def async_iterator() -> Any:
        yield mock_response

    # モックストリームを作成。
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=async_iterator())
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    thread_id = "thread-999"
    updates: list[ChatResponseUpdate] = []
    async for update in chat_client._process_stream_events(mock_stream, thread_id):  # type: ignore
        updates.append(update)

    # 1つのusage更新をyieldすべき。
    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, ChatResponseUpdate)
    assert update.conversation_id == thread_id
    assert update.role == Role.ASSISTANT
    assert len(update.contents) == 1

    # usage内容をチェック。
    usage_content = update.contents[0]
    assert isinstance(usage_content, UsageContent)
    assert usage_content.details.input_token_count == 100
    assert usage_content.details.output_token_count == 50
    assert usage_content.details.total_token_count == 150
    assert update.raw_representation == mock_run


def test_openai_assistants_client_create_function_call_contents_basic(mock_async_openai: MagicMock) -> None:
    """単純なfunction callでの_create_function_call_contentsのテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # actionを必要とするモックRunイベントを作成。
    mock_run = MagicMock()
    mock_run.required_action = MagicMock()
    mock_run.required_action.submit_tool_outputs = MagicMock()

    # モックツールコールを作成。
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Seattle"}'

    mock_run.required_action.submit_tool_outputs.tool_calls = [mock_tool_call]

    # メソッドを呼び出す。
    response_id = "response_456"
    contents = chat_client._create_function_call_contents(mock_run, response_id)  # type: ignore

    # 1つのfunction call contentが作成されたことをテスト。
    assert len(contents) == 1
    assert isinstance(contents[0], FunctionCallContent)
    assert contents[0].name == "get_weather"
    assert contents[0].arguments == {"location": "Seattle"}


def test_openai_assistants_client_prepare_options_basic(mock_async_openai: MagicMock) -> None:
    """基本的なchatオプションでの_prepare_optionsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 基本的なchatオプションを作成。
    chat_options = ChatOptions(
        max_tokens=100,
        model_id="gpt-4",
        temperature=0.7,
        top_p=0.9,
    )

    messages = [ChatMessage(role=Role.USER, text="Hello")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # 基本オプションが設定されたことをチェック。
    assert run_options["max_completion_tokens"] == 100
    assert run_options["model"] == "gpt-4"
    assert run_options["temperature"] == 0.7
    assert run_options["top_p"] == 0.9
    assert tool_results is None


def test_openai_assistants_client_prepare_options_with_ai_function_tool(mock_async_openai: MagicMock) -> None:
    """AIFunctionツールでの_prepare_optionsのテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # テスト用の単純な関数を作成しデコレート。
    @ai_function
    def test_function(query: str) -> str:
        """テスト用の関数。"""
        return f"Result for {query}"

    chat_options = ChatOptions(
        tools=[test_function],
        tool_choice="auto",
    )

    messages = [ChatMessage(role=Role.USER, text="Hello")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # ツールが正しく設定されたことをチェック。
    assert "tools" in run_options
    assert len(run_options["tools"]) == 1
    assert run_options["tools"][0]["type"] == "function"
    assert "function" in run_options["tools"][0]
    assert run_options["tool_choice"] == "auto"


def test_openai_assistants_client_prepare_options_with_code_interpreter(mock_async_openai: MagicMock) -> None:
    """HostedCodeInterpreterToolでの_prepare_optionsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 実際のHostedCodeInterpreterToolを作成。
    code_tool = HostedCodeInterpreterTool()

    chat_options = ChatOptions(
        tools=[code_tool],
        tool_choice="auto",
    )

    messages = [ChatMessage(role=Role.USER, text="Calculate something")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # コードインタプリタツールが正しく設定されたことをチェック。
    assert "tools" in run_options
    assert len(run_options["tools"]) == 1
    assert run_options["tools"][0] == {"type": "code_interpreter"}
    assert run_options["tool_choice"] == "auto"


def test_openai_assistants_client_prepare_options_tool_choice_none(mock_async_openai: MagicMock) -> None:
    """tool_choiceが'none'に設定された場合の_prepare_optionsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    chat_options = ChatOptions(
        tool_choice="none",
    )

    messages = [ChatMessage(role=Role.USER, text="Hello")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # tool_choiceがnoneに設定され、toolsが含まれないこと。
    assert run_options["tool_choice"] == "none"
    assert "tools" not in run_options


def test_openai_assistants_client_prepare_options_required_function(mock_async_openai: MagicMock) -> None:
    """必須のfunction tool choiceでの_prepare_optionsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 必須のfunction tool choiceを作成。
    tool_choice = ToolMode(mode="required", required_function_name="specific_function")

    chat_options = ChatOptions(
        tool_choice=tool_choice,
    )

    messages = [ChatMessage(role=Role.USER, text="Hello")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # 必須のfunction tool choiceが正しく設定されたことをチェック。
    expected_tool_choice = {
        "type": "function",
        "function": {"name": "specific_function"},
    }
    assert run_options["tool_choice"] == expected_tool_choice


def test_openai_assistants_client_prepare_options_with_file_search_tool(mock_async_openai: MagicMock) -> None:
    """HostedFileSearchToolでの_prepare_optionsのテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # max_results付きのHostedFileSearchToolを作成。
    file_search_tool = HostedFileSearchTool(max_results=10)

    chat_options = ChatOptions(
        tools=[file_search_tool],
        tool_choice="auto",
    )

    messages = [ChatMessage(role=Role.USER, text="Search for information")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # ファイル検索ツールが正しく設定されたことをチェック。
    assert "tools" in run_options
    assert len(run_options["tools"]) == 1
    expected_tool = {"type": "file_search", "max_num_results": 10}
    assert run_options["tools"][0] == expected_tool
    assert run_options["tool_choice"] == "auto"


def test_openai_assistants_client_prepare_options_with_mapping_tool(mock_async_openai: MagicMock) -> None:
    """MutableMappingツールでの_prepare_optionsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # MutableMapping（dict）としてのツールを作成。
    mapping_tool = {"type": "custom_tool", "parameters": {"setting": "value"}}

    chat_options = ChatOptions(
        tools=[mapping_tool],  # type: ignore
        tool_choice="auto",
    )

    messages = [ChatMessage(role=Role.USER, text="Use custom tool")]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, chat_options)  # type: ignore

    # マッピングツールが正しく設定されたことをチェック。
    assert "tools" in run_options
    assert len(run_options["tools"]) == 1
    assert run_options["tools"][0] == mapping_tool
    assert run_options["tool_choice"] == "auto"


def test_openai_assistants_client_prepare_options_with_system_message(mock_async_openai: MagicMock) -> None:
    """system messageをinstructionsに変換した場合の_prepare_optionsのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    messages = [
        ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant."),
        ChatMessage(role=Role.USER, text="Hello"),
    ]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, None)  # type: ignore

    # additional_messagesにはuser messageのみが含まれることをチェック system
    # messageはinstructionsに変換される（これは内部で処理される）。
    assert "additional_messages" in run_options
    assert len(run_options["additional_messages"]) == 1
    assert run_options["additional_messages"][0]["role"] == "user"


def test_openai_assistants_client_prepare_options_with_image_content(mock_async_openai: MagicMock) -> None:
    """画像コンテンツを含む_prepare_optionsのテスト。"""

    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 画像コンテンツを含むメッセージを作成。
    image_content = UriContent(uri="https://example.com/image.jpg", media_type="image/jpeg")
    messages = [ChatMessage(role=Role.USER, contents=[image_content])]

    # メソッドを呼び出す。
    run_options, tool_results = chat_client._prepare_options(messages, None)  # type: ignore

    # 画像コンテンツが処理されたことをチェック。
    assert "additional_messages" in run_options
    assert len(run_options["additional_messages"]) == 1
    message = run_options["additional_messages"][0]
    assert message["role"] == "user"
    assert len(message["content"]) == 1
    assert message["content"][0]["type"] == "image_url"
    assert message["content"][0]["image_url"]["url"] == "https://example.com/image.jpg"


def test_openai_assistants_client_convert_function_results_to_tool_output_empty(mock_async_openai: MagicMock) -> None:
    """空リストでの_convert_function_results_to_tool_outputのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    run_id, tool_outputs = chat_client._convert_function_results_to_tool_output([])  # type: ignore

    assert run_id is None
    assert tool_outputs is None


def test_openai_assistants_client_convert_function_results_to_tool_output_valid(mock_async_openai: MagicMock) -> None:
    """有効なfunction resultsでの_convert_function_results_to_tool_outputのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    call_id = json.dumps(["run-123", "call-456"])
    function_result = FunctionResultContent(call_id=call_id, result="Function executed successfully")

    run_id, tool_outputs = chat_client._convert_function_results_to_tool_output([function_result])  # type: ignore

    assert run_id == "run-123"
    assert tool_outputs is not None
    assert len(tool_outputs) == 1
    assert tool_outputs[0].get("tool_call_id") == "call-456"
    assert tool_outputs[0].get("output") == "Function executed successfully"


def test_openai_assistants_client_convert_function_results_to_tool_output_mismatched_run_ids(
    mock_async_openai: MagicMock,
) -> None:
    """run IDが不一致の場合の_convert_function_results_to_tool_outputのテスト。"""
    chat_client = create_test_openai_assistants_client(mock_async_openai)

    # 異なるrun IDを持つfunction resultsを作成。
    call_id1 = json.dumps(["run-123", "call-456"])
    call_id2 = json.dumps(["run-789", "call-xyz"])  # 異なるrun ID。
    function_result1 = FunctionResultContent(call_id=call_id1, result="Result 1")
    function_result2 = FunctionResultContent(call_id=call_id2, result="Result 2")

    run_id, tool_outputs = chat_client._convert_function_results_to_tool_output([function_result1, function_result2])  # type: ignore

    # run IDが一致しないため最初の1つだけ処理されるべき。
    assert run_id == "run-123"
    assert tool_outputs is not None
    assert len(tool_outputs) == 1
    assert tool_outputs[0].get("tool_call_id") == "call-456"


def test_openai_assistants_client_update_agent_name(mock_async_openai: MagicMock) -> None:
    """assistant_nameが未設定の場合に_update_agent_nameメソッドが更新するテスト。"""
    # assistant_nameがNoneの場合のagent name更新テスト。
    chat_client = create_test_openai_assistants_client(mock_async_openai, assistant_name=None)

    # agent nameを更新するプライベートメソッドを呼び出す。
    chat_client._update_agent_name("New Assistant Name")  # type: ignore

    assert chat_client.assistant_name == "New Assistant Name"


def test_openai_assistants_client_update_agent_name_existing(mock_async_openai: MagicMock) -> None:
    """既存のassistant_nameを上書きしない_update_agent_nameメソッドのテスト。"""
    # 既存のassistant_nameが上書きされないことをテスト。
    chat_client = create_test_openai_assistants_client(mock_async_openai, assistant_name="Existing Assistant")

    # agent nameを更新するプライベートメソッドを呼び出す。
    chat_client._update_agent_name("New Assistant Name")  # type: ignore

    # 既存の名前を保持すべき。
    assert chat_client.assistant_name == "Existing Assistant"


def test_openai_assistants_client_update_agent_name_none(mock_async_openai: MagicMock) -> None:
    """Noneのagent_nameパラメータでの_update_agent_nameメソッドのテスト。"""
    # Noneのagent_nameが何も変更しないことをテスト。
    chat_client = create_test_openai_assistants_client(mock_async_openai, assistant_name=None)

    # Noneを渡してプライベートメソッドを呼び出す。
    chat_client._update_agent_name(None)  # type: ignore

    # Noneのままであるべき。
    assert chat_client.assistant_name is None


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得します。"""
    return f"The weather in {location} is sunny with a high of 25°C."


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_client_get_response() -> None:
    """OpenAI Assistants Clientのレスポンスをテストします。"""
    async with OpenAIAssistantsClient() as openai_assistants_client:
        assert isinstance(openai_assistants_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(
            ChatMessage(
                role="user",
                text="The weather in Seattle is currently sunny with a high of 25°C. "
                "It's a beautiful day for outdoor activities.",
            )
        )
        messages.append(ChatMessage(role="user", text="What's the weather like today?"))

        # クライアントがレスポンスを取得できることをテストします。
        response = await openai_assistants_client.get_response(messages=messages)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert any(word in response.text.lower() for word in ["sunny", "25", "weather", "seattle"])


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_client_get_response_tools() -> None:
    """ツールを使ったOpenAI Assistants Clientのレスポンスをテストします。"""
    async with OpenAIAssistantsClient() as openai_assistants_client:
        assert isinstance(openai_assistants_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(ChatMessage(role="user", text="What's the weather like in Seattle?"))

        # クライアントがレスポンスを取得できることをテストします。
        response = await openai_assistants_client.get_response(
            messages=messages,
            tools=[get_weather],
            tool_choice="auto",
        )

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert any(word in response.text.lower() for word in ["sunny", "25", "weather"])


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_client_streaming() -> None:
    """OpenAI Assistants Clientのストリーミングレスポンスをテストします。"""
    async with OpenAIAssistantsClient() as openai_assistants_client:
        assert isinstance(openai_assistants_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(
            ChatMessage(
                role="user",
                text="The weather in Seattle is currently sunny with a high of 25°C. "
                "It's a beautiful day for outdoor activities.",
            )
        )
        messages.append(ChatMessage(role="user", text="What's the weather like today?"))

        # クライアントがレスポンスを取得できることをテストします。
        response = openai_assistants_client.get_streaming_response(messages=messages)

        full_message: str = ""
        async for chunk in response:
            assert chunk is not None
            assert isinstance(chunk, ChatResponseUpdate)
            for content in chunk.contents:
                if isinstance(content, TextContent) and content.text:
                    full_message += content.text

        assert any(word in full_message.lower() for word in ["sunny", "25", "weather", "seattle"])


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_client_streaming_tools() -> None:
    """ツールを使ったOpenAI Assistants Clientのストリーミングレスポンスをテストします。"""
    async with OpenAIAssistantsClient() as openai_assistants_client:
        assert isinstance(openai_assistants_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(ChatMessage(role="user", text="What's the weather like in Seattle?"))

        # クライアントがレスポンスを取得できることをテストします。
        response = openai_assistants_client.get_streaming_response(
            messages=messages,
            tools=[get_weather],
            tool_choice="auto",
        )
        full_message: str = ""
        async for chunk in response:
            assert chunk is not None
            assert isinstance(chunk, ChatResponseUpdate)
            for content in chunk.contents:
                if isinstance(content, TextContent) and content.text:
                    full_message += content.text

        assert any(word in full_message.lower() for word in ["sunny", "25", "weather"])


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_client_with_existing_assistant() -> None:
    """既存のassistant IDを使ったOpenAI Assistants Clientをテストします。"""
    # テストで使用するために最初にassistantを作成します。
    async with OpenAIAssistantsClient() as temp_client:
        # assistant作成をトリガーしてassistant IDを取得します。
        messages = [ChatMessage(role="user", text="Hello")]
        await temp_client.get_response(messages=messages)
        assistant_id = temp_client.assistant_id

        # 既存のassistantを使ってテストします。
        async with OpenAIAssistantsClient(
            model_id="gpt-4o-mini", assistant_id=assistant_id
        ) as openai_assistants_client:
            assert isinstance(openai_assistants_client, ChatClientProtocol)
            assert openai_assistants_client.assistant_id == assistant_id

            messages = [ChatMessage(role="user", text="What can you do?")]

            # クライアントがレスポンスを取得できることをテストします。
            response = await openai_assistants_client.get_response(messages=messages)

            assert response is not None
            assert isinstance(response, ChatResponse)
            assert len(response.text) > 0


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
@pytest.mark.skip(reason="OpenAI file search functionality is currently broken - tracked in GitHub issue")
async def test_openai_assistants_client_file_search() -> None:
    """OpenAI Assistants Clientのレスポンスをテストします。"""
    async with OpenAIAssistantsClient() as openai_assistants_client:
        assert isinstance(openai_assistants_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(ChatMessage(role="user", text="What's the weather like today?"))

        file_id, vector_store = await create_vector_store(openai_assistants_client)
        response = await openai_assistants_client.get_response(
            messages=messages,
            tools=[HostedFileSearchTool()],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}},
        )
        await delete_vector_store(openai_assistants_client, file_id, vector_store.vector_store_id)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert any(word in response.text.lower() for word in ["sunny", "25", "weather"])


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
@pytest.mark.skip(reason="OpenAI file search functionality is currently broken - tracked in GitHub issue")
async def test_openai_assistants_client_file_search_streaming() -> None:
    """OpenAI Assistants Clientのレスポンスをテストします。"""
    async with OpenAIAssistantsClient() as openai_assistants_client:
        assert isinstance(openai_assistants_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(ChatMessage(role="user", text="What's the weather like today?"))

        file_id, vector_store = await create_vector_store(openai_assistants_client)
        response = openai_assistants_client.get_streaming_response(
            messages=messages,
            tools=[HostedFileSearchTool()],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.vector_store_id]}},
        )

        assert response is not None
        full_message: str = ""
        async for chunk in response:
            assert chunk is not None
            assert isinstance(chunk, ChatResponseUpdate)
            for content in chunk.contents:
                if isinstance(content, TextContent) and content.text:
                    full_message += content.text
        await delete_vector_store(openai_assistants_client, file_id, vector_store.vector_store_id)

        assert any(word in full_message.lower() for word in ["sunny", "25", "weather"])


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_agent_basic_run():
    """OpenAIAssistantsClientを使ったChatAgentの基本的な実行機能をテストします。"""
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
    ) as agent:
        # 簡単なクエリを実行します。
        response = await agent.run("Hello! Please respond with 'Hello World' exactly.")

        # レスポンスを検証します。
        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0
        assert "Hello World" in response.text


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_agent_basic_run_streaming():
    """OpenAIAssistantsClientを使ったChatAgentの基本的なストリーミング機能をテストします。"""
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
    ) as agent:
        # ストリーミングクエリを実行します。
        full_message: str = ""
        async for chunk in agent.run_stream("Please respond with exactly: 'This is a streaming response test.'"):
            assert chunk is not None
            assert isinstance(chunk, AgentRunResponseUpdate)
            if chunk.text:
                full_message += chunk.text

        # ストリーミングレスポンスを検証します。
        assert len(full_message) > 0
        assert "streaming response test" in full_message.lower()


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_agent_thread_persistence():
    """OpenAIAssistantsClientを使ったChatAgentの実行間のスレッド永続性をテストします。"""
    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant with good memory.",
    ) as agent:
        # 再利用される新しいスレッドを作成します。
        thread = agent.get_new_thread()

        # 最初のメッセージ - コンテキストを確立します。
        first_response = await agent.run(
            "Remember this number: 42. What number did I just tell you to remember?", thread=thread
        )
        assert isinstance(first_response, AgentRunResponse)
        assert "42" in first_response.text

        # 2番目のメッセージ - 会話の記憶をテストします。
        second_response = await agent.run(
            "What number did I tell you to remember in my previous message?", thread=thread
        )
        assert isinstance(second_response, AgentRunResponse)
        assert "42" in second_response.text

        # スレッドに会話IDが設定されていることを確認します。
        assert thread.service_thread_id is not None


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_agent_existing_thread_id():
    """既存のスレッドIDを使ってエージェントインスタンス間で会話を継続するChatAgentをテストします。"""
    # まず会話を作成し、スレッドIDを取得します。
    existing_thread_id = None

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    ) as agent:
        # 会話を開始してスレッドIDを取得します。
        thread = agent.get_new_thread()
        response1 = await agent.run("What's the weather in Paris?", thread=thread)

        # 最初のレスポンスを検証します。
        assert isinstance(response1, AgentRunResponse)
        assert response1.text is not None
        assert any(word in response1.text.lower() for word in ["weather", "paris"])

        # 最初のレスポンス後にスレッドIDが設定されます。
        existing_thread_id = thread.service_thread_id
        assert existing_thread_id is not None

    # 同じスレッドIDを使って新しいエージェントインスタンスで続行します。

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(thread_id=existing_thread_id),
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    ) as agent:
        # 既存のIDでスレッドを作成します。
        thread = AgentThread(service_thread_id=existing_thread_id)

        # 前回の会話について質問します。
        response2 = await agent.run("What was the last city I asked about?", thread=thread)

        # エージェントが前回の会話を覚えていることを検証します。
        assert isinstance(response2, AgentRunResponse)
        assert response2.text is not None
        # 前回の会話でのパリについて言及するはずです。
        assert "paris" in response2.text.lower()


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_agent_code_interpreter():
    """OpenAIAssistantsClientを通じてコードインタープリターを使うChatAgentをテストします。"""

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that can write and execute Python code.",
        tools=[HostedCodeInterpreterTool()],
    ) as agent:
        # コード実行をリクエストします。
        response = await agent.run("Write Python code to calculate the factorial of 5 and show the result.")

        # レスポンスを検証します。
        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        # 5の階乗は120です。
        assert "120" in response.text or "factorial" in response.text.lower()


@pytest.mark.flaky
@skip_if_openai_integration_tests_disabled
async def test_openai_assistants_client_agent_level_tool_persistence():
    """OpenAI Assistants Clientを使ってエージェントレベルのツールが複数回の実行で持続することをテストします。"""

    async with ChatAgent(
        chat_client=OpenAIAssistantsClient(),
        instructions="You are a helpful assistant that uses available tools.",
        tools=[get_weather],  # Agent-level tool
    ) as agent:
        # 最初の実行 - エージェントレベルのツールが利用可能であるべきです。
        first_response = await agent.run("What's the weather like in Chicago?")

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None
        # エージェントレベルのweatherツールを使うはずです。
        assert any(term in first_response.text.lower() for term in ["chicago", "sunny", "72"])

        # 2回目の実行 - エージェントレベルのツールは依然として利用可能であるべきです（持続性テスト）。
        second_response = await agent.run("What's the weather in Miami?")

        assert isinstance(second_response, AgentRunResponse)
        assert second_response.text is not None
        # 再びエージェントレベルのweatherツールを使うはずです。
        assert any(term in second_response.text.lower() for term in ["miami", "sunny", "72"])


# 呼び出し可能なAPIキーのテスト。
def test_openai_assistants_client_with_callable_api_key() -> None:
    """呼び出し可能なAPIキーでOpenAIAssistantsClientの初期化をテストします。"""

    async def get_api_key() -> str:
        return "test-api-key-123"

    client = OpenAIAssistantsClient(model_id="gpt-4o", api_key=get_api_key)

    # クライアントが正常に作成されたことを検証します。
    assert client.model_id == "gpt-4o"
    # OpenAI SDKは現在、呼び出し可能なAPIキーを内部で管理しています。
    assert client.client is not None
