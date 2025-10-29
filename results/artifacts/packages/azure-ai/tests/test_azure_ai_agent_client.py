# Copyright (c) Microsoft. All rights reserved.

import json
import os
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    AIFunction,
    ChatAgent,
    ChatClientProtocol,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    CitationAnnotation,
    FunctionApprovalRequestContent,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    FunctionResultContent,
    HostedCodeInterpreterTool,
    HostedFileSearchTool,
    HostedMCPTool,
    HostedVectorStoreContent,
    HostedWebSearchTool,
    Role,
    TextContent,
    UriContent,
)
from agent_framework._serialization import SerializationMixin
from agent_framework.exceptions import ServiceInitializationError
from azure.ai.agents.models import (
    CodeInterpreterToolDefinition,
    FileInfo,
    MessageDeltaChunk,
    MessageDeltaTextContent,
    MessageDeltaTextUrlCitationAnnotation,
    RequiredFunctionToolCall,
    RequiredMcpToolCall,
    RunStatus,
    SubmitToolApprovalAction,
    SubmitToolOutputsAction,
    ThreadRun,
    VectorStore,
)
from azure.ai.projects.models import ConnectionType
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity.aio import AzureCliCredential
from pydantic import BaseModel, Field, ValidationError
from pytest import MonkeyPatch

from agent_framework_azure_ai import AzureAIAgentClient, AzureAISettings

skip_if_azure_ai_integration_tests_disabled = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
    or os.getenv("AZURE_AI_PROJECT_ENDPOINT", "") in ("", "https://test-project.cognitiveservices.azure.com/"),
    reason="No real AZURE_AI_PROJECT_ENDPOINT provided; skipping integration tests."
    if os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    else "Integration tests are disabled.",
)


def create_test_azure_ai_chat_client(
    mock_ai_project_client: MagicMock,
    agent_id: str | None = None,
    thread_id: str | None = None,
    azure_ai_settings: AzureAISettings | None = None,
    should_delete_agent: bool = False,
    agent_name: str | None = None,
) -> AzureAIAgentClient:
    """通常の検証をバイパスしてテスト用に AzureAIAgentClient インスタンスを作成するヘルパー関数。"""
    if azure_ai_settings is None:
        azure_ai_settings = AzureAISettings(env_file_path="test.env")

    # クライアントインスタンスを直接作成します。
    client = object.__new__(AzureAIAgentClient)

    # 属性を直接設定します。
    client.project_client = mock_ai_project_client
    client.credential = None
    client.agent_id = agent_id
    client.agent_name = agent_name
    client.model_id = azure_ai_settings.model_deployment_name
    client.thread_id = thread_id
    client._should_delete_agent = should_delete_agent  # type: ignore
    client._should_close_client = False  # type: ignore
    client._agent_definition = None  # type: ignore
    client.additional_properties = {}
    client.middleware = None

    return client


def test_azure_ai_settings_init(azure_ai_unit_test_env: dict[str, str]) -> None:
    """AzureAISettings の初期化をテストします。"""
    settings = AzureAISettings()

    assert settings.project_endpoint == azure_ai_unit_test_env["AZURE_AI_PROJECT_ENDPOINT"]
    assert settings.model_deployment_name == azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"]


def test_azure_ai_settings_init_with_explicit_values() -> None:
    """明示的な値での AzureAISettings 初期化をテストします。"""
    settings = AzureAISettings(
        project_endpoint="https://custom-endpoint.com/",
        model_deployment_name="custom-model",
    )

    assert settings.project_endpoint == "https://custom-endpoint.com/"
    assert settings.model_deployment_name == "custom-model"


def test_azure_ai_chat_client_init_with_client(mock_ai_project_client: MagicMock) -> None:
    """既存の project_client での AzureAIAgentClient 初期化をテストします。"""
    chat_client = create_test_azure_ai_chat_client(
        mock_ai_project_client, agent_id="existing-agent-id", thread_id="test-thread-id"
    )

    assert chat_client.project_client is mock_ai_project_client
    assert chat_client.agent_id == "existing-agent-id"
    assert chat_client.thread_id == "test-thread-id"
    assert not chat_client._should_delete_agent  # type: ignore
    assert isinstance(chat_client, ChatClientProtocol)


def test_azure_ai_chat_client_init_auto_create_client(
    azure_ai_unit_test_env: dict[str, str],
    mock_ai_project_client: MagicMock,
) -> None:
    """自動作成された project_client での AzureAIAgentClient 初期化をテストします。"""
    azure_ai_settings = AzureAISettings(**azure_ai_unit_test_env)  # type: ignore

    # クライアントインスタンスを直接作成します。
    chat_client = object.__new__(AzureAIAgentClient)
    chat_client.project_client = mock_ai_project_client
    chat_client.agent_id = None
    chat_client.thread_id = None
    chat_client._should_delete_agent = False
    chat_client._should_close_client = False
    chat_client.credential = None
    chat_client.model_id = azure_ai_settings.model_deployment_name
    chat_client.agent_name = None
    chat_client.additional_properties = {}
    chat_client.middleware = None

    assert chat_client.project_client is mock_ai_project_client
    assert chat_client.agent_id is None
    assert not chat_client._should_delete_agent  # type: ignore


def test_azure_ai_chat_client_init_missing_project_endpoint() -> None:
    """project_endpoint がなく project_client が提供されていない場合の AzureAIAgentClient 初期化をテストします。"""
    # None の project_endpoint を返すように AzureAISettings をモックします。
    with patch("agent_framework_azure_ai._chat_client.AzureAISettings") as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.project_endpoint = None  # これによりエラーが発生するはずです。
        mock_settings_instance.model_deployment_name = "test-model"
        mock_settings_instance.agent_name = "test-agent"
        mock_settings.return_value = mock_settings_instance

        with pytest.raises(ServiceInitializationError, match="project endpoint is required"):
            AzureAIAgentClient(
                project_client=None,
                agent_id=None,
                project_endpoint=None,  # Missing endpoint
                model_deployment_name="test-model",
                async_credential=AsyncMock(spec=AsyncTokenCredential),
            )


def test_azure_ai_chat_client_init_missing_model_deployment_for_agent_creation() -> None:
    """agent 作成時にモデルデプロイメントがない場合の AzureAIAgentClient 初期化をテストします。"""
    # None の model_deployment_name を返すように AzureAISettings をモックします。
    with patch("agent_framework_azure_ai._chat_client.AzureAISettings") as mock_settings:
        mock_settings_instance = MagicMock()
        mock_settings_instance.project_endpoint = "https://test.com"
        mock_settings_instance.model_deployment_name = None  # これによりエラーが発生するはずです。
        mock_settings_instance.agent_name = "test-agent"
        mock_settings.return_value = mock_settings_instance

        with pytest.raises(ServiceInitializationError, match="model deployment name is required"):
            AzureAIAgentClient(
                project_client=None,
                agent_id=None,  # No existing agent
                project_endpoint="https://test.com",
                model_deployment_name=None,  # Missing for agent creation
                async_credential=AsyncMock(spec=AsyncTokenCredential),
            )


def test_azure_ai_chat_client_from_dict(mock_ai_project_client: MagicMock) -> None:
    """AzureAIAgentClient.from_dict メソッドをテストします。"""
    settings = {
        "project_client": mock_ai_project_client,
        "agent_id": "test-agent-id",
        "thread_id": "test-thread-id",
        "project_endpoint": "https://test-endpoint.com/",
        "model_deployment_name": "test-model",
        "agent_name": "TestAgent",
    }

    azure_ai_settings = AzureAISettings(
        project_endpoint=settings["project_endpoint"],
        model_deployment_name=settings["model_deployment_name"],
    )

    chat_client: AzureAIAgentClient = create_test_azure_ai_chat_client(
        mock_ai_project_client,
        agent_id=settings["agent_id"],  # type: ignore
        thread_id=settings["thread_id"],  # type: ignore
        azure_ai_settings=azure_ai_settings,
    )

    assert chat_client.project_client is mock_ai_project_client
    assert chat_client.agent_id == "test-agent-id"
    assert chat_client.thread_id == "test-thread-id"


def test_azure_ai_chat_client_init_missing_credential(azure_ai_unit_test_env: dict[str, str]) -> None:
    """async_credential がなく project_client が提供されていない場合の AzureAIAgentClient.__init__ をテストします。"""
    with pytest.raises(
        ServiceInitializationError, match="Azure credential is required when project_client is not provided"
    ):
        AzureAIAgentClient(
            project_client=None,
            agent_id="existing-agent",
            project_endpoint=azure_ai_unit_test_env["AZURE_AI_PROJECT_ENDPOINT"],
            model_deployment_name=azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            async_credential=None,  # Missing credential
        )


def test_azure_ai_chat_client_init_validation_error(mock_azure_credential: MagicMock) -> None:
    """AzureAISettings の ValidationError が適切に処理されることをテストします。"""
    with patch("agent_framework_azure_ai._chat_client.AzureAISettings") as mock_settings:
        # 空の errors リストと model 辞書で適切な ValidationError を作成します。
        mock_settings.side_effect = ValidationError.from_exception_data("AzureAISettings", [])

        with pytest.raises(ServiceInitializationError, match="Failed to create Azure AI settings."):
            AzureAIAgentClient(
                project_endpoint="https://test.com",
                model_deployment_name="test-model",
                async_credential=mock_azure_credential,
            )


def test_azure_ai_chat_client_from_settings() -> None:
    """from_settings クラスメソッドをテストします。"""
    mock_project_client = MagicMock()
    settings = {
        "project_client": mock_project_client,
        "agent_id": "test-agent",
        "thread_id": "test-thread",
        "project_endpoint": "https://test.com",
        "model_deployment_name": "test-model",
        "agent_name": "TestAgent",
    }

    client = AzureAIAgentClient.from_settings(settings)

    assert client.project_client is mock_project_client
    assert client.agent_id == "test-agent"
    assert client.thread_id == "test-thread"
    assert client.agent_name == "TestAgent"


async def test_azure_ai_chat_client_get_agent_id_or_create_existing_agent(
    mock_ai_project_client: MagicMock,
) -> None:
    """agent_id が既に提供されている場合の _get_agent_id_or_create をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="existing-agent-id")

    agent_id = await chat_client._get_agent_id_or_create()  # type: ignore

    assert agent_id == "existing-agent-id"
    assert not chat_client._should_delete_agent  # type: ignore


async def test_azure_ai_chat_client_get_agent_id_or_create_create_new(
    mock_ai_project_client: MagicMock,
    azure_ai_unit_test_env: dict[str, str],
) -> None:
    """新しい agent を作成する場合の _get_agent_id_or_create をテストします。"""
    azure_ai_settings = AzureAISettings(model_deployment_name=azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, azure_ai_settings=azure_ai_settings)

    agent_id = await chat_client._get_agent_id_or_create(run_options={"model": azure_ai_settings.model_deployment_name})  # type: ignore

    assert agent_id == "test-agent-id"
    assert chat_client._should_delete_agent  # type: ignore


async def test_azure_ai_chat_client_tool_results_without_thread_error_via_public_api(
    mock_ai_project_client: MagicMock,
) -> None:
    """thread ID なしのツール結果がパブリック API 経由でエラーを発生させることをテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # get_agent をモックします。
    mock_ai_project_client.agents.get_agent = AsyncMock(return_value=None)

    # thread/conversation ID なしのツール結果を含むメッセージを作成します。
    messages = [
        ChatMessage(role=Role.USER, text="Hello"),
        ChatMessage(
            role=Role.TOOL, contents=[FunctionResultContent(call_id='["run_123", "call_456"]', result="Result")]
        ),
    ]

    # パブリック API 経由で呼び出すと ValueError を発生させるはずです。
    with pytest.raises(ValueError, match="No thread ID was provided, but chat messages includes tool results"):
        async for _ in chat_client.get_streaming_response(messages):
            pass


async def test_azure_ai_chat_client_thread_management_through_public_api(mock_ai_project_client: MagicMock) -> None:
    """パブリック API を通じたスレッド作成と管理をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 非同期エラーを回避するために get_agent をモックします。
    mock_ai_project_client.agents.get_agent = AsyncMock(return_value=None)

    mock_thread = MagicMock()
    mock_thread.id = "new-thread-456"
    mock_ai_project_client.agents.threads.create = AsyncMock(return_value=mock_thread)

    mock_stream = AsyncMock()
    mock_ai_project_client.agents.runs.stream = AsyncMock(return_value=mock_stream)

    # 何も生成しない非同期イテレータを作成します（空のストリーム）。
    async def empty_async_iter():
        return
        yield  # これをジェネレータにします（到達不能）。

    mock_stream.__aenter__ = AsyncMock(return_value=empty_async_iter())
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    messages = [ChatMessage(role=Role.USER, text="Hello")]

    # 既存のスレッドなしで呼び出し - 新しいスレッドを作成するはずです。
    response = chat_client.get_streaming_response(messages)
    # メソッド実行をトリガーするためにジェネレータを消費します。
    async for _ in response:
        pass

    # スレッド作成が呼び出されたことを検証します。
    mock_ai_project_client.agents.threads.create.assert_called_once()


@pytest.mark.parametrize("exclude_list", [["AZURE_AI_MODEL_DEPLOYMENT_NAME"]], indirect=True)
async def test_azure_ai_chat_client_get_agent_id_or_create_missing_model(
    mock_ai_project_client: MagicMock, azure_ai_unit_test_env: dict[str, str]
) -> None:
    """model_deployment_name がない場合の _get_agent_id_or_create をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    with pytest.raises(ServiceInitializationError, match="Model deployment name is required"):
        await chat_client._get_agent_id_or_create()  # type: ignore


async def test_azure_ai_chat_client_cleanup_agent_if_needed_should_delete(
    mock_ai_project_client: MagicMock,
) -> None:
    """agent を削除すべき場合の _cleanup_agent_if_needed をテストします。"""
    chat_client = create_test_azure_ai_chat_client(
        mock_ai_project_client, agent_id="agent-to-delete", should_delete_agent=True
    )

    await chat_client._cleanup_agent_if_needed()  # type: ignore
    # agent 削除が呼び出されたことを検証します。
    mock_ai_project_client.agents.delete_agent.assert_called_once_with("agent-to-delete")
    assert not chat_client._should_delete_agent  # type: ignore


async def test_azure_ai_chat_client_cleanup_agent_if_needed_should_not_delete(
    mock_ai_project_client: MagicMock,
) -> None:
    """agent を削除すべきでない場合の _cleanup_agent_if_needed をテストします。"""
    chat_client = create_test_azure_ai_chat_client(
        mock_ai_project_client, agent_id="agent-to-keep", should_delete_agent=False
    )

    await chat_client._cleanup_agent_if_needed()  # type: ignore

    # agent 削除が呼び出されなかったことを検証します。
    mock_ai_project_client.agents.delete_agent.assert_not_called()
    assert not chat_client._should_delete_agent  # type: ignore


async def test_azure_ai_chat_client_cleanup_agent_if_needed_exception_handling(
    mock_ai_project_client: MagicMock,
) -> None:
    """_cleanup_agent_if_needed が例外を伝播することをテストします（例外は処理しません）。"""
    chat_client = create_test_azure_ai_chat_client(
        mock_ai_project_client, agent_id="agent-to-delete", should_delete_agent=True
    )
    mock_ai_project_client.agents.delete_agent.side_effect = Exception("Deletion failed")

    with pytest.raises(Exception, match="Deletion failed"):
        await chat_client._cleanup_agent_if_needed()  # type: ignore


async def test_azure_ai_chat_client_aclose(mock_ai_project_client: MagicMock) -> None:
    """aclose メソッドが cleanup を呼び出すことをテストします。"""
    chat_client = create_test_azure_ai_chat_client(
        mock_ai_project_client, agent_id="agent-to-delete", should_delete_agent=True
    )

    await chat_client.close()

    # agent 削除が呼び出されたことを検証します。
    mock_ai_project_client.agents.delete_agent.assert_called_once_with("agent-to-delete")


async def test_azure_ai_chat_client_async_context_manager(mock_ai_project_client: MagicMock) -> None:
    """非同期コンテキストマネージャの機能をテストします。"""
    chat_client = create_test_azure_ai_chat_client(
        mock_ai_project_client, agent_id="agent-to-delete", should_delete_agent=True
    )

    # コンテキストマネージャをテストします。
    async with chat_client:
        pass  # 単に入退出できることをテストします。

    # 退出時に cleanup が呼び出されたことを検証します。
    mock_ai_project_client.agents.delete_agent.assert_called_once_with("agent-to-delete")


async def test_azure_ai_chat_client_create_run_options_basic(mock_ai_project_client: MagicMock) -> None:
    """基本的な ChatOptions で _create_run_options をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    messages = [ChatMessage(role=Role.USER, text="Hello")]
    chat_options = ChatOptions(max_tokens=100, temperature=0.7)

    run_options, tool_results = await chat_client._create_run_options(messages, chat_options)  # type: ignore

    assert run_options is not None
    assert tool_results is None


async def test_azure_ai_chat_client_create_run_options_no_chat_options(mock_ai_project_client: MagicMock) -> None:
    """ChatOptions なしで _create_run_options をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    messages = [ChatMessage(role=Role.USER, text="Hello")]

    run_options, tool_results = await chat_client._create_run_options(messages, None)  # type: ignore

    assert run_options is not None
    assert tool_results is None


async def test_azure_ai_chat_client_create_run_options_with_image_content(mock_ai_project_client: MagicMock) -> None:
    """画像コンテンツで _create_run_options をテストします。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # get_agent をモックします。
    mock_ai_project_client.agents.get_agent = AsyncMock(return_value=None)

    image_content = UriContent(uri="https://example.com/image.jpg", media_type="image/jpeg")
    messages = [ChatMessage(role=Role.USER, contents=[image_content])]

    run_options, _ = await chat_client._create_run_options(messages, None)  # type: ignore

    assert "additional_messages" in run_options
    assert len(run_options["additional_messages"]) == 1
    # 画像が MessageInputImageUrlBlock に変換されたことを検証します。
    message = run_options["additional_messages"][0]
    assert len(message.content) == 1


def test_azure_ai_chat_client_convert_function_results_to_tool_output_none(mock_ai_project_client: MagicMock) -> None:
    """None 入力で _convert_required_action_to_tool_output をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    run_id, tool_outputs, tool_approvals = chat_client._convert_required_action_to_tool_output(None)  # type: ignore

    assert run_id is None
    assert tool_outputs is None
    assert tool_approvals is None


async def test_azure_ai_chat_client_close_client_when_should_close_true(mock_ai_project_client: MagicMock) -> None:
    """should_close_client が True のとき _close_client_if_needed が project_client を閉じることをテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)
    chat_client._should_close_client = True  # type: ignore

    mock_ai_project_client.close = AsyncMock()

    await chat_client._close_client_if_needed()  # type: ignore

    mock_ai_project_client.close.assert_called_once()


async def test_azure_ai_chat_client_close_client_when_should_close_false(mock_ai_project_client: MagicMock) -> None:
    """should_close_client が False のとき _close_client_if_needed が project_client を閉じないことをテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)
    chat_client._should_close_client = False  # type: ignore

    await chat_client._close_client_if_needed()  # type: ignore

    mock_ai_project_client.close.assert_not_called()


def test_azure_ai_chat_client_update_agent_name_when_current_is_none(mock_ai_project_client: MagicMock) -> None:
    """現在の agent_name が None のとき _update_agent_name が名前を更新することをテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)
    chat_client.agent_name = None  # type: ignore

    chat_client._update_agent_name("NewAgentName")  # type: ignore

    assert chat_client.agent_name == "NewAgentName"


def test_azure_ai_chat_client_update_agent_name_when_current_exists(mock_ai_project_client: MagicMock) -> None:
    """現在の agent_name が存在する場合、_update_agent_name が更新しないことをテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)
    chat_client.agent_name = "ExistingName"  # type: ignore

    chat_client._update_agent_name("NewAgentName")  # type: ignore

    assert chat_client.agent_name == "ExistingName"


def test_azure_ai_chat_client_update_agent_name_with_none_input(mock_ai_project_client: MagicMock) -> None:
    """None 入力で _update_agent_name をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)
    chat_client.agent_name = None  # type: ignore

    chat_client._update_agent_name(None)  # type: ignore

    assert chat_client.agent_name is None


async def test_azure_ai_chat_client_create_run_options_with_messages(mock_ai_project_client: MagicMock) -> None:
    """異なるメッセージタイプで _create_run_options をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    # system メッセージでテスト（指示に変換される）
    messages = [
        ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant"),
        ChatMessage(role=Role.USER, text="Hello"),
    ]

    run_options, _ = await chat_client._create_run_options(messages, None)  # type: ignore

    assert "instructions" in run_options
    assert "You are a helpful assistant" in run_options["instructions"]
    assert "additional_messages" in run_options
    assert len(run_options["additional_messages"]) == 1  # ユーザーメッセージのみ


async def test_azure_ai_chat_client_instructions_sent_once(mock_ai_project_client: MagicMock) -> None:
    """AzureAIAgentClient で指示が一度だけ送信されることを保証します。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    instructions = "You are a helpful assistant."
    chat_options = ChatOptions(instructions=instructions)
    messages = chat_client.prepare_messages([ChatMessage(role=Role.USER, text="Hello")], chat_options)

    run_options, _ = await chat_client._create_run_options(messages, chat_options)  # type: ignore

    assert run_options.get("instructions") == instructions


async def test_azure_ai_chat_client_inner_get_response(mock_ai_project_client: MagicMock) -> None:
    """_inner_get_response メソッドをテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")
    messages = [ChatMessage(role=Role.USER, text="Hello")]
    chat_options = ChatOptions()

    async def mock_streaming_response():
        yield ChatResponseUpdate(role=Role.ASSISTANT, text="Hello back")

    with (
        patch.object(chat_client, "_inner_get_streaming_response", return_value=mock_streaming_response()),
        patch("agent_framework.ChatResponse.from_chat_response_generator") as mock_from_generator,
    ):
        mock_response = ChatResponse(role=Role.ASSISTANT, text="Hello back")
        mock_from_generator.return_value = mock_response

        result = await chat_client._inner_get_response(messages=messages, chat_options=chat_options)  # type: ignore

        assert result is mock_response
        mock_from_generator.assert_called_once()


async def test_azure_ai_chat_client_get_agent_id_or_create_with_run_options(
    mock_ai_project_client: MagicMock, azure_ai_unit_test_env: dict[str, str]
) -> None:
    """ツールと指示を含む run_options で _get_agent_id_or_create をテストします。"""
    azure_ai_settings = AzureAISettings(model_deployment_name=azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, azure_ai_settings=azure_ai_settings)

    run_options = {
        "tools": [{"type": "function", "function": {"name": "test_tool"}}],
        "instructions": "Test instructions",
        "response_format": {"type": "json_object"},
        "model": azure_ai_settings.model_deployment_name,
    }

    agent_id = await chat_client._get_agent_id_or_create(run_options)  # type: ignore

    assert agent_id == "test-agent-id"
    # run_options パラメータで create_agent が呼び出されたことを検証します。
    mock_ai_project_client.agents.create_agent.assert_called_once()
    call_args = mock_ai_project_client.agents.create_agent.call_args[1]
    assert "tools" in call_args
    assert "instructions" in call_args
    assert "response_format" in call_args


async def test_azure_ai_chat_client_prepare_thread_cancels_active_run(mock_ai_project_client: MagicMock) -> None:
    """提供された場合、アクティブなスレッド実行をキャンセルする _prepare_thread をテストします。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    mock_thread_run = MagicMock()
    mock_thread_run.id = "run_123"
    mock_thread_run.thread_id = "test-thread"

    run_options = {"additional_messages": []}  # type: ignore

    result = await chat_client._prepare_thread("test-thread", mock_thread_run, run_options)  # type: ignore

    assert result == "test-thread"
    mock_ai_project_client.agents.runs.cancel.assert_called_once_with("test-thread", "run_123")


def test_azure_ai_chat_client_create_function_call_contents_basic(mock_ai_project_client: MagicMock) -> None:
    """基本的な関数呼び出しで _create_function_call_contents をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    mock_tool_call = MagicMock(spec=RequiredFunctionToolCall)
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Seattle"}'

    mock_submit_action = MagicMock(spec=SubmitToolOutputsAction)
    mock_submit_action.submit_tool_outputs.tool_calls = [mock_tool_call]

    mock_event_data = MagicMock(spec=ThreadRun)
    mock_event_data.required_action = mock_submit_action

    result = chat_client._create_function_call_contents(mock_event_data, "response_123")  # type: ignore

    assert len(result) == 1
    assert isinstance(result[0], FunctionCallContent)
    assert result[0].name == "get_weather"
    assert result[0].call_id == '["response_123", "call_123"]'


def test_azure_ai_chat_client_create_function_call_contents_no_submit_action(mock_ai_project_client: MagicMock) -> None:
    """required_action が SubmitToolOutputsAction でない場合の _create_function_call_contents をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    mock_event_data = MagicMock(spec=ThreadRun)
    mock_event_data.required_action = MagicMock()

    result = chat_client._create_function_call_contents(mock_event_data, "response_123")  # type: ignore

    assert result == []


def test_azure_ai_chat_client_create_function_call_contents_non_function_tool_call(
    mock_ai_project_client: MagicMock,
) -> None:
    """関数ではないツール呼び出しで _create_function_call_contents をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    mock_tool_call = MagicMock()

    mock_submit_action = MagicMock(spec=SubmitToolOutputsAction)
    mock_submit_action.submit_tool_outputs.tool_calls = [mock_tool_call]

    mock_event_data = MagicMock(spec=ThreadRun)
    mock_event_data.required_action = mock_submit_action

    result = chat_client._create_function_call_contents(mock_event_data, "response_123")  # type: ignore

    assert result == []


async def test_azure_ai_chat_client_create_run_options_with_none_tool_choice(
    mock_ai_project_client: MagicMock,
) -> None:
    """tool_choice が 'none' に設定された場合の _create_run_options をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    chat_options = ChatOptions()
    chat_options.tool_choice = "none"

    run_options, _ = await chat_client._create_run_options([], chat_options)

    from azure.ai.agents.models import AgentsToolChoiceOptionMode

    assert run_options["tool_choice"] == AgentsToolChoiceOptionMode.NONE


async def test_azure_ai_chat_client_create_run_options_with_auto_tool_choice(
    mock_ai_project_client: MagicMock,
) -> None:
    """tool_choice が 'auto' に設定された場合の _create_run_options をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    chat_options = ChatOptions()
    chat_options.tool_choice = "auto"

    run_options, _ = await chat_client._create_run_options([], chat_options)

    from azure.ai.agents.models import AgentsToolChoiceOptionMode

    assert run_options["tool_choice"] == AgentsToolChoiceOptionMode.AUTO


async def test_azure_ai_chat_client_create_run_options_with_response_format(
    mock_ai_project_client: MagicMock,
) -> None:
    """response_format が設定された場合の _create_run_options をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    class TestResponseModel(BaseModel):
        name: str = Field(description="Test name")

    chat_options = ChatOptions()
    chat_options.response_format = TestResponseModel

    run_options, _ = await chat_client._create_run_options([], chat_options)

    assert "response_format" in run_options
    response_format = run_options["response_format"]
    assert response_format.json_schema.name == "TestResponseModel"


def test_azure_ai_chat_client_service_url_method(mock_ai_project_client: MagicMock) -> None:
    """service_url メソッドがエンドポイントを返すことをテストする。"""
    mock_ai_project_client._config.endpoint = "https://test-endpoint.com/"
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    url = chat_client.service_url()
    assert url == "https://test-endpoint.com/"


async def test_azure_ai_chat_client_prep_tools_ai_function(mock_ai_project_client: MagicMock) -> None:
    """AIFunction ツールで _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # モックの AIFunction を作成する。
    mock_ai_function = MagicMock(spec=AIFunction)
    mock_ai_function.to_json_schema_spec.return_value = {"type": "function", "function": {"name": "test_function"}}

    result = await chat_client._prep_tools([mock_ai_function])  # type: ignore

    assert len(result) == 1
    assert result[0] == {"type": "function", "function": {"name": "test_function"}}
    mock_ai_function.to_json_schema_spec.assert_called_once()


async def test_azure_ai_chat_client_prep_tools_code_interpreter(mock_ai_project_client: MagicMock) -> None:
    """HostedCodeInterpreterTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    code_interpreter_tool = HostedCodeInterpreterTool()

    result = await chat_client._prep_tools([code_interpreter_tool])  # type: ignore

    assert len(result) == 1
    assert isinstance(result[0], CodeInterpreterToolDefinition)


async def test_azure_ai_chat_client_prep_tools_mcp_tool(mock_ai_project_client: MagicMock) -> None:
    """HostedMCPTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    mcp_tool = HostedMCPTool(name="Test MCP Tool", url="https://example.com/mcp", allowed_tools=["tool1", "tool2"])

    # definitions 属性を持つように McpTool をモックする。
    with patch("agent_framework_azure_ai._chat_client.McpTool") as mock_mcp_tool_class:
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.definitions = [{"type": "mcp", "name": "test_mcp"}]
        mock_mcp_tool_class.return_value = mock_mcp_tool

        result = await chat_client._prep_tools([mcp_tool])  # type: ignore

        assert len(result) == 1
        assert result[0] == {"type": "mcp", "name": "test_mcp"}
        # 呼び出しが行われたことを確認する（allowed_tools の順序は異なる場合がある）。
        mock_mcp_tool_class.assert_called_once()
        call_args = mock_mcp_tool_class.call_args[1]
        assert call_args["server_label"] == "Test_MCP_Tool"
        assert call_args["server_url"] == "https://example.com/mcp"
        assert set(call_args["allowed_tools"]) == {"tool1", "tool2"}


async def test_azure_ai_chat_client_create_run_options_mcp_never_require(mock_ai_project_client: MagicMock) -> None:
    """approval モードが never_require の HostedMCPTool で _create_run_options をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    mcp_tool = HostedMCPTool(name="Test MCP Tool", url="https://example.com/mcp", approval_mode="never_require")

    messages = [ChatMessage(role=Role.USER, text="Hello")]
    chat_options = ChatOptions(tools=[mcp_tool], tool_choice="auto")

    with patch("agent_framework_azure_ai._chat_client.McpTool") as mock_mcp_tool_class:
        # 実際のツール準備を避けるために _prep_tools をモックする。
        mock_mcp_tool_instance = MagicMock()
        mock_mcp_tool_instance.definitions = [{"type": "mcp", "name": "test_mcp"}]
        mock_mcp_tool_class.return_value = mock_mcp_tool_instance

        run_options, _ = await chat_client._create_run_options(messages, chat_options)  # type: ignore

        # 正しい MCP 承認構造で tool_resources が作成されることを検証する。
        assert "tool_resources" in run_options, (
            f"Expected 'tool_resources' in run_options keys: {list(run_options.keys())}"
        )
        assert "mcp" in run_options["tool_resources"]
        assert len(run_options["tool_resources"]["mcp"]) == 1

        mcp_resource = run_options["tool_resources"]["mcp"][0]
        assert mcp_resource["server_label"] == "Test_MCP_Tool"
        assert mcp_resource["require_approval"] == "never"


async def test_azure_ai_chat_client_create_run_options_mcp_with_headers(mock_ai_project_client: MagicMock) -> None:
    """ヘッダーを持つ HostedMCPTool で _create_run_options をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client)

    # ヘッダー付きでテストする。
    headers = {"Authorization": "Bearer DUMMY_TOKEN", "X-API-Key": "DUMMY_KEY"}
    mcp_tool = HostedMCPTool(
        name="Test MCP Tool", url="https://example.com/mcp", headers=headers, approval_mode="never_require"
    )

    messages = [ChatMessage(role=Role.USER, text="Hello")]
    chat_options = ChatOptions(tools=[mcp_tool], tool_choice="auto")

    with patch("agent_framework_azure_ai._chat_client.McpTool") as mock_mcp_tool_class:
        # 実際のツール準備を避けるために _prep_tools をモックする。
        mock_mcp_tool_instance = MagicMock()
        mock_mcp_tool_instance.definitions = [{"type": "mcp", "name": "test_mcp"}]
        mock_mcp_tool_class.return_value = mock_mcp_tool_instance

        run_options, _ = await chat_client._create_run_options(messages, chat_options)  # type: ignore

        # ヘッダー付きで tool_resources が作成されることを検証する。
        assert "tool_resources" in run_options
        assert "mcp" in run_options["tool_resources"]
        assert len(run_options["tool_resources"]["mcp"]) == 1

        mcp_resource = run_options["tool_resources"]["mcp"][0]
        assert mcp_resource["server_label"] == "Test_MCP_Tool"
        assert mcp_resource["require_approval"] == "never"
        assert mcp_resource["headers"] == headers


async def test_azure_ai_chat_client_prep_tools_web_search_bing_grounding(mock_ai_project_client: MagicMock) -> None:
    """Bing Grounding を使用する HostedWebSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    web_search_tool = HostedWebSearchTool(
        additional_properties={
            "connection_name": "test-connection-name",
            "count": 5,
            "freshness": "Day",
            "market": "en-US",
            "set_lang": "en",
        }
    )

    # connection get をモックする。
    mock_connection = MagicMock()
    mock_connection.id = "test-connection-id"
    mock_ai_project_client.connections.get = AsyncMock(return_value=mock_connection)

    # BingGroundingTool をモックする。
    with patch("agent_framework_azure_ai._chat_client.BingGroundingTool") as mock_bing_grounding:
        mock_bing_tool = MagicMock()
        mock_bing_tool.definitions = [{"type": "bing_grounding"}]
        mock_bing_grounding.return_value = mock_bing_tool

        result = await chat_client._prep_tools([web_search_tool])  # type: ignore

        assert len(result) == 1
        assert result[0] == {"type": "bing_grounding"}
        mock_bing_grounding.assert_called_once_with(
            connection_id="test-connection-id", count=5, freshness="Day", market="en-US", set_lang="en"
        )


async def test_azure_ai_chat_client_prep_tools_web_search_bing_grounding_with_connection_id(
    mock_ai_project_client: MagicMock,
) -> None:
    """connection_id を使った Bing Grounding を使用する HostedWebSearchTool で _prep_tools をテストする（HTTP 呼び出しなし）。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    web_search_tool = HostedWebSearchTool(
        additional_properties={
            "connection_id": "direct-connection-id",
            "count": 3,
        }
    )

    # BingGroundingTool をモックする。
    with patch("agent_framework_azure_ai._chat_client.BingGroundingTool") as mock_bing_grounding:
        mock_bing_tool = MagicMock()
        mock_bing_tool.definitions = [{"type": "bing_grounding"}]
        mock_bing_grounding.return_value = mock_bing_tool

        result = await chat_client._prep_tools([web_search_tool])  # type: ignore

        assert len(result) == 1
        assert result[0] == {"type": "bing_grounding"}
        # connection_id が直接使用されたことを検証する（connections.get への HTTP 呼び出しなし）。
        mock_ai_project_client.connections.get.assert_not_called()
        mock_bing_grounding.assert_called_once_with(connection_id="direct-connection-id", count=3)


async def test_azure_ai_chat_client_prep_tools_web_search_custom_bing(mock_ai_project_client: MagicMock) -> None:
    """Custom Bing Search を使用する HostedWebSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    web_search_tool = HostedWebSearchTool(
        additional_properties={
            "custom_connection_name": "custom-bing-connection",
            "custom_instance_name": "custom-instance",
            "count": 10,
        }
    )

    # connection get をモックする。
    mock_connection = MagicMock()
    mock_connection.id = "custom-connection-id"
    mock_ai_project_client.connections.get = AsyncMock(return_value=mock_connection)

    # BingCustomSearchTool をモックする。
    with patch("agent_framework_azure_ai._chat_client.BingCustomSearchTool") as mock_custom_bing:
        mock_custom_tool = MagicMock()
        mock_custom_tool.definitions = [{"type": "bing_custom_search"}]
        mock_custom_bing.return_value = mock_custom_tool

        result = await chat_client._prep_tools([web_search_tool])  # type: ignore

        assert len(result) == 1
        assert result[0] == {"type": "bing_custom_search"}
        mock_ai_project_client.connections.get.assert_called_once_with(name="custom-bing-connection")
        mock_custom_bing.assert_called_once_with(
            connection_id="custom-connection-id", instance_name="custom-instance", count=10
        )


async def test_azure_ai_chat_client_prep_tools_web_search_custom_bing_connection_error(
    mock_ai_project_client: MagicMock,
) -> None:
    """カスタム接続が見つからない場合の HostedWebSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    web_search_tool = HostedWebSearchTool(
        additional_properties={
            "custom_connection_name": "nonexistent-connection",
            "custom_instance_name": "custom-instance",
        }
    )

    # HttpResponseError を発生させるように connection get をモックする。
    mock_ai_project_client.connections.get = AsyncMock(side_effect=HttpResponseError("Connection not found"))

    with pytest.raises(ServiceInitializationError, match="Bing custom connection 'nonexistent-connection' not found"):
        await chat_client._prep_tools([web_search_tool])  # type: ignore


async def test_azure_ai_chat_client_prep_tools_web_search_bing_grounding_connection_error(
    mock_ai_project_client: MagicMock,
) -> None:
    """Bing Grounding 接続が見つからない場合の HostedWebSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    web_search_tool = HostedWebSearchTool(
        additional_properties={
            "connection_name": "nonexistent-bing-connection",
        }
    )

    # HttpResponseError を発生させるように connection get をモックする。
    mock_ai_project_client.connections.get = AsyncMock(side_effect=HttpResponseError("Connection not found"))

    with pytest.raises(ServiceInitializationError, match="Bing connection 'nonexistent-bing-connection' not found"):
        await chat_client._prep_tools([web_search_tool])  # type: ignore


async def test_azure_ai_chat_client_prep_tools_file_search_with_vector_stores(
    mock_ai_project_client: MagicMock,
) -> None:
    """ベクターストアを使用する HostedFileSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    vector_store_input = HostedVectorStoreContent(vector_store_id="vs-123")
    file_search_tool = HostedFileSearchTool(inputs=[vector_store_input])

    # FileSearchTool をモックする。
    with patch("agent_framework_azure_ai._chat_client.FileSearchTool") as mock_file_search:
        mock_file_tool = MagicMock()
        mock_file_tool.definitions = [{"type": "file_search"}]
        mock_file_tool.resources = {"vector_store_ids": ["vs-123"]}
        mock_file_search.return_value = mock_file_tool

        run_options = {}
        result = await chat_client._prep_tools([file_search_tool], run_options)  # type: ignore

        assert len(result) == 1
        assert result[0] == {"type": "file_search"}
        assert run_options["tool_resources"] == {"vector_store_ids": ["vs-123"]}
        mock_file_search.assert_called_once_with(vector_store_ids=["vs-123"])


async def test_azure_ai_chat_client_prep_tools_file_search_with_ai_search(mock_ai_project_client: MagicMock) -> None:
    """Azure AI Search を使用する HostedFileSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    file_search_tool = HostedFileSearchTool(
        additional_properties={
            "index_name": "test-index",
            "query_type": "simple",
            "top_k": 5,
            "filter": "category eq 'docs'",
        }
    )

    # connections.get_default をモックする。
    mock_connection = MagicMock()
    mock_connection.id = "search-connection-id"
    mock_ai_project_client.connections.get_default = AsyncMock(return_value=mock_connection)

    # AzureAISearchTool をモックする。
    with patch("agent_framework_azure_ai._chat_client.AzureAISearchTool") as mock_ai_search:
        mock_search_tool = MagicMock()
        mock_search_tool.definitions = [{"type": "azure_ai_search"}]
        mock_ai_search.return_value = mock_search_tool

        # AzureAISearchQueryType をモックする。
        with patch("agent_framework_azure_ai._chat_client.AzureAISearchQueryType") as mock_query_type:
            mock_query_type.SIMPLE = "simple"
            mock_query_type.return_value = "simple"

            result = await chat_client._prep_tools([file_search_tool])  # type: ignore

            assert len(result) == 1
            assert result[0] == {"type": "azure_ai_search"}
            mock_ai_project_client.connections.get_default.assert_called_once_with(ConnectionType.AZURE_AI_SEARCH)
            mock_ai_search.assert_called_once_with(
                index_connection_id="search-connection-id",
                index_name="test-index",
                query_type="simple",
                top_k=5,
                filter="category eq 'docs'",
            )


async def test_azure_ai_chat_client_prep_tools_file_search_invalid_query_type(
    mock_ai_project_client: MagicMock,
) -> None:
    """無効な query_type を使用する HostedFileSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    file_search_tool = HostedFileSearchTool(
        additional_properties={"index_name": "test-index", "query_type": "invalid_type"}
    )

    # connections.get_default をモックする。
    mock_connection = MagicMock()
    mock_connection.id = "search-connection-id"
    mock_ai_project_client.connections.get_default = AsyncMock(return_value=mock_connection)

    # ValueError を発生させるように AzureAISearchQueryType をモックする。
    with patch("agent_framework_azure_ai._chat_client.AzureAISearchQueryType") as mock_query_type:
        mock_query_type.side_effect = ValueError("Invalid query type")

        with pytest.raises(ServiceInitializationError, match="Invalid query_type 'invalid_type'"):
            await chat_client._prep_tools([file_search_tool])  # type: ignore


async def test_azure_ai_chat_client_prep_tools_file_search_no_connection(mock_ai_project_client: MagicMock) -> None:
    """AI Search 接続が存在しない場合の HostedFileSearchTool で _prep_tools をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    file_search_tool = HostedFileSearchTool(additional_properties={"index_name": "test-index"})

    # ValueError を発生させるように connections.get_default をモックする。
    mock_ai_project_client.connections.get_default = AsyncMock(side_effect=ValueError("No connection found"))

    with pytest.raises(ServiceInitializationError, match="No default Azure AI Search connection found"):
        await chat_client._prep_tools([file_search_tool])  # type: ignore


async def test_azure_ai_chat_client_prep_tools_file_search_no_index_name(
    mock_ai_project_client: MagicMock, monkeypatch: MonkeyPatch
) -> None:
    """index_name とベクターストアが欠落している HostedFileSearchTool で _prep_tools をテストする。"""
    monkeypatch.delenv("AZURE_AI_SEARCH_INDEX_NAME", raising=False)

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # ベクターストアも index_name もないファイル検索ツール。
    file_search_tool = HostedFileSearchTool()

    with pytest.raises(ServiceInitializationError, match="File search tool requires at least one vector store input"):
        await chat_client._prep_tools([file_search_tool])  # type: ignore


async def test_azure_ai_chat_client_prep_tools_dict_tool(mock_ai_project_client: MagicMock) -> None:
    """辞書によるツール定義で _prep_tools をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    dict_tool = {"type": "custom_tool", "config": {"param": "value"}}

    result = await chat_client._prep_tools([dict_tool])  # type: ignore

    assert len(result) == 1
    assert result[0] == dict_tool


async def test_azure_ai_chat_client_prep_tools_unsupported_tool(mock_ai_project_client: MagicMock) -> None:
    """サポートされていないツールタイプで _prep_tools をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    unsupported_tool = "not_a_tool"

    with pytest.raises(ServiceInitializationError, match="Unsupported tool type: <class 'str'>"):
        await chat_client._prep_tools([unsupported_tool])  # type: ignore


async def test_azure_ai_chat_client_get_active_thread_run_with_active_run(mock_ai_project_client: MagicMock) -> None:
    """アクティブな実行がある場合の _get_active_thread_run をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # アクティブな実行をモックする。
    mock_run = MagicMock()
    mock_run.status = RunStatus.IN_PROGRESS

    async def mock_list_runs(*args, **kwargs):
        yield mock_run

    mock_ai_project_client.agents.runs.list = mock_list_runs

    result = await chat_client._get_active_thread_run("thread-123")  # type: ignore

    assert result == mock_run


async def test_azure_ai_chat_client_get_active_thread_run_no_active_run(mock_ai_project_client: MagicMock) -> None:
    """アクティブな実行がない場合の _get_active_thread_run をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 完了した実行（アクティブでない）をモックする。
    mock_run = MagicMock()
    mock_run.status = RunStatus.COMPLETED

    async def mock_list_runs(*args, **kwargs):
        yield mock_run

    mock_ai_project_client.agents.runs.list = mock_list_runs

    result = await chat_client._get_active_thread_run("thread-123")  # type: ignore

    assert result is None


async def test_azure_ai_chat_client_get_active_thread_run_no_thread(mock_ai_project_client: MagicMock) -> None:
    """thread_id が None の場合の _get_active_thread_run をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    result = await chat_client._get_active_thread_run(None)  # type: ignore

    assert result is None
    # thread_id が None のため list を呼び出すべきでない。
    mock_ai_project_client.agents.runs.list.assert_not_called()


async def test_azure_ai_chat_client_service_url(mock_ai_project_client: MagicMock) -> None:
    """service_url メソッドをテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # config エンドポイントをモックする。
    mock_config = MagicMock()
    mock_config.endpoint = "https://test-endpoint.com/"
    mock_ai_project_client._config = mock_config

    result = chat_client.service_url()

    assert result == "https://test-endpoint.com/"


async def test_azure_ai_chat_client_convert_required_action_to_tool_output_function_result(
    mock_ai_project_client: MagicMock,
) -> None:
    """FunctionResultContent を使った _convert_required_action_to_tool_output をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 単純な結果でテストする。
    function_result = FunctionResultContent(call_id='["run_123", "call_456"]', result="Simple result")

    run_id, tool_outputs, tool_approvals = chat_client._convert_required_action_to_tool_output([function_result])  # type: ignore

    assert run_id == "run_123"
    assert tool_approvals is None
    assert tool_outputs is not None
    assert len(tool_outputs) == 1
    assert tool_outputs[0].tool_call_id == "call_456"
    assert tool_outputs[0].output == "Simple result"


async def test_azure_ai_chat_client_convert_required_action_invalid_call_id(mock_ai_project_client: MagicMock) -> None:
    """無効な call_id フォーマットで _convert_required_action_to_tool_output をテストする。"""

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 無効な call_id フォーマット - JSONDecodeError を発生させるべき。
    function_result = FunctionResultContent(call_id="invalid_json", result="result")

    with pytest.raises(json.JSONDecodeError):
        chat_client._convert_required_action_to_tool_output([function_result])  # type: ignore


async def test_azure_ai_chat_client_convert_required_action_invalid_structure(
    mock_ai_project_client: MagicMock,
) -> None:
    """無効な call_id 構造で _convert_required_action_to_tool_output をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 有効な JSON だが構造が無効（第2要素が欠落）。
    function_result = FunctionResultContent(call_id='["run_123"]', result="result")

    run_id, tool_outputs, tool_approvals = chat_client._convert_required_action_to_tool_output([function_result])  # type: ignore

    # 構造が無効な場合は None 値を返すべき。
    assert run_id is None
    assert tool_outputs is None
    assert tool_approvals is None


async def test_azure_ai_chat_client_convert_required_action_serde_model_results(
    mock_ai_project_client: MagicMock,
) -> None:
    """BaseModel 結果で _convert_required_action_to_tool_output をテストする。"""

    class MockResult(SerializationMixin):
        def __init__(self, name: str, value: int):
            self.name = name
            self.value = value

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # BaseModel 結果でテストする。
    mock_result = MockResult(name="test", value=42)
    function_result = FunctionResultContent(call_id='["run_123", "call_456"]', result=mock_result)

    run_id, tool_outputs, tool_approvals = chat_client._convert_required_action_to_tool_output([function_result])  # type: ignore

    assert run_id == "run_123"
    assert tool_approvals is None
    assert tool_outputs is not None
    assert len(tool_outputs) == 1
    assert tool_outputs[0].tool_call_id == "call_456"
    # BaseModel には model_dump_json を使用すべき。
    expected_json = mock_result.to_json()
    assert tool_outputs[0].output == expected_json


async def test_azure_ai_chat_client_convert_required_action_multiple_results(
    mock_ai_project_client: MagicMock,
) -> None:
    """複数の結果で _convert_required_action_to_tool_output をテストする。"""

    class MockResult(SerializationMixin):
        def __init__(self, data: str):
            self.data = data

    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 複数の結果でテスト - BaseModel と通常のオブジェクトの混合。
    mock_basemodel = MockResult(data="model_data")
    results_list = [mock_basemodel, {"key": "value"}, "string_result"]
    function_result = FunctionResultContent(call_id='["run_123", "call_456"]', result=results_list)

    run_id, tool_outputs, tool_approvals = chat_client._convert_required_action_to_tool_output([function_result])  # type: ignore

    assert run_id == "run_123"
    assert tool_outputs is not None
    assert len(tool_outputs) == 1
    assert tool_outputs[0].tool_call_id == "call_456"

    # len > 1 のため結果配列全体を JSON dump すべき。
    expected_results = [
        mock_basemodel.to_dict(),
        {"key": "value"},
        "string_result",
    ]
    expected_output = json.dumps(expected_results)
    assert tool_outputs[0].output == expected_output


async def test_azure_ai_chat_client_convert_required_action_approval_response(
    mock_ai_project_client: MagicMock,
) -> None:
    """FunctionApprovalResponseContent で _convert_required_action_to_tool_output をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 承認レスポンスでテスト - 必須フィールドを提供する必要がある。
    approval_response = FunctionApprovalResponseContent(
        id='["run_123", "call_456"]',
        function_call=FunctionCallContent(call_id='["run_123", "call_456"]', name="test_function", arguments="{}"),
        approved=True,
    )

    run_id, tool_outputs, tool_approvals = chat_client._convert_required_action_to_tool_output([approval_response])  # type: ignore

    assert run_id == "run_123"
    assert tool_outputs is None
    assert tool_approvals is not None
    assert len(tool_approvals) == 1
    assert tool_approvals[0].tool_call_id == "call_456"
    assert tool_approvals[0].approve is True


async def test_azure_ai_chat_client_create_function_call_contents_approval_request(
    mock_ai_project_client: MagicMock,
) -> None:
    """承認アクションで _create_function_call_contents をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # RequiredMcpToolCall を持つ SubmitToolApprovalAction をモックする。
    mock_tool_call = MagicMock(spec=RequiredMcpToolCall)
    mock_tool_call.id = "approval_call_123"
    mock_tool_call.name = "approve_action"
    mock_tool_call.arguments = '{"action": "approve"}'

    mock_approval_action = MagicMock(spec=SubmitToolApprovalAction)
    mock_approval_action.submit_tool_approval.tool_calls = [mock_tool_call]

    mock_event_data = MagicMock(spec=ThreadRun)
    mock_event_data.required_action = mock_approval_action

    result = chat_client._create_function_call_contents(mock_event_data, "response_123")  # type: ignore

    assert len(result) == 1
    assert isinstance(result[0], FunctionApprovalRequestContent)
    assert result[0].id == '["response_123", "approval_call_123"]'
    assert result[0].function_call.name == "approve_action"
    assert result[0].function_call.call_id == '["response_123", "approval_call_123"]'


async def test_azure_ai_chat_client_get_agent_id_or_create_with_agent_name(
    mock_ai_project_client: MagicMock, azure_ai_unit_test_env: dict[str, str]
) -> None:
    """agent_name が設定されていない場合、_get_agent_id_or_create がデフォルト名を使用することをテストする。"""
    azure_ai_settings = AzureAISettings(model_deployment_name=azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, azure_ai_settings=azure_ai_settings)

    # デフォルトをテストするため agent_name を None にする。
    chat_client.agent_name = None  # type: ignore

    agent_id = await chat_client._get_agent_id_or_create(run_options={"model": azure_ai_settings.model_deployment_name})  # type: ignore

    assert agent_id == "test-agent-id"
    # create_agent がデフォルトの "UnnamedAgent" で呼ばれたことを検証する。
    mock_ai_project_client.agents.create_agent.assert_called_once()
    call_kwargs = mock_ai_project_client.agents.create_agent.call_args[1]
    assert call_kwargs["name"] == "UnnamedAgent"


async def test_azure_ai_chat_client_get_agent_id_or_create_with_response_format(
    mock_ai_project_client: MagicMock, azure_ai_unit_test_env: dict[str, str]
) -> None:
    """run_options に response_format がある場合の _get_agent_id_or_create をテストする。"""
    azure_ai_settings = AzureAISettings(model_deployment_name=azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, azure_ai_settings=azure_ai_settings)

    # run_options に response_format がある場合でテストする。
    run_options = {"response_format": {"type": "json_object"}, "model": azure_ai_settings.model_deployment_name}

    agent_id = await chat_client._get_agent_id_or_create(run_options)  # type: ignore

    assert agent_id == "test-agent-id"
    # create_agent が response_format で呼ばれたことを検証する。
    mock_ai_project_client.agents.create_agent.assert_called_once()
    call_kwargs = mock_ai_project_client.agents.create_agent.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}


async def test_azure_ai_chat_client_get_agent_id_or_create_with_tool_resources(
    mock_ai_project_client: MagicMock, azure_ai_unit_test_env: dict[str, str]
) -> None:
    """run_options に tool_resources がある場合の _get_agent_id_or_create をテストする。"""
    azure_ai_settings = AzureAISettings(model_deployment_name=azure_ai_unit_test_env["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, azure_ai_settings=azure_ai_settings)

    # run_options に tool_resources がある場合でテストする。
    run_options = {
        "tool_resources": {"vector_store_ids": ["vs-123"]},
        "model": azure_ai_settings.model_deployment_name,
    }

    agent_id = await chat_client._get_agent_id_or_create(run_options)  # type: ignore

    assert agent_id == "test-agent-id"
    # create_agent が tool_resources で呼ばれたことを検証する。
    mock_ai_project_client.agents.create_agent.assert_called_once()
    call_kwargs = mock_ai_project_client.agents.create_agent.call_args[1]
    assert call_kwargs["tool_resources"] == {"vector_store_ids": ["vs-123"]}


async def test_azure_ai_chat_client_close_method(mock_ai_project_client: MagicMock) -> None:
    """close メソッドをテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, should_delete_agent=True)
    chat_client._should_close_client = True
    chat_client.agent_id = "test-agent"

    # クリーンアップメソッドをモックする。
    mock_ai_project_client.agents.delete_agent = AsyncMock()
    mock_ai_project_client.close = AsyncMock()

    await chat_client.close()

    # クリーンアップが呼ばれたことを検証する。
    mock_ai_project_client.agents.delete_agent.assert_called_once_with("test-agent")
    mock_ai_project_client.close.assert_called_once()


async def test_azure_ai_chat_client_create_agent_stream_submit_tool_outputs(
    mock_ai_project_client: MagicMock,
) -> None:
    """ツール出力の送信パスで _create_agent_stream をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # ツール実行 ID と一致するアクティブなスレッド実行をモックする。
    mock_thread_run = MagicMock()
    mock_thread_run.thread_id = "test-thread"
    mock_thread_run.id = "test-run-id"
    chat_client._get_active_thread_run = AsyncMock(return_value=mock_thread_run)

    # 一致する実行 ID を持つ required action 結果をモックする。
    function_result = FunctionResultContent(call_id='["test-run-id", "test-call-id"]', result="test result")

    # submit_tool_outputs_stream をモックする。
    mock_handler = MagicMock()
    mock_ai_project_client.agents.runs.submit_tool_outputs_stream = AsyncMock()

    with patch("azure.ai.agents.models.AsyncAgentEventHandler", return_value=mock_handler):
        stream, final_thread_id = await chat_client._create_agent_stream(
            thread_id="test-thread", agent_id="test-agent", run_options={}, required_action_results=[function_result]
        )

        # 一致する実行 ID があるため submit_tool_outputs_stream を呼び出すべき。
        mock_ai_project_client.agents.runs.submit_tool_outputs_stream.assert_called_once()
        assert final_thread_id == "test-thread"


def test_azure_ai_chat_client_extract_url_citations_with_citations(mock_ai_project_client: MagicMock) -> None:
    """URL 引用を含む MessageDeltaChunk で _extract_url_citations をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # モックの URL 引用注釈を作成する。
    mock_url_citation = MagicMock()
    mock_url_citation.url = "https://example.com/test"
    mock_url_citation.title = "Test Title"

    mock_annotation = MagicMock(spec=MessageDeltaTextUrlCitationAnnotation)
    mock_annotation.url_citation = mock_url_citation
    mock_annotation.start_index = 10
    mock_annotation.end_index = 20

    # 注釈付きのモックテキストコンテンツを作成する。
    mock_text = MagicMock()
    mock_text.annotations = [mock_annotation]

    mock_text_content = MagicMock(spec=MessageDeltaTextContent)
    mock_text_content.text = mock_text

    # モックの delta を作成する。
    mock_delta = MagicMock()
    mock_delta.content = [mock_text_content]

    # モックの MessageDeltaChunk を作成する。
    mock_chunk = MagicMock(spec=MessageDeltaChunk)
    mock_chunk.delta = mock_delta

    # メソッドを呼び出す。
    citations = chat_client._extract_url_citations(mock_chunk)  # type: ignore

    # 結果を検証する。
    assert len(citations) == 1
    citation = citations[0]
    assert isinstance(citation, CitationAnnotation)
    assert citation.url == "https://example.com/test"
    assert citation.title == "Test Title"
    assert citation.snippet is None
    assert citation.annotated_regions is not None
    assert len(citation.annotated_regions) == 1
    assert citation.annotated_regions[0].start_index == 10
    assert citation.annotated_regions[0].end_index == 20


def test_azure_ai_chat_client_extract_url_citations_no_citations(mock_ai_project_client: MagicMock) -> None:
    """引用がない MessageDeltaChunk で _extract_url_citations をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 注釈なしのモックテキストコンテンツを作成する。
    mock_text_content = MagicMock(spec=MessageDeltaTextContent)
    mock_text_content.text = None  # テキストがないため注釈もない。

    # モックの delta を作成する。
    mock_delta = MagicMock()
    mock_delta.content = [mock_text_content]

    # モックの MessageDeltaChunk を作成する。
    mock_chunk = MagicMock(spec=MessageDeltaChunk)
    mock_chunk.delta = mock_delta

    # メソッドを呼び出す。
    citations = chat_client._extract_url_citations(mock_chunk)  # type: ignore

    # 引用が返されないことを検証する。
    assert len(citations) == 0


def test_azure_ai_chat_client_extract_url_citations_empty_delta(mock_ai_project_client: MagicMock) -> None:
    """空の delta コンテンツで _extract_url_citations をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # 空のコンテンツを持つモック delta を作成する。
    mock_delta = MagicMock()
    mock_delta.content = []

    # モックの MessageDeltaChunk を作成する。
    mock_chunk = MagicMock(spec=MessageDeltaChunk)
    mock_chunk.delta = mock_delta

    # メソッドを呼び出す。
    citations = chat_client._extract_url_citations(mock_chunk)  # type: ignore

    # 引用が返されないことを検証する。
    assert len(citations) == 0


def test_azure_ai_chat_client_extract_url_citations_without_indices(mock_ai_project_client: MagicMock) -> None:
    """開始/終了インデックスを持たない URL 引用で _extract_url_citations をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # インデックスなしのモック URL 引用注釈を作成する。
    mock_url_citation = MagicMock()
    mock_url_citation.url = "https://example.com/no-indices"

    mock_annotation = MagicMock(spec=MessageDeltaTextUrlCitationAnnotation)
    mock_annotation.url_citation = mock_url_citation
    mock_annotation.start_index = None
    mock_annotation.end_index = None

    # 注釈付きのモックテキストコンテンツを作成する。
    mock_text = MagicMock()
    mock_text.annotations = [mock_annotation]

    mock_text_content = MagicMock(spec=MessageDeltaTextContent)
    mock_text_content.text = mock_text

    # モックの delta を作成する。
    mock_delta = MagicMock()
    mock_delta.content = [mock_text_content]

    # モックの MessageDeltaChunk を作成する。
    mock_chunk = MagicMock(spec=MessageDeltaChunk)
    mock_chunk.delta = mock_delta

    # メソッドを呼び出す。
    citations = chat_client._extract_url_citations(mock_chunk)  # type: ignore

    # 結果を検証する。
    assert len(citations) == 1
    citation = citations[0]
    assert citation.url == "https://example.com/no-indices"
    assert citation.annotated_regions is not None
    assert len(citation.annotated_regions) == 0  # インデックスが None の場合は領域なし。


async def test_azure_ai_chat_client_setup_azure_ai_observability_resource_not_found(
    mock_ai_project_client: MagicMock,
) -> None:
    """Application Insights 接続文字列が見つからない場合の setup_azure_ai_observability をテストする。"""
    chat_client = create_test_azure_ai_chat_client(mock_ai_project_client, agent_id="test-agent")

    # ResourceNotFoundError を発生させるように
    # telemetry.get_application_insights_connection_string をモックする。
    mock_ai_project_client.telemetry.get_application_insights_connection_string = AsyncMock(
        side_effect=ResourceNotFoundError("No Application Insights found")
    )

    # 警告メッセージをキャプチャするために logger.warning をモックする
    with patch("agent_framework_azure_ai._chat_client.logger") as mock_logger:
        await chat_client.setup_azure_ai_observability()

        # 警告がログに記録されたことを検証する
        mock_logger.warning.assert_called_once_with(
            "No Application Insights connection string found for the Azure AI Project, "
            "please call setup_observability() manually."
        )


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    return f"The weather in {location} is sunny with a high of 25°C."


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_get_response() -> None:
    """Azure AI Chat Client のレスポンスをテストする。"""
    async with AzureAIAgentClient(async_credential=AzureCliCredential()) as azure_ai_chat_client:
        assert isinstance(azure_ai_chat_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(
            ChatMessage(
                role="user",
                text="The weather in Seattle is currently sunny with a high of 25°C. "
                "It's a beautiful day for outdoor activities.",
            )
        )
        messages.append(ChatMessage(role="user", text="What's the weather like today?"))

        # project_client を使ってレスポンスを取得できることをテストする
        response = await azure_ai_chat_client.get_response(messages=messages)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert any(word in response.text.lower() for word in ["sunny", "25"])


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_get_response_tools() -> None:
    """ツールを使った Azure AI Chat Client のレスポンスをテストする。"""
    async with AzureAIAgentClient(async_credential=AzureCliCredential()) as azure_ai_chat_client:
        assert isinstance(azure_ai_chat_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(ChatMessage(role="user", text="What's the weather like in Seattle?"))

        # project_client を使ってレスポンスを取得できることをテストする
        response = await azure_ai_chat_client.get_response(
            messages=messages,
            tools=[get_weather],
            tool_choice="auto",
        )

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert any(word in response.text.lower() for word in ["sunny", "25"])


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_streaming() -> None:
    """Azure AI Chat Client のストリーミングレスポンスをテストする。"""
    async with AzureAIAgentClient(async_credential=AzureCliCredential()) as azure_ai_chat_client:
        assert isinstance(azure_ai_chat_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(
            ChatMessage(
                role="user",
                text="The weather in Seattle is currently sunny with a high of 25°C. "
                "It's a beautiful day for outdoor activities.",
            )
        )
        messages.append(ChatMessage(role="user", text="What's the weather like today?"))

        # project_client を使ってレスポンスを取得できることをテストする
        response = azure_ai_chat_client.get_streaming_response(messages=messages)

        full_message: str = ""
        async for chunk in response:
            assert chunk is not None
            assert isinstance(chunk, ChatResponseUpdate)
            for content in chunk.contents:
                if isinstance(content, TextContent) and content.text:
                    full_message += content.text

        assert any(word in full_message.lower() for word in ["sunny", "25"])


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_streaming_tools() -> None:
    """ツールを使った Azure AI Chat Client のストリーミングレスポンスをテストする。"""
    async with AzureAIAgentClient(async_credential=AzureCliCredential()) as azure_ai_chat_client:
        assert isinstance(azure_ai_chat_client, ChatClientProtocol)

        messages: list[ChatMessage] = []
        messages.append(ChatMessage(role="user", text="What's the weather like in Seattle?"))

        # project_client を使ってレスポンスを取得できることをテストする
        response = azure_ai_chat_client.get_streaming_response(
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

        assert any(word in full_message.lower() for word in ["sunny", "25"])


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_basic_run() -> None:
    """AzureAIAgentClient を使った ChatAgent の基本的な実行機能をテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
    ) as agent:
        # シンプルなクエリを実行する
        response = await agent.run("Hello! Please respond with 'Hello World' exactly.")

        # レスポンスを検証する
        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0
        assert "Hello World" in response.text


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_basic_run_streaming() -> None:
    """AzureAIAgentClient を使った ChatAgent の基本的なストリーミング機能をテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
    ) as agent:
        # ストリーミングクエリを実行する
        full_message: str = ""
        async for chunk in agent.run_stream("Please respond with exactly: 'This is a streaming response test.'"):
            assert chunk is not None
            assert isinstance(chunk, AgentRunResponseUpdate)
            if chunk.text:
                full_message += chunk.text

        # ストリーミングレスポンスを検証する
        assert len(full_message) > 0
        assert "streaming response test" in full_message.lower()


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_thread_persistence() -> None:
    """AzureAIAgentClient を使った ChatAgent のスレッド持続性を複数回の実行でテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as agent:
        # 再利用される新しいスレッドを作成する
        thread = agent.get_new_thread()

        # 最初のメッセージ - コンテキストを確立する
        first_response = await agent.run(
            "Remember this number: 42. What number did I just tell you to remember?", thread=thread
        )
        assert isinstance(first_response, AgentRunResponse)
        assert "42" in first_response.text

        # 2番目のメッセージ - 会話のメモリをテストする
        second_response = await agent.run(
            "What number did I tell you to remember in my previous message?", thread=thread
        )
        assert isinstance(second_response, AgentRunResponse)
        assert "42" in second_response.text


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_existing_thread_id() -> None:
    """AzureAIAgentClient を使った ChatAgent の既存スレッドID機能をテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as first_agent:
        # 会話を開始してスレッドIDを取得する
        thread = first_agent.get_new_thread()
        first_response = await first_agent.run("My name is Alice. Remember this.", thread=thread)

        # 最初のレスポンスを検証する
        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None

        # 最初のレスポンス後にスレッドIDが設定される
        existing_thread_id = thread.service_thread_id
        assert existing_thread_id is not None

    # 新しいエージェントインスタンスで同じスレッドIDを使って続行する
    async with ChatAgent(
        chat_client=AzureAIAgentClient(thread_id=existing_thread_id, async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as second_agent:
        # 既存のIDでスレッドを作成する
        thread = AgentThread(service_thread_id=existing_thread_id)

        # 前の会話について質問する
        response2 = await second_agent.run("What is my name?", thread=thread)

        # エージェントが前の会話を覚えていることを検証する
        assert isinstance(response2, AgentRunResponse)
        assert response2.text is not None
        # 前の会話のAliceを参照しているはずである
        assert "alice" in response2.text.lower()


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_code_interpreter():
    """AzureAIAgentClient を通じてコードインタープリターを使った ChatAgent をテストする。"""

    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant that can write and execute Python code.",
        tools=[HostedCodeInterpreterTool()],
    ) as agent:
        # コード実行をリクエストする
        response = await agent.run("Write Python code to calculate the factorial of 5 and show the result.")

        # レスポンスを検証する
        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        # 5の階乗は120である
        assert "120" in response.text or "factorial" in response.text.lower()


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_file_search():
    """AzureAIAgentClient を通じてファイル検索を使った ChatAgent をテストする。"""

    client = AzureAIAgentClient(async_credential=AzureCliCredential())
    file: FileInfo | None = None
    vector_store: VectorStore | None = None

    try:
        # 1. テストファイルを読み込み、Azure AI agent サービスにアップロードする
        test_file_path = Path(__file__).parent / "resources" / "employees.pdf"
        file = await client.project_client.agents.files.upload_and_poll(
            file_path=str(test_file_path), purpose="assistants"
        )
        vector_store = await client.project_client.agents.vector_stores.create_and_poll(
            file_ids=[file.id], name="test_employees_vectorstore"
        )

        # 2. アップロードしたリソースでファイル検索ツールを作成する
        file_search_tool = HostedFileSearchTool(inputs=[HostedVectorStoreContent(vector_store_id=vector_store.id)])

        async with ChatAgent(
            chat_client=client,
            instructions="You are a helpful assistant that can search through uploaded employee files.",
            tools=[file_search_tool],
        ) as agent:
            # 3. ファイル検索機能をテストする
            response = await agent.run("Who is the youngest employee in the files?")

            # レスポンスを検証する
            assert isinstance(response, AgentRunResponse)
            assert response.text is not None
            # Alice Johnson（24歳）が最年少である情報を見つけるはずである
            assert any(term in response.text.lower() for term in ["alice", "johnson", "24"])

    finally:
        # 4. クリーンアップ：ベクターストアとファイルを削除する
        try:
            if vector_store:
                await client.project_client.agents.vector_stores.delete(vector_store.id)
            if file:
                await client.project_client.agents.files.delete(file.id)
        except Exception:
            # 実際のテスト失敗を隠さないようにクリーンアップエラーは無視する
            pass
        finally:
            await client.close()


@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_hosted_mcp_tool() -> None:
    """Microsoft Learn MCP を使った Azure AI Agent の HostedMCPTool の統合テスト。"""

    mcp_tool = HostedMCPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
        description="A Microsoft Learn MCP server for documentation questions",
        approval_mode="never_require",
    )

    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=[mcp_tool],
    ) as agent:
        response = await agent.run(
            "How to create an Azure storage account using az cli?",
            max_tokens=200,
        )

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0

        # never_require 承認モードでは承認リクエストが発生しないはずである
        assert len(response.user_input_requests) == 0, (
            f"Expected no approval requests with never_require mode, but got {len(response.user_input_requests)}"
        )

        # Azure CLI について尋ねているため、Azure 関連の内容が含まれているはずである
        assert any(term in response.text.lower() for term in ["azure", "storage", "account", "cli"])


@pytest.mark.flaky
@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_level_tool_persistence():
    """AzureAIAgentClient を使ったエージェントレベルのツールが複数回の実行で持続することをテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant that uses available tools.",
        tools=[get_weather],
    ) as agent:
        # 最初の実行 - エージェントレベルのツールが利用可能であるはずである
        first_response = await agent.run("What's the weather like in Chicago?")

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None
        # エージェントレベルの天気ツールを使うはずである
        assert any(term in first_response.text.lower() for term in ["chicago", "sunny", "25"])

        # 2回目の実行 - エージェントレベルのツールがまだ利用可能であるはずである（持続性テスト）
        second_response = await agent.run("What's the weather in Miami?")

        assert isinstance(second_response, AgentRunResponse)
        assert second_response.text is not None
        # 再びエージェントレベルの天気ツールを使うはずである
        assert any(term in second_response.text.lower() for term in ["miami", "sunny", "25"])


@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_chat_options_run_level() -> None:
    """実行レベルでの ChatOptions パラメータのカバレッジをテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant.",
    ) as agent:
        response = await agent.run(
            "Provide a brief, helpful response.",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            seed=123,
            user="comprehensive-test-user",
            tools=[get_weather],
            tool_choice="auto",
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            store=True,
            logit_bias={"test": 1},
            metadata={"test": "value"},
            additional_properties={"custom_param": "test_value"},
        )

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0


@skip_if_azure_ai_integration_tests_disabled
async def test_azure_ai_chat_client_agent_chat_options_agent_level() -> None:
    """エージェントレベルでの ChatOptions パラメータのカバレッジをテストする。"""
    async with ChatAgent(
        chat_client=AzureAIAgentClient(async_credential=AzureCliCredential()),
        instructions="You are a helpful assistant.",
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        seed=123,
        user="comprehensive-test-user",
        tools=[get_weather],
        tool_choice="auto",
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=["END"],
        store=True,
        logit_bias={"test": 1},
        metadata={"test": "value"},
        request_kwargs={"custom_param": "test_value"},
    ) as agent:
        response = await agent.run(
            "Provide a brief, helpful response.",
        )

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0
