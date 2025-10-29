# Copyright (c) Microsoft. All rights reserved.

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from azure.identity import AzureCliCredential
from httpx import Request, Response
from openai import AsyncAzureOpenAI, AsyncStream
from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as ChunkChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    BaseChatClient,
    ChatAgent,
    ChatClientProtocol,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    TextContent,
    ai_function,
)
from agent_framework._telemetry import USER_AGENT_KEY
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.exceptions import ServiceInitializationError, ServiceResponseException
from agent_framework.openai import (
    ContentFilterResultSeverity,
    OpenAIContentFilterException,
)

# region サービスセットアップ

skip_if_azure_integration_tests_disabled = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
    or os.getenv("AZURE_OPENAI_ENDPOINT", "") in ("", "https://test-endpoint.com"),
    reason="No real AZURE_OPENAI_ENDPOINT provided; skipping integration tests."
    if os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    else "Integration tests are disabled.",
)


def test_init(azure_openai_unit_test_env: dict[str, str]) -> None:
    # 正常な初期化のテスト
    azure_chat_client = AzureOpenAIChatClient()

    assert azure_chat_client.client is not None
    assert isinstance(azure_chat_client.client, AsyncAzureOpenAI)
    assert azure_chat_client.model_id == azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    assert isinstance(azure_chat_client, BaseChatClient)


def test_init_client(azure_openai_unit_test_env: dict[str, str]) -> None:
    # クライアントを使った正常な初期化のテスト
    client = MagicMock(spec=AsyncAzureOpenAI)
    azure_chat_client = AzureOpenAIChatClient(async_client=client)

    assert azure_chat_client.client is not None
    assert isinstance(azure_chat_client.client, AsyncAzureOpenAI)


def test_init_base_url(azure_openai_unit_test_env: dict[str, str]) -> None:
    # テスト用のカスタムヘッダー
    default_headers = {"X-Unit-Test": "test-guid"}

    azure_chat_client = AzureOpenAIChatClient(
        default_headers=default_headers,
    )

    assert azure_chat_client.client is not None
    assert isinstance(azure_chat_client.client, AsyncAzureOpenAI)
    assert azure_chat_client.model_id == azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    assert isinstance(azure_chat_client, BaseChatClient)
    for key, value in default_headers.items():
        assert key in azure_chat_client.client.default_headers
        assert azure_chat_client.client.default_headers[key] == value


def test_azure_openai_chat_client_instructions_sent_once(azure_openai_unit_test_env: dict[str, str]) -> None:
    """Azure OpenAIチャットリクエストの準備時に指示が一度だけ含まれることを保証する。"""
    client = AzureOpenAIChatClient()
    instructions = "You are a helpful assistant."
    chat_options = ChatOptions(instructions=instructions)

    prepared_messages = client.prepare_messages([ChatMessage(role="user", text="Hello")], chat_options)
    request_options = client._prepare_options(prepared_messages, chat_options)  # type: ignore[reportPrivateUsage]

    assert json.dumps(request_options).count(instructions) == 1


@pytest.mark.parametrize("exclude_list", [["AZURE_OPENAI_BASE_URL"]], indirect=True)
def test_init_endpoint(azure_openai_unit_test_env: dict[str, str]) -> None:
    azure_chat_client = AzureOpenAIChatClient()

    assert azure_chat_client.client is not None
    assert isinstance(azure_chat_client.client, AsyncAzureOpenAI)
    assert azure_chat_client.model_id == azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    assert isinstance(azure_chat_client, BaseChatClient)


@pytest.mark.parametrize("exclude_list", [["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]], indirect=True)
def test_init_with_empty_deployment_name(azure_openai_unit_test_env: dict[str, str]) -> None:
    with pytest.raises(ServiceInitializationError):
        AzureOpenAIChatClient(
            env_file_path="test.env",
        )


@pytest.mark.parametrize("exclude_list", [["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL"]], indirect=True)
def test_init_with_empty_endpoint_and_base_url(azure_openai_unit_test_env: dict[str, str]) -> None:
    with pytest.raises(ServiceInitializationError):
        AzureOpenAIChatClient(
            env_file_path="test.env",
        )


@pytest.mark.parametrize("override_env_param_dict", [{"AZURE_OPENAI_ENDPOINT": "http://test.com"}], indirect=True)
def test_init_with_invalid_endpoint(azure_openai_unit_test_env: dict[str, str]) -> None:
    with pytest.raises(ServiceInitializationError):
        AzureOpenAIChatClient()


@pytest.mark.parametrize("exclude_list", [["AZURE_OPENAI_BASE_URL"]], indirect=True)
def test_serialize(azure_openai_unit_test_env: dict[str, str]) -> None:
    default_headers = {"X-Test": "test"}

    settings = {
        "deployment_name": azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        "endpoint": azure_openai_unit_test_env["AZURE_OPENAI_ENDPOINT"],
        "api_key": azure_openai_unit_test_env["AZURE_OPENAI_API_KEY"],
        "api_version": azure_openai_unit_test_env["AZURE_OPENAI_API_VERSION"],
        "default_headers": default_headers,
        "env_file_path": "test.env",
    }

    azure_chat_client = AzureOpenAIChatClient.from_dict(settings)
    dumped_settings = azure_chat_client.to_dict()
    assert dumped_settings["model_id"] == settings["deployment_name"]
    assert str(settings["endpoint"]) in str(dumped_settings["endpoint"])
    assert str(settings["deployment_name"]) == str(dumped_settings["deployment_name"])
    assert settings["api_version"] == dumped_settings["api_version"]
    assert "api_key" not in dumped_settings

    # 追加したデフォルトヘッダーがdumped_settingsのデフォルトヘッダーに存在することをアサートする
    for key, value in default_headers.items():
        assert key in dumped_settings["default_headers"]
        assert dumped_settings["default_headers"][key] == value

    # 'User-agent'ヘッダーがdumped_settingsのデフォルトヘッダーに存在しないことをアサートする
    assert USER_AGENT_KEY not in dumped_settings["default_headers"]


# endregion region CMC


@pytest.fixture
def mock_chat_completion_response() -> ChatCompletion:
    return ChatCompletion(
        id="test_id",
        choices=[
            Choice(index=0, message=ChatCompletionMessage(content="test", role="assistant"), finish_reason="stop")
        ],
        created=0,
        model="test",
        object="chat.completion",
    )


@pytest.fixture
def mock_streaming_chat_completion_response() -> AsyncStream[ChatCompletionChunk]:
    content = ChatCompletionChunk(
        id="test_id",
        choices=[ChunkChoice(index=0, delta=ChunkChoiceDelta(content="test", role="assistant"), finish_reason="stop")],
        created=0,
        model="test",
        object="chat.completion.chunk",
    )
    stream = MagicMock(spec=AsyncStream)
    stream.__aiter__.return_value = [content]
    return stream


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_cmc(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_chat_completion_response: ChatCompletion,
) -> None:
    mock_create.return_value = mock_chat_completion_response
    chat_history.append(ChatMessage(text="hello world", role="user"))

    azure_chat_client = AzureOpenAIChatClient()
    await azure_chat_client.get_response(
        messages=chat_history,
    )
    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        stream=False,
        messages=azure_chat_client._prepare_chat_history_for_request(chat_history),  # type: ignore
    )


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_cmc_with_logit_bias(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_chat_completion_response: ChatCompletion,
) -> None:
    mock_create.return_value = mock_chat_completion_response
    prompt = "hello world"
    chat_history.append(ChatMessage(text=prompt, role="user"))

    token_bias: dict[str | int, float] = {"1": -100}

    azure_chat_client = AzureOpenAIChatClient()

    await azure_chat_client.get_response(messages=chat_history, logit_bias=token_bias)

    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=azure_chat_client._prepare_chat_history_for_request(chat_history),  # type: ignore
        stream=False,
        logit_bias=token_bias,
    )


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_cmc_with_stop(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_chat_completion_response: ChatCompletion,
) -> None:
    mock_create.return_value = mock_chat_completion_response
    prompt = "hello world"
    chat_history.append(ChatMessage(text=prompt, role="user"))

    stop = ["!"]

    azure_chat_client = AzureOpenAIChatClient()

    await azure_chat_client.get_response(messages=chat_history, stop=stop)

    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=azure_chat_client._prepare_chat_history_for_request(chat_history),  # type: ignore
        stream=False,
        stop=stop,
    )


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_azure_on_your_data(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_chat_completion_response: ChatCompletion,
) -> None:
    mock_chat_completion_response.choices = [
        Choice(
            index=0,
            message=ChatCompletionMessage(
                content="test",
                role="assistant",
                context={  # type: ignore
                    "citations": [
                        {
                            "content": "test content",
                            "title": "test title",
                            "url": "test url",
                            "filepath": "test filepath",
                            "chunk_id": "test chunk_id",
                        }
                    ],
                    "intent": "query used",
                },
            ),
            finish_reason="stop",
        )
    ]
    mock_create.return_value = mock_chat_completion_response
    prompt = "hello world"
    messages_in = chat_history
    chat_history.append(ChatMessage(text=prompt, role="user"))
    messages_out: list[ChatMessage] = []
    messages_out.append(ChatMessage(text=prompt, role="user"))

    expected_data_settings = {
        "data_sources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "indexName": "test_index",
                    "endpoint": "https://test-endpoint-search.com",
                    "key": "test_key",
                },
            }
        ]
    }

    azure_chat_client = AzureOpenAIChatClient()

    content = await azure_chat_client.get_response(
        messages=messages_in,
        additional_properties={"extra_body": expected_data_settings},
    )
    assert len(content.messages) == 1
    assert len(content.messages[0].contents) == 1
    assert isinstance(content.messages[0].contents[0], TextContent)
    assert len(content.messages[0].contents[0].annotations) == 1
    assert content.messages[0].contents[0].annotations[0].title == "test title"
    assert content.messages[0].contents[0].text == "test"

    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=azure_chat_client._prepare_chat_history_for_request(messages_out),  # type: ignore
        stream=False,
        extra_body=expected_data_settings,
    )


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_azure_on_your_data_string(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_chat_completion_response: ChatCompletion,
) -> None:
    mock_chat_completion_response.choices = [
        Choice(
            index=0,
            message=ChatCompletionMessage(
                content="test",
                role="assistant",
                context=json.dumps({  # type: ignore
                    "citations": [
                        {
                            "content": "test content",
                            "title": "test title",
                            "url": "test url",
                            "filepath": "test filepath",
                            "chunk_id": "test chunk_id",
                        }
                    ],
                    "intent": "query used",
                }),
            ),
            finish_reason="stop",
        )
    ]
    mock_create.return_value = mock_chat_completion_response
    prompt = "hello world"
    messages_in = chat_history
    messages_in.append(ChatMessage(text=prompt, role="user"))
    messages_out: list[ChatMessage] = []
    messages_out.append(ChatMessage(text=prompt, role="user"))

    expected_data_settings = {
        "data_sources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "indexName": "test_index",
                    "endpoint": "https://test-endpoint-search.com",
                    "key": "test_key",
                },
            }
        ]
    }

    azure_chat_client = AzureOpenAIChatClient()

    content = await azure_chat_client.get_response(
        messages=messages_in,
        additional_properties={"extra_body": expected_data_settings},
    )
    assert len(content.messages) == 1
    assert len(content.messages[0].contents) == 1
    assert isinstance(content.messages[0].contents[0], TextContent)
    assert len(content.messages[0].contents[0].annotations) == 1
    assert content.messages[0].contents[0].annotations[0].title == "test title"
    assert content.messages[0].contents[0].text == "test"

    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=azure_chat_client._prepare_chat_history_for_request(messages_out),  # type: ignore
        stream=False,
        extra_body=expected_data_settings,
    )


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_azure_on_your_data_fail(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_chat_completion_response: ChatCompletion,
) -> None:
    mock_chat_completion_response.choices = [
        Choice(
            index=0,
            message=ChatCompletionMessage(
                content="test",
                role="assistant",
                context="not a dictionary",  # type: ignore
            ),
            finish_reason="stop",
        )
    ]
    mock_create.return_value = mock_chat_completion_response
    prompt = "hello world"
    messages_in = chat_history
    messages_in.append(ChatMessage(text=prompt, role="user"))
    messages_out: list[ChatMessage] = []
    messages_out.append(ChatMessage(text=prompt, role="user"))

    expected_data_settings = {
        "data_sources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "indexName": "test_index",
                    "endpoint": "https://test-endpoint-search.com",
                    "key": "test_key",
                },
            }
        ]
    }

    azure_chat_client = AzureOpenAIChatClient()

    content = await azure_chat_client.get_response(
        messages=messages_in,
        additional_properties={"extra_body": expected_data_settings},
    )
    assert len(content.messages) == 1
    assert len(content.messages[0].contents) == 1
    assert isinstance(content.messages[0].contents[0], TextContent)
    assert content.messages[0].contents[0].text == "test"

    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=azure_chat_client._prepare_chat_history_for_request(messages_out),  # type: ignore
        stream=False,
        extra_body=expected_data_settings,
    )


CONTENT_FILTERED_ERROR_MESSAGE = (
    "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please "
    "modify your prompt and retry. To learn more about our content filtering policies please read our "
    "documentation: https://go.microsoft.com/fwlink/?linkid=2198766"
)
CONTENT_FILTERED_ERROR_FULL_MESSAGE = (
    "Error code: 400 - {'error': {'message': \"%s\", 'type': null, 'param': 'prompt', 'code': 'content_filter', "
    "'status': 400, 'innererror': {'code': 'ResponsibleAIPolicyViolation', 'content_filter_result': {'hate': "
    "{'filtered': True, 'severity': 'high'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': "
    "{'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}}}"
) % CONTENT_FILTERED_ERROR_MESSAGE


@patch.object(AsyncChatCompletions, "create")
async def test_content_filtering_raises_correct_exception(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
) -> None:
    prompt = "some prompt that would trigger the content filtering"
    chat_history.append(ChatMessage(text=prompt, role="user"))

    test_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert test_endpoint is not None
    mock_create.side_effect = openai.BadRequestError(
        CONTENT_FILTERED_ERROR_FULL_MESSAGE,
        response=Response(400, request=Request("POST", test_endpoint)),
        body={
            "message": CONTENT_FILTERED_ERROR_MESSAGE,
            "type": None,
            "param": "prompt",
            "code": "content_filter",
            "status": 400,
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": {
                    "hate": {"filtered": True, "severity": "high"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            },
        },
    )

    azure_chat_client = AzureOpenAIChatClient()

    with pytest.raises(OpenAIContentFilterException, match="service encountered a content error") as exc_info:
        await azure_chat_client.get_response(
            messages=chat_history,
        )

    content_filter_exc = exc_info.value
    assert content_filter_exc.param == "prompt"
    assert content_filter_exc.content_filter_result["hate"].filtered
    assert content_filter_exc.content_filter_result["hate"].severity == ContentFilterResultSeverity.HIGH


@patch.object(AsyncChatCompletions, "create")
async def test_content_filtering_without_response_code_raises_with_default_code(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
) -> None:
    prompt = "some prompt that would trigger the content filtering"
    chat_history.append(ChatMessage(text=prompt, role="user"))

    test_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert test_endpoint is not None
    mock_create.side_effect = openai.BadRequestError(
        CONTENT_FILTERED_ERROR_FULL_MESSAGE,
        response=Response(400, request=Request("POST", test_endpoint)),
        body={
            "message": CONTENT_FILTERED_ERROR_MESSAGE,
            "type": None,
            "param": "prompt",
            "code": "content_filter",
            "status": 400,
            "innererror": {
                "content_filter_result": {
                    "hate": {"filtered": True, "severity": "high"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            },
        },
    )

    azure_chat_client = AzureOpenAIChatClient()

    with pytest.raises(OpenAIContentFilterException, match="service encountered a content error"):
        await azure_chat_client.get_response(
            messages=chat_history,
        )


@patch.object(AsyncChatCompletions, "create")
async def test_bad_request_non_content_filter(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
) -> None:
    prompt = "some prompt that would trigger the content filtering"
    chat_history.append(ChatMessage(text=prompt, role="user"))

    test_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert test_endpoint is not None
    mock_create.side_effect = openai.BadRequestError(
        "The request was bad.", response=Response(400, request=Request("POST", test_endpoint)), body={}
    )

    azure_chat_client = AzureOpenAIChatClient()

    with pytest.raises(ServiceResponseException, match="service failed to complete the prompt"):
        await azure_chat_client.get_response(
            messages=chat_history,
        )


@patch.object(AsyncChatCompletions, "create", new_callable=AsyncMock)
async def test_get_streaming(
    mock_create: AsyncMock,
    azure_openai_unit_test_env: dict[str, str],
    chat_history: list[ChatMessage],
    mock_streaming_chat_completion_response: AsyncStream[ChatCompletionChunk],
) -> None:
    mock_create.return_value = mock_streaming_chat_completion_response
    chat_history.append(ChatMessage(text="hello world", role="user"))

    azure_chat_client = AzureOpenAIChatClient()
    async for msg in azure_chat_client.get_streaming_response(
        messages=chat_history,
    ):
        assert msg is not None
        assert msg.message_id is not None
        assert msg.response_id is not None
    mock_create.assert_awaited_once_with(
        model=azure_openai_unit_test_env["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        stream=True,
        messages=azure_chat_client._prepare_chat_history_for_request(chat_history),  # type: ignore
        # 注意: `stream_options={"include_usage": True}` は
        # `OpenAIChatCompletionBase._inner_get_streaming_response` で明示的に強制されています。
        # 一貫性を保つために、ここでも引数を合わせています。
        stream_options={"include_usage": True},
    )


@ai_function
def get_story_text() -> str:
    """EmilyとDavidの物語を返す。"""
    return (
        "Emily and David, two passionate scientists, met during a research expedition to Antarctica. "
        "Bonded by their love for the natural world and shared curiosity, they uncovered a "
        "groundbreaking phenomenon in glaciology that could potentially reshape our understanding "
        "of climate change."
    )


@ai_function
def get_weather(location: str) -> str:
    """指定された場所の現在の天気を取得する。"""
    return f"The weather in {location} is sunny and 72°F."


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_response() -> None:
    """Azure OpenAIチャット完了レスポンスのテスト。"""
    azure_chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    assert isinstance(azure_chat_client, ChatClientProtocol)

    messages: list[ChatMessage] = []
    messages.append(
        ChatMessage(
            role="user",
            text="Emily and David, two passionate scientists, met during a research expedition to Antarctica. "
            "Bonded by their love for the natural world and shared curiosity, they uncovered a "
            "groundbreaking phenomenon in glaciology that could potentially reshape our understanding "
            "of climate change.",
        )
    )
    messages.append(ChatMessage(role="user", text="who are Emily and David?"))

    # クライアントがレスポンス取得に使用できることをテストする
    response = await azure_chat_client.get_response(messages=messages)

    assert response is not None
    assert isinstance(response, ChatResponse)
    # AIがコンテキストを理解したことを示す関連キーワードをチェックする
    assert any(
        word in response.text.lower() for word in ["scientists", "research", "antarctica", "glaciology", "climate"]
    )


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_response_tools() -> None:
    """AzureOpenAIチャット完了レスポンスのテスト。"""
    azure_chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    assert isinstance(azure_chat_client, ChatClientProtocol)

    messages: list[ChatMessage] = []
    messages.append(ChatMessage(role="user", text="who are Emily and David?"))

    # クライアントがレスポンス取得に使用できることをテストする
    response = await azure_chat_client.get_response(
        messages=messages,
        tools=[get_story_text],
        tool_choice="auto",
    )

    assert response is not None
    assert isinstance(response, ChatResponse)
    assert "scientists" in response.text


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_streaming() -> None:
    """Azure OpenAIチャット完了レスポンスのテスト。"""
    azure_chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    assert isinstance(azure_chat_client, ChatClientProtocol)

    messages: list[ChatMessage] = []
    messages.append(
        ChatMessage(
            role="user",
            text="Emily and David, two passionate scientists, met during a research expedition to Antarctica. "
            "Bonded by their love for the natural world and shared curiosity, they uncovered a "
            "groundbreaking phenomenon in glaciology that could potentially reshape our understanding "
            "of climate change.",
        )
    )
    messages.append(ChatMessage(role="user", text="who are Emily and David?"))

    # クライアントがレスポンス取得に使用できることをテストする
    response = azure_chat_client.get_streaming_response(messages=messages)

    full_message: str = ""
    async for chunk in response:
        assert chunk is not None
        assert isinstance(chunk, ChatResponseUpdate)
        assert chunk.message_id is not None
        assert chunk.response_id is not None
        for content in chunk.contents:
            if isinstance(content, TextContent) and content.text:
                full_message += content.text

    assert "scientists" in full_message


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_streaming_tools() -> None:
    """AzureOpenAIチャット完了レスポンスのテスト。"""
    azure_chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    assert isinstance(azure_chat_client, ChatClientProtocol)

    messages: list[ChatMessage] = []
    messages.append(ChatMessage(role="user", text="who are Emily and David?"))

    # クライアントがレスポンス取得に使用できることをテストする
    response = azure_chat_client.get_streaming_response(
        messages=messages,
        tools=[get_story_text],
        tool_choice="auto",
    )
    full_message: str = ""
    async for chunk in response:
        assert chunk is not None
        assert isinstance(chunk, ChatResponseUpdate)
        for content in chunk.contents:
            if isinstance(content, TextContent) and content.text:
                full_message += content.text

    assert "scientists" in full_message


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_agent_basic_run():
    """AzureOpenAIChatClientを使ったAzure OpenAIチャットクライアントエージェントの基本的な実行機能のテスト。"""
    async with ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
    ) as agent:
        # 基本的な実行のテスト
        response = await agent.run("Please respond with exactly: 'This is a response test.'")

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0
        assert "response test" in response.text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_agent_basic_run_streaming():
    """AzureOpenAIChatClientを使ったAzure OpenAIチャットクライアントエージェントの基本的なストリーミング機能のテスト。"""
    async with ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
    ) as agent:
        # ストリーミング実行のテスト
        full_text = ""
        async for chunk in agent.run_stream("Please respond with exactly: 'This is a streaming response test.'"):
            assert isinstance(chunk, AgentRunResponseUpdate)
            if chunk.text:
                full_text += chunk.text

        assert len(full_text) > 0
        assert "streaming response test" in full_text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_agent_thread_persistence():
    """AzureOpenAIChatClientを使ったAzure OpenAIチャットクライアントエージェントの実行間でのスレッド永続性のテスト。"""
    async with ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as agent:
        # 再利用される新しいスレッドを作成する
        thread = agent.get_new_thread()

        # 最初のインタラクション
        response1 = await agent.run("My name is Alice. Remember this.", thread=thread)

        assert isinstance(response1, AgentRunResponse)
        assert response1.text is not None

        # 2回目のインタラクション - メモリをテストする
        response2 = await agent.run("What is my name?", thread=thread)

        assert isinstance(response2, AgentRunResponse)
        assert response2.text is not None
        assert "alice" in response2.text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_openai_chat_client_agent_existing_thread():
    """既存のスレッドを使ってAzure OpenAIチャットクライアントエージェントがエージェントインスタンス間で会話を継続できるかテストする。"""
    # 最初の会話 - スレッドを取得する
    preserved_thread = None

    async with ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as first_agent:
        # 会話を開始してスレッドを取得する
        thread = first_agent.get_new_thread()
        first_response = await first_agent.run("My name is Alice. Remember this.", thread=thread)

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None

        # スレッドを保存して再利用する
        preserved_thread = thread

    # 2回目の会話 - 新しいエージェントインスタンスでスレッドを再利用する
    if preserved_thread:
        async with ChatAgent(
            chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
            instructions="You are a helpful assistant with good memory.",
        ) as second_agent:
            # 保存したスレッドを再利用する
            second_response = await second_agent.run("What is my name?", thread=preserved_thread)

            assert isinstance(second_response, AgentRunResponse)
            assert second_response.text is not None
            assert "alice" in second_response.text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_chat_client_agent_level_tool_persistence():
    """Azure Chat Clientでエージェントレベルのツールが複数回の実行で持続することをテストする。"""

    async with ChatAgent(
        chat_client=AzureOpenAIChatClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant that uses available tools.",
        tools=[get_weather],  # Agent-level tool
    ) as agent:
        # 最初の実行 - エージェントレベルのツールが利用可能であるべき
        first_response = await agent.run("What's the weather like in Chicago?")

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None
        # エージェントレベルの天気ツールを使用するはずである
        assert any(term in first_response.text.lower() for term in ["chicago", "sunny", "72"])

        # 2回目の実行 - エージェントレベルのツールがまだ利用可能であるべき（持続性テスト）
        second_response = await agent.run("What's the weather in Miami?")

        assert isinstance(second_response, AgentRunResponse)
        assert second_response.text is not None
        # 再びエージェントレベルの天気ツールを使用するはずである
        assert any(term in second_response.text.lower() for term in ["miami", "sunny", "72"])
