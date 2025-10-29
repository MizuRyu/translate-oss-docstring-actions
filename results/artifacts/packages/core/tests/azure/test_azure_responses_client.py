# Copyright (c) Microsoft. All rights reserved.

import json
import os
from typing import Annotated

import pytest
from azure.identity import AzureCliCredential
from pydantic import BaseModel

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
    HostedCodeInterpreterTool,
    HostedFileSearchTool,
    HostedMCPTool,
    HostedVectorStoreContent,
    TextContent,
    ai_function,
)
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.exceptions import ServiceInitializationError

skip_if_azure_integration_tests_disabled = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true"
    or os.getenv("AZURE_OPENAI_ENDPOINT", "") in ("", "https://test-endpoint.com"),
    reason="No real AZURE_OPENAI_ENDPOINT provided; skipping integration tests."
    if os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    else "Integration tests are disabled.",
)


class OutputStruct(BaseModel):
    """テスト目的のための構造化された出力。"""

    location: str
    weather: str


@ai_function
async def get_weather(location: Annotated[str, "The location as a city name"]) -> str:
    """指定された場所の現在の天気を取得する。"""
    # 天気を取得するツールの実装。
    return f"The weather in {location} is sunny and 72°F."


async def create_vector_store(client: AzureOpenAIResponsesClient) -> tuple[str, HostedVectorStoreContent]:
    """テスト用のサンプルドキュメントでベクターストアを作成する。"""
    file = await client.client.files.create(
        file=("todays_weather.txt", b"The weather today is sunny with a high of 75F."), purpose="assistants"
    )
    vector_store = await client.client.vector_stores.create(
        name="knowledge_base",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    result = await client.client.vector_stores.files.create_and_poll(vector_store_id=vector_store.id, file_id=file.id)
    if result.last_error is not None:
        raise Exception(f"Vector store file processing failed with status: {result.last_error.message}")

    return file.id, HostedVectorStoreContent(vector_store_id=vector_store.id)


async def delete_vector_store(client: AzureOpenAIResponsesClient, file_id: str, vector_store_id: str) -> None:
    """テスト後にベクターストアを削除する。"""

    await client.client.vector_stores.delete(vector_store_id=vector_store_id)
    await client.client.files.delete(file_id=file_id)


def test_init(azure_openai_unit_test_env: dict[str, str]) -> None:
    # 正常な初期化のテスト
    azure_responses_client = AzureOpenAIResponsesClient()

    assert azure_responses_client.model_id == azure_openai_unit_test_env["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"]
    assert isinstance(azure_responses_client, ChatClientProtocol)


def test_init_validation_fail() -> None:
    # 正常な初期化のテスト
    with pytest.raises(ServiceInitializationError):
        AzureOpenAIResponsesClient(api_key="34523", deployment_name={"test": "dict"})  # type: ignore


def test_init_model_id_constructor(azure_openai_unit_test_env: dict[str, str]) -> None:
    # 正常な初期化のテスト
    model_id = "test_model_id"
    azure_responses_client = AzureOpenAIResponsesClient(deployment_name=model_id)

    assert azure_responses_client.model_id == model_id
    assert isinstance(azure_responses_client, ChatClientProtocol)


def test_init_with_default_header(azure_openai_unit_test_env: dict[str, str]) -> None:
    default_headers = {"X-Unit-Test": "test-guid"}

    # 正常な初期化のテスト
    azure_responses_client = AzureOpenAIResponsesClient(
        default_headers=default_headers,
    )

    assert azure_responses_client.model_id == azure_openai_unit_test_env["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"]
    assert isinstance(azure_responses_client, ChatClientProtocol)

    # 追加したデフォルトヘッダーがクライアントのデフォルトヘッダーに存在することをアサートする
    for key, value in default_headers.items():
        assert key in azure_responses_client.client.default_headers
        assert azure_responses_client.client.default_headers[key] == value


def test_azure_responses_client_instructions_sent_once(azure_openai_unit_test_env: dict[str, str]) -> None:
    """Azure OpenAI Responsesリクエストで指示が一度だけ含まれることを保証する。"""
    client = AzureOpenAIResponsesClient()
    instructions = "You are a helpful assistant."
    chat_options = ChatOptions(instructions=instructions)

    prepared_messages = client.prepare_messages([ChatMessage(role="user", text="Hello")], chat_options)
    request_options = client._prepare_options(prepared_messages, chat_options)  # type: ignore[reportPrivateUsage]

    assert json.dumps(request_options).count(instructions) == 1


@pytest.mark.parametrize("exclude_list", [["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"]], indirect=True)
def test_init_with_empty_model_id(azure_openai_unit_test_env: dict[str, str]) -> None:
    with pytest.raises(ServiceInitializationError):
        AzureOpenAIResponsesClient(
            env_file_path="test.env",
        )


def test_serialize(azure_openai_unit_test_env: dict[str, str]) -> None:
    default_headers = {"X-Unit-Test": "test-guid"}

    settings = {
        "deployment_name": azure_openai_unit_test_env["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"],
        "api_key": azure_openai_unit_test_env["AZURE_OPENAI_API_KEY"],
        "default_headers": default_headers,
    }

    azure_responses_client = AzureOpenAIResponsesClient.from_dict(settings)
    dumped_settings = azure_responses_client.to_dict()
    assert dumped_settings["deployment_name"] == azure_openai_unit_test_env["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"]
    assert "api_key" not in dumped_settings
    # 追加したデフォルトヘッダーが dumped_settings のデフォルトヘッダーに存在することをアサートする
    for key, value in default_headers.items():
        assert key in dumped_settings["default_headers"]
        assert dumped_settings["default_headers"][key] == value
    # 'User-Agent' ヘッダーが dumped_settings のデフォルトヘッダーに存在しないことをアサートする
    assert "User-Agent" not in dumped_settings["default_headers"]


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_response() -> None:
    """azure responses client のレスポンスをテストする。"""
    azure_responses_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    assert isinstance(azure_responses_client, ChatClientProtocol)

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

    # クライアントがレスポンスを取得できることをテストする
    response = await azure_responses_client.get_response(messages=messages)

    assert response is not None
    assert isinstance(response, ChatResponse)
    assert "scientists" in response.text

    messages.clear()
    messages.append(ChatMessage(role="user", text="The weather in New York is sunny"))
    messages.append(ChatMessage(role="user", text="What is the weather in New York?"))

    # クライアントが構造化されたレスポンスを取得できることをテストする
    structured_response = await azure_responses_client.get_response(  # type: ignore[reportAssignmentType]
        messages=messages,
        response_format=OutputStruct,
    )

    assert structured_response is not None
    assert isinstance(structured_response, ChatResponse)
    assert isinstance(structured_response.value, OutputStruct)
    assert structured_response.value.location == "New York"
    assert "sunny" in structured_response.value.weather.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_response_tools() -> None:
    """azure responses client のツールをテストする。"""
    azure_responses_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    assert isinstance(azure_responses_client, ChatClientProtocol)

    messages: list[ChatMessage] = []
    messages.append(ChatMessage(role="user", text="What is the weather in New York?"))

    # クライアントがレスポンスを取得できることをテストする
    response = await azure_responses_client.get_response(
        messages=messages,
        tools=[get_weather],
        tool_choice="auto",
    )

    assert response is not None
    assert isinstance(response, ChatResponse)
    assert "sunny" in response.text

    messages.clear()
    messages.append(ChatMessage(role="user", text="What is the weather in Seattle?"))

    # クライアントがレスポンスを取得できることをテストする
    structured_response: ChatResponse = await azure_responses_client.get_response(  # type: ignore[reportAssignmentType]
        messages=messages,
        tools=[get_weather],
        tool_choice="auto",
        response_format=OutputStruct,
    )

    assert structured_response is not None
    assert isinstance(structured_response, ChatResponse)
    assert isinstance(structured_response.value, OutputStruct)
    assert "Seattle" in structured_response.value.location
    assert "sunny" in structured_response.value.weather.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_streaming() -> None:
    """Azure azure responses client のストリーミングレスポンスをテストする。"""
    azure_responses_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    assert isinstance(azure_responses_client, ChatClientProtocol)

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

    # クライアントがレスポンスを取得できることをテストする
    response = azure_responses_client.get_streaming_response(messages=messages)

    full_message: str = ""
    async for chunk in response:
        assert chunk is not None
        assert isinstance(chunk, ChatResponseUpdate)
        for content in chunk.contents:
            if isinstance(content, TextContent) and content.text:
                full_message += content.text

    assert "scientists" in full_message

    messages.clear()
    messages.append(ChatMessage(role="user", text="The weather in Seattle is sunny"))
    messages.append(ChatMessage(role="user", text="What is the weather in Seattle?"))

    structured_response = await ChatResponse.from_chat_response_generator(
        azure_responses_client.get_streaming_response(
            messages=messages,
            response_format=OutputStruct,
        ),
        output_format_type=OutputStruct,
    )
    assert structured_response is not None
    assert isinstance(structured_response, ChatResponse)
    assert isinstance(structured_response.value, OutputStruct)
    assert "Seattle" in structured_response.value.location
    assert "sunny" in structured_response.value.weather.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_streaming_tools() -> None:
    """azure responses client のストリーミングツールをテストする。"""
    azure_responses_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    assert isinstance(azure_responses_client, ChatClientProtocol)

    messages: list[ChatMessage] = [ChatMessage(role="user", text="What is the weather in Seattle?")]

    # クライアントがレスポンスを取得できることをテストする
    response = azure_responses_client.get_streaming_response(
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

    assert "sunny" in full_message

    messages.clear()
    messages.append(ChatMessage(role="user", text="What is the weather in Seattle?"))

    structured_response = azure_responses_client.get_streaming_response(
        messages=messages,
        tools=[get_weather],
        tool_choice="auto",
        response_format=OutputStruct,
    )
    full_message = ""
    async for chunk in structured_response:
        assert chunk is not None
        assert isinstance(chunk, ChatResponseUpdate)
        for content in chunk.contents:
            if isinstance(content, TextContent) and content.text:
                full_message += content.text

    output = OutputStruct.model_validate_json(full_message)
    assert "Seattle" in output.location
    assert "sunny" in output.weather.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_basic_run():
    """AzureOpenAIResponsesClient を使った Azure Responses Client agent の基本的な実行機能をテストする。"""
    agent = AzureOpenAIResponsesClient(credential=AzureCliCredential()).create_agent(
        instructions="You are a helpful assistant.",
    )

    # 基本的な実行をテストする
    response = await agent.run("Hello! Please respond with 'Hello World' exactly.")

    assert isinstance(response, AgentRunResponse)
    assert response.text is not None
    assert len(response.text) > 0
    assert "hello world" in response.text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_basic_run_streaming():
    """AzureOpenAIResponsesClient を使った Azure Responses Client agent の基本的なストリーミング機能をテストする。"""
    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
    ) as agent:
        # ストリーミング実行をテストする
        full_text = ""
        async for chunk in agent.run_stream("Please respond with exactly: 'This is a streaming response test.'"):
            assert isinstance(chunk, AgentRunResponseUpdate)
            if chunk.text:
                full_text += chunk.text

        assert len(full_text) > 0
        assert "streaming response test" in full_text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_thread_persistence():
    """AzureOpenAIResponsesClient を使った Azure Responses Client agent の実行間でのスレッド永続性をテストする。"""
    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as agent:
        # 再利用される新しいスレッドを作成する
        thread = agent.get_new_thread()

        # 最初のインタラクション
        first_response = await agent.run("My favorite programming language is Python. Remember this.", thread=thread)

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None

        # 2回目のインタラクション - メモリをテストする
        second_response = await agent.run("What is my favorite programming language?", thread=thread)

        assert isinstance(second_response, AgentRunResponse)
        assert second_response.text is not None


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_thread_storage_with_store_true():
    """store=True の Azure Responses Client agent をテストし、service_thread_id が返されることを検証する。"""
    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant.",
    ) as agent:
        # 新しいスレッドを作成する
        thread = AgentThread()

        # 最初は service_thread_id は None であるべき
        assert thread.service_thread_id is None

        # store=True で実行し、Azure/OpenAI 側にメッセージを保存する
        response = await agent.run(
            "Hello! Please remember that my name is Alex.",
            thread=thread,
            store=True,
        )

        # レスポンスを検証する
        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0

        # store=True の後、service_thread_id が設定されているべき
        assert thread.service_thread_id is not None
        assert isinstance(thread.service_thread_id, str)
        assert len(thread.service_thread_id) > 0


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_existing_thread():
    """既存のスレッドを使ってエージェントインスタンス間で会話を継続する Azure Responses Client agent をテストする。"""
    # 最初の会話 - スレッドをキャプチャする
    preserved_thread = None

    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant with good memory.",
    ) as first_agent:
        # 会話を開始しスレッドをキャプチャする
        thread = first_agent.get_new_thread()
        first_response = await first_agent.run("My hobby is photography. Remember this.", thread=thread)

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None

        # 再利用のためにスレッドを保持する
        preserved_thread = thread

    # 2回目の会話 - 新しいエージェントインスタンスでスレッドを再利用する
    if preserved_thread:
        async with ChatAgent(
            chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
            instructions="You are a helpful assistant with good memory.",
        ) as second_agent:
            # 保持したスレッドを再利用する
            second_response = await second_agent.run("What is my hobby?", thread=preserved_thread)

            assert isinstance(second_response, AgentRunResponse)
            assert second_response.text is not None
            assert "photography" in second_response.text.lower()


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_hosted_code_interpreter_tool():
    """AzureOpenAIResponsesClient を通じて HostedCodeInterpreterTool を使う Azure Responses Client agent をテストする。"""
    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant that can execute Python code.",
        tools=[HostedCodeInterpreterTool()],
    ) as agent:
        # コードインタプリタ機能をテストする
        response = await agent.run("Calculate the sum of numbers from 1 to 10 using Python code.")

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0
        # 計算結果（1-10 の合計 = 55）またはコード実行内容を含むべき
        contains_relevant_content = any(
            term in response.text.lower() for term in ["55", "sum", "code", "python", "calculate", "10"]
        )
        assert contains_relevant_content or len(response.text.strip()) > 10


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_level_tool_persistence():
    """Azure Responses Client でエージェントレベルのツールが複数回の実行で持続することをテストする。"""

    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant that uses available tools.",
        tools=[get_weather],  # Agent-level tool
    ) as agent:
        # 最初の実行 - エージェントレベルのツールが利用可能であるべき
        first_response = await agent.run("What's the weather like in Chicago?")

        assert isinstance(first_response, AgentRunResponse)
        assert first_response.text is not None
        # エージェントレベルの天気ツールを使うべき
        assert any(term in first_response.text.lower() for term in ["chicago", "sunny", "72"])

        # 2回目の実行 - エージェントレベルのツールがまだ利用可能であるべき（持続性テスト）
        second_response = await agent.run("What's the weather in Miami?")

        assert isinstance(second_response, AgentRunResponse)
        assert second_response.text is not None
        # 再びエージェントレベルの天気ツールを使うべき
        assert any(term in second_response.text.lower() for term in ["miami", "sunny", "72"])


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_chat_options_run_level() -> None:
    """Azure Response Agent で ChatOptions パラメータの包括的な統合テスト。"""
    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
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
        )

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_chat_options_agent_level() -> None:
    """Azure Response Agent で ChatOptions パラメータの包括的な統合テスト。"""
    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        instructions="You are a helpful assistant.",
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        seed=123,
        user="comprehensive-test-user",
        tools=[get_weather],
        tool_choice="auto",
    ) as agent:
        response = await agent.run(
            "Provide a brief, helpful response.",
        )

        assert isinstance(response, AgentRunResponse)
        assert response.text is not None
        assert len(response.text) > 0


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
async def test_azure_responses_client_agent_hosted_mcp_tool() -> None:
    """Microsoft Learn MCP を使った Azure Response Agent の HostedMCPTool 統合テスト。"""

    mcp_tool = HostedMCPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
        description="A Microsoft Learn MCP server for documentation questions",
        approval_mode="never_require",
    )

    async with ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
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
        # Azure CLI に関する質問なので Azure 関連の内容を含むべき
        assert any(term in response.text.lower() for term in ["azure", "storage", "account", "cli"])


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
@pytest.mark.skip(reason="File search requires API key auth, subscription only allows token auth")
async def test_azure_responses_client_file_search() -> None:
    """ファイル検索ツールを使った Azure responses client をテストする。"""
    azure_responses_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    assert isinstance(azure_responses_client, ChatClientProtocol)

    file_id, vector_store = await create_vector_store(azure_responses_client)
    # クライアントがウェブ検索ツールを使うことをテストする
    response = await azure_responses_client.get_response(
        messages=[
            ChatMessage(
                role="user",
                text="What is the weather today? Do a file search to find the answer.",
            )
        ],
        tools=[HostedFileSearchTool(inputs=vector_store)],
        tool_choice="auto",
    )

    await delete_vector_store(azure_responses_client, file_id, vector_store.vector_store_id)
    assert "sunny" in response.text.lower()
    assert "75" in response.text


@pytest.mark.flaky
@skip_if_azure_integration_tests_disabled
@pytest.mark.skip(reason="File search requires API key auth, subscription only allows token auth")
async def test_azure_responses_client_file_search_streaming() -> None:
    """ファイル検索ツールとストリーミングを使った Azure responses client をテストする。"""
    azure_responses_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    assert isinstance(azure_responses_client, ChatClientProtocol)

    file_id, vector_store = await create_vector_store(azure_responses_client)
    # クライアントがウェブ検索ツールを使うことをテストする
    response = azure_responses_client.get_streaming_response(
        messages=[
            ChatMessage(
                role="user",
                text="What is the weather today? Do a file search to find the answer.",
            )
        ],
        tools=[HostedFileSearchTool(inputs=vector_store)],
        tool_choice="auto",
    )

    assert response is not None
    full_message: str = ""
    async for chunk in response:
        assert chunk is not None
        assert isinstance(chunk, ChatResponseUpdate)
        for content in chunk.contents:
            if isinstance(content, TextContent) and content.text:
                full_message += content.text

    await delete_vector_store(azure_responses_client, file_id, vector_store.vector_store_id)

    assert "sunny" in full_message.lower()
    assert "75" in full_message
