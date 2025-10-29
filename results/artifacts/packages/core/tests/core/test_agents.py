# Copyright (c) Microsoft. All rights reserved.

import contextlib
from collections.abc import AsyncIterable, MutableSequence, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from pytest import raises

from agent_framework import (
    AgentProtocol,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    AggregateContextProvider,
    ChatAgent,
    ChatClientProtocol,
    ChatMessage,
    ChatMessageStore,
    ChatResponse,
    Context,
    ContextProvider,
    HostedCodeInterpreterTool,
    Role,
    TextContent,
)
from agent_framework._mcp import MCPTool
from agent_framework.exceptions import AgentExecutionException


def test_agent_thread_type(agent_thread: AgentThread) -> None:
    assert isinstance(agent_thread, AgentThread)


def test_agent_type(agent: AgentProtocol) -> None:
    assert isinstance(agent, AgentProtocol)


async def test_agent_run(agent: AgentProtocol) -> None:
    response = await agent.run("test")
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[0].text == "Response"


async def test_agent_run_streaming(agent: AgentProtocol) -> None:
    async def collect_updates(updates: AsyncIterable[AgentRunResponseUpdate]) -> list[AgentRunResponseUpdate]:
        return [u async for u in updates]

    updates = await collect_updates(agent.run_stream(messages="test"))
    assert len(updates) == 1
    assert updates[0].text == "Response"


def test_chat_client_agent_type(chat_client: ChatClientProtocol) -> None:
    chat_client_agent = ChatAgent(chat_client=chat_client)
    assert isinstance(chat_client_agent, AgentProtocol)


async def test_chat_client_agent_init(chat_client: ChatClientProtocol) -> None:
    agent_id = str(uuid4())
    agent = ChatAgent(chat_client=chat_client, id=agent_id, description="Test")

    assert agent.id == agent_id
    assert agent.name is None
    assert agent.description == "Test"
    assert agent.display_name == agent_id  # name が None の場合、表示名は id をデフォルトとする


async def test_chat_client_agent_init_with_name(chat_client: ChatClientProtocol) -> None:
    agent_id = str(uuid4())
    agent = ChatAgent(chat_client=chat_client, id=agent_id, name="Test Agent", description="Test")

    assert agent.id == agent_id
    assert agent.name == "Test Agent"
    assert agent.description == "Test"
    assert agent.display_name == "Test Agent"  # 表示名は name があればそれを使う


async def test_chat_client_agent_run(chat_client: ChatClientProtocol) -> None:
    agent = ChatAgent(chat_client=chat_client)

    result = await agent.run("Hello")

    assert result.text == "test response"


async def test_chat_client_agent_run_streaming(chat_client: ChatClientProtocol) -> None:
    agent = ChatAgent(chat_client=chat_client)

    result = await AgentRunResponse.from_agent_response_generator(agent.run_stream("Hello"))

    assert result.text == "test streaming response another update"


async def test_chat_client_agent_get_new_thread(chat_client: ChatClientProtocol) -> None:
    agent = ChatAgent(chat_client=chat_client)
    thread = agent.get_new_thread()

    assert isinstance(thread, AgentThread)


async def test_chat_client_agent_prepare_thread_and_messages(chat_client: ChatClientProtocol) -> None:
    agent = ChatAgent(chat_client=chat_client)
    message = ChatMessage(role=Role.USER, text="Hello")
    thread = AgentThread(message_store=ChatMessageStore(messages=[message]))

    _, _, result_messages = await agent._prepare_thread_and_messages(  # type: ignore[reportPrivateUsage]
        thread=thread,
        input_messages=[ChatMessage(role=Role.USER, text="Test")],
    )

    assert len(result_messages) == 2
    assert result_messages[0] == message
    assert result_messages[1].text == "Test"


async def test_chat_client_agent_update_thread_id(chat_client_base: ChatClientProtocol) -> None:
    mock_response = ChatResponse(
        messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent("test response")])],
        conversation_id="123",
    )
    chat_client_base.run_responses = [mock_response]
    agent = ChatAgent(
        chat_client=chat_client_base,
        tools=HostedCodeInterpreterTool(),
    )
    thread = agent.get_new_thread()

    result = await agent.run("Hello", thread=thread)
    assert result.text == "test response"

    assert thread.service_thread_id == "123"


async def test_chat_client_agent_update_thread_messages(chat_client: ChatClientProtocol) -> None:
    agent = ChatAgent(chat_client=chat_client)
    thread = agent.get_new_thread()

    result = await agent.run("Hello", thread=thread)
    assert result.text == "test response"

    assert thread.service_thread_id is None
    assert thread.message_store is not None

    chat_messages: list[ChatMessage] = await thread.message_store.list_messages()

    assert chat_messages is not None
    assert len(chat_messages) == 2
    assert chat_messages[0].text == "Hello"
    assert chat_messages[1].text == "test response"


async def test_chat_client_agent_update_thread_conversation_id_missing(chat_client: ChatClientProtocol) -> None:
    agent = ChatAgent(chat_client=chat_client)
    thread = AgentThread(service_thread_id="123")

    with raises(AgentExecutionException, match="Service did not return a valid conversation id"):
        await agent._update_thread_with_type_and_conversation_id(thread, None)  # type: ignore[reportPrivateUsage]


async def test_chat_client_agent_default_author_name(chat_client: ChatClientProtocol) -> None:
    # ここでは name が指定されていないのでデフォルト名を使うべき
    agent = ChatAgent(chat_client=chat_client)

    result = await agent.run("Hello")
    assert result.text == "test response"
    assert result.messages[0].author_name == "UnnamedAgent"


async def test_chat_client_agent_author_name_as_agent_name(chat_client: ChatClientProtocol) -> None:
    # ここでは name が指定されているので著者名として使うべき
    agent = ChatAgent(chat_client=chat_client, name="TestAgent")

    result = await agent.run("Hello")
    assert result.text == "test response"
    assert result.messages[0].author_name == "TestAgent"


async def test_chat_client_agent_author_name_is_used_from_response(chat_client_base: ChatClientProtocol) -> None:
    chat_client_base.run_responses = [
        ChatResponse(
            messages=[
                ChatMessage(role=Role.ASSISTANT, contents=[TextContent("test response")], author_name="TestAuthor")
            ]
        )
    ]

    agent = ChatAgent(chat_client=chat_client_base, tools=HostedCodeInterpreterTool())

    result = await agent.run("Hello")
    assert result.text == "test response"
    assert result.messages[0].author_name == "TestAuthor"


# テスト用のモックコンテキストプロバイダー
class MockContextProvider(ContextProvider):
    def __init__(self, messages: list[ChatMessage] | None = None) -> None:
        self.context_messages = messages
        self.thread_created_called = False
        self.invoked_called = False
        self.invoking_called = False
        self.thread_created_thread_id = None
        self.invoked_thread_id = None
        self.new_messages: list[ChatMessage] = []

    async def thread_created(self, thread_id: str | None) -> None:
        self.thread_created_called = True
        self.thread_created_thread_id = thread_id

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Any = None,
        **kwargs: Any,
    ) -> None:
        self.invoked_called = True
        if isinstance(request_messages, ChatMessage):
            self.new_messages.append(request_messages)
        else:
            self.new_messages.extend(request_messages)
        if isinstance(response_messages, ChatMessage):
            self.new_messages.append(response_messages)
        else:
            self.new_messages.extend(response_messages)

    async def invoking(self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any) -> Context:
        self.invoking_called = True
        return Context(messages=self.context_messages)


async def test_chat_agent_context_providers_model_invoking(chat_client: ChatClientProtocol) -> None:
    """Agent 実行中にコンテキストプロバイダーの invoking が呼ばれることをテストする。"""
    mock_provider = MockContextProvider(messages=[ChatMessage(role=Role.SYSTEM, text="Test context instructions")])
    agent = ChatAgent(chat_client=chat_client, context_providers=mock_provider)

    await agent.run("Hello")

    assert mock_provider.invoking_called


async def test_chat_agent_context_providers_thread_created(chat_client_base: ChatClientProtocol) -> None:
    """Agent 実行中にコンテキストプロバイダーの thread_created が呼ばれることをテストする。"""
    mock_provider = MockContextProvider()
    chat_client_base.run_responses = [
        ChatResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent("test response")])],
            conversation_id="test-thread-id",
        )
    ]

    agent = ChatAgent(chat_client=chat_client_base, context_providers=mock_provider)

    await agent.run("Hello")

    assert mock_provider.thread_created_called
    assert mock_provider.thread_created_thread_id == "test-thread-id"


async def test_chat_agent_context_providers_messages_adding(chat_client: ChatClientProtocol) -> None:
    """Agent 実行中にコンテキストプロバイダーの invoked が呼ばれることをテストする。"""
    mock_provider = MockContextProvider()
    agent = ChatAgent(chat_client=chat_client, context_providers=mock_provider)

    await agent.run("Hello")

    assert mock_provider.invoked_called
    # 入力メッセージとレスポンスメッセージの両方で呼ばれるべき
    assert len(mock_provider.new_messages) >= 2


async def test_chat_agent_context_instructions_in_messages(chat_client: ChatClientProtocol) -> None:
    """AI コンテキストの指示がメッセージに含まれることをテストする。"""
    mock_provider = MockContextProvider(messages=[ChatMessage(role="system", text="Context-specific instructions")])
    agent = ChatAgent(chat_client=chat_client, instructions="Agent instructions", context_providers=mock_provider)

    # _prepare_thread_and_messages メソッドを直接テストする必要がある
    _, _, messages = await agent._prepare_thread_and_messages(  # type: ignore[reportPrivateUsage]
        thread=None, input_messages=[ChatMessage(role=Role.USER, text="Hello")]
    )

    # コンテキスト指示とユーザーメッセージを持つべき
    assert len(messages) == 2
    assert messages[0].role == Role.SYSTEM
    assert messages[0].text == "Context-specific instructions"
    assert messages[1].role == Role.USER
    assert messages[1].text == "Hello"
    # instructions システムメッセージは chat_client によって追加される


async def test_chat_agent_no_context_instructions(chat_client: ChatClientProtocol) -> None:
    """AI コンテキストに指示がない場合の動作をテストする。"""
    mock_provider = MockContextProvider()
    agent = ChatAgent(chat_client=chat_client, instructions="Agent instructions", context_providers=mock_provider)

    _, _, messages = await agent._prepare_thread_and_messages(  # type: ignore[reportPrivateUsage]
        thread=None, input_messages=[ChatMessage(role=Role.USER, text="Hello")]
    )

    # エージェント指示とユーザーメッセージのみを持つべき
    assert len(messages) == 1
    assert messages[0].role == Role.USER
    assert messages[0].text == "Hello"


async def test_chat_agent_run_stream_context_providers(chat_client: ChatClientProtocol) -> None:
    """run_stream メソッドでコンテキストプロバイダーが動作することをテストする。"""
    mock_provider = MockContextProvider(messages=[ChatMessage(role=Role.SYSTEM, text="Stream context instructions")])
    agent = ChatAgent(chat_client=chat_client, context_providers=mock_provider)

    # すべてのストリーム更新を収集する
    updates: list[AgentRunResponseUpdate] = []
    async for update in agent.run_stream("Hello"):
        updates.append(update)

    # コンテキストプロバイダーが呼ばれたことを検証する
    assert mock_provider.invoking_called
    # 会話IDが作成されないため、thread_createを呼び出す必要はありません。
    assert not mock_provider.thread_created_called
    assert mock_provider.invoked_called


async def test_chat_agent_multiple_context_providers(chat_client: ChatClientProtocol) -> None:
    """複数のcontext providersが一緒に動作することをテストします。"""
    provider1 = MockContextProvider(messages=[ChatMessage(role=Role.SYSTEM, text="First provider instructions")])
    provider2 = MockContextProvider(messages=[ChatMessage(role=Role.SYSTEM, text="Second provider instructions")])

    agent = ChatAgent(chat_client=chat_client, context_providers=[provider1, provider2])

    await agent.run("Hello")

    # 両方のprovidersが呼び出されるべきです。
    assert provider1.invoking_called
    assert not provider1.thread_created_called
    assert provider1.invoked_called

    assert provider2.invoking_called
    assert not provider2.thread_created_called
    assert provider2.invoked_called


async def test_chat_agent_aggregate_context_provider_combines_instructions() -> None:
    """AggregateContextProviderが複数のprovidersからの指示を結合することをテストします。"""
    provider1 = MockContextProvider(messages=[ChatMessage(role=Role.SYSTEM, text="First instruction")])
    provider2 = MockContextProvider(messages=[ChatMessage(role=Role.SYSTEM, text="Second instruction")])

    aggregate = AggregateContextProvider()
    aggregate.providers.append(provider1)
    aggregate.providers.append(provider2)

    # invokingが指示を結合することをテストします。
    result = await aggregate.invoking([ChatMessage(role=Role.USER, text="Test")])

    assert result.messages
    assert isinstance(result.messages[0], ChatMessage)
    assert isinstance(result.messages[1], ChatMessage)
    assert result.messages[0].text == "First instruction"
    assert result.messages[1].text == "Second instruction"


async def test_chat_agent_context_providers_with_thread_service_id(chat_client_base: ChatClientProtocol) -> None:
    """サービス管理のthreadを持つcontext providersをテストします。"""
    mock_provider = MockContextProvider()
    chat_client_base.run_responses = [
        ChatResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent("test response")])],
            conversation_id="service-thread-123",
        )
    ]

    agent = ChatAgent(chat_client=chat_client_base, context_providers=mock_provider)

    # 既存のサービス管理のthreadを使用します。
    thread = agent.get_new_thread(service_thread_id="existing-thread-id")
    await agent.run("Hello", thread=thread)

    # invokedはresponseからのサービスthread IDで呼び出されるべきです。
    assert mock_provider.invoked_called


# as_toolメソッドのテストです。
async def test_chat_agent_as_tool_basic(chat_client: ChatClientProtocol) -> None:
    """基本的なas_toolの機能をテストします。"""
    agent = ChatAgent(chat_client=chat_client, name="TestAgent", description="Test agent for as_tool")

    tool = agent.as_tool()

    assert tool.name == "TestAgent"
    assert tool.description == "Test agent for as_tool"
    assert hasattr(tool, "func")
    assert hasattr(tool, "input_model")


async def test_chat_agent_as_tool_custom_parameters(chat_client: ChatClientProtocol) -> None:
    """カスタムパラメータを使ったas_toolのテストです。"""
    agent = ChatAgent(chat_client=chat_client, name="TestAgent", description="Original description")

    tool = agent.as_tool(
        name="CustomTool",
        description="Custom description",
        arg_name="query",
        arg_description="Custom input description",
    )

    assert tool.name == "CustomTool"
    assert tool.description == "Custom description"

    # 入力モデルにカスタムフィールド名があることを確認します。
    schema = tool.input_model.model_json_schema()
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["description"] == "Custom input description"


async def test_chat_agent_as_tool_defaults(chat_client: ChatClientProtocol) -> None:
    """デフォルトパラメータを使ったas_toolのテストです。"""
    agent = ChatAgent(
        chat_client=chat_client,
        name="TestAgent",
        # 説明は提供されていません。
    )

    tool = agent.as_tool()

    assert tool.name == "TestAgent"
    assert tool.description == ""  # 空文字列がデフォルトになるべきです。

    # デフォルトの入力フィールドを確認します。
    schema = tool.input_model.model_json_schema()
    assert "task" in schema["properties"]
    assert "Task for TestAgent" in schema["properties"]["task"]["description"]


async def test_chat_agent_as_tool_no_name(chat_client: ChatClientProtocol) -> None:
    """agentに名前がない場合のas_toolをテストします（ValueErrorを発生させるべきです）。"""
    agent = ChatAgent(chat_client=chat_client)  # 名前が提供されていません。

    # agentに名前がないためValueErrorを発生させるべきです。
    with raises(ValueError, match="Agent tool name cannot be None"):
        agent.as_tool()


async def test_chat_agent_as_tool_function_execution(chat_client: ChatClientProtocol) -> None:
    """生成されたAIFunctionが実行可能であることをテストします。"""
    agent = ChatAgent(chat_client=chat_client, name="TestAgent", description="Test agent")

    tool = agent.as_tool()

    # 関数の実行をテストします。
    result = await tool.invoke(arguments=tool.input_model(task="Hello"))

    # agentのresponseテキストを返すべきです。
    assert isinstance(result, str)
    assert result == "test response"  # モックのchat clientから。


async def test_chat_agent_as_tool_with_stream_callback(chat_client: ChatClientProtocol) -> None:
    """stream callback機能を使ったas_toolのテストです。"""
    agent = ChatAgent(chat_client=chat_client, name="StreamingAgent")

    # ストリーミングの更新を収集します。
    collected_updates: list[AgentRunResponseUpdate] = []

    def stream_callback(update: AgentRunResponseUpdate) -> None:
        collected_updates.append(update)

    tool = agent.as_tool(stream_callback=stream_callback)

    # ツールを実行します。
    result = await tool.invoke(arguments=tool.input_model(task="Hello"))

    # ストリーミングの更新が収集されているべきです。
    assert len(collected_updates) > 0
    assert isinstance(result, str)
    # 結果はすべてのストリーミング更新の連結であるべきです。
    expected_text = "".join(update.text for update in collected_updates)
    assert result == expected_text


async def test_chat_agent_as_tool_with_custom_arg_name(chat_client: ChatClientProtocol) -> None:
    """カスタム引数名を使ったas_toolのテストです。"""
    agent = ChatAgent(chat_client=chat_client, name="CustomArgAgent")

    tool = agent.as_tool(arg_name="prompt", arg_description="Custom prompt input")

    # カスタム引数名が機能することをテストします。
    result = await tool.invoke(arguments=tool.input_model(prompt="Test prompt"))
    assert result == "test response"


async def test_chat_agent_as_tool_with_async_stream_callback(chat_client: ChatClientProtocol) -> None:
    """非同期ストリームcallback機能を使ったas_toolのテストです。"""
    agent = ChatAgent(chat_client=chat_client, name="AsyncStreamingAgent")

    # 非同期callbackを使ってストリーミングの更新を収集します。
    collected_updates: list[AgentRunResponseUpdate] = []

    async def async_stream_callback(update: AgentRunResponseUpdate) -> None:
        collected_updates.append(update)

    tool = agent.as_tool(stream_callback=async_stream_callback)

    # ツールを実行します。
    result = await tool.invoke(arguments=tool.input_model(task="Hello"))

    # ストリーミングの更新が収集されているべきです。
    assert len(collected_updates) > 0
    assert isinstance(result, str)
    # 結果はすべてのストリーミング更新の連結であるべきです。
    expected_text = "".join(update.text for update in collected_updates)
    assert result == expected_text


async def test_chat_agent_as_tool_name_sanitization(chat_client: ChatClientProtocol) -> None:
    """as_toolの名前のサニタイズをテストします。"""
    test_cases = [
        ("Invoice & Billing Agent", "Invoice_Billing_Agent"),
        ("Travel & Logistics Agent", "Travel_Logistics_Agent"),
        ("Agent@Company.com", "Agent_Company_com"),
        ("Agent___Multiple___Underscores", "Agent_Multiple_Underscores"),
        ("123Agent", "_123Agent"),  # Test digit prefix handling
        ("9to5Helper", "_9to5Helper"),  # Another digit prefix case
        ("@@@", "agent"),  # Test empty sanitization fallback
    ]

    for agent_name, expected_tool_name in test_cases:
        agent = ChatAgent(chat_client=chat_client, name=agent_name, description="Test agent")
        tool = agent.as_tool()
        assert tool.name == expected_tool_name, f"Expected {expected_tool_name}, got {tool.name} for input {agent_name}"


async def test_chat_agent_as_mcp_server_basic(chat_client: ChatClientProtocol) -> None:
    """基本的なas_mcp_serverの機能をテストします。"""
    agent = ChatAgent(chat_client=chat_client, name="TestAgent", description="Test agent for MCP")

    # デフォルトパラメータでMCPサーバーを作成します。
    server = agent.as_mcp_server()

    # サーバーが作成されたことを検証します。
    assert server is not None
    assert hasattr(server, "name")
    assert hasattr(server, "version")


async def test_chat_agent_run_with_mcp_tools(chat_client: ChatClientProtocol) -> None:
    """MCPツールを使ったrunメソッドをテストし、MCPツール処理コードをカバーします。"""
    agent = ChatAgent(chat_client=chat_client, name="TestAgent", description="Test agent")

    # モックのMCPツールを作成します。
    mock_mcp_tool = MagicMock(spec=MCPTool)
    mock_mcp_tool.is_connected = False
    mock_mcp_tool.functions = [MagicMock()]

    # 非同期コンテキストマネージャのエントリをモックします。
    mock_mcp_tool.__aenter__ = AsyncMock(return_value=mock_mcp_tool)
    mock_mcp_tool.__aexit__ = AsyncMock(return_value=None)

    # MCPツールを使ったrunをテストします - これはMCPツール処理コードに到達するはずです。
    with contextlib.suppress(Exception):
        # モックを使っているため失敗することが予想されますが、コードパスを実行したいです。
        await agent.run(messages="Test message", tools=[mock_mcp_tool])


async def test_chat_agent_with_local_mcp_tools(chat_client: ChatClientProtocol) -> None:
    """ローカルMCPツールを使ったagentの初期化をテストします。"""
    # モックのMCPツールを作成します。
    mock_mcp_tool = MagicMock(spec=MCPTool)
    mock_mcp_tool.is_connected = False
    mock_mcp_tool.__aenter__ = AsyncMock(return_value=mock_mcp_tool)
    mock_mcp_tool.__aexit__ = AsyncMock(return_value=None)

    # コンストラクタでMCPツールを持つagentをテストします。
    with contextlib.suppress(Exception):
        agent = ChatAgent(chat_client=chat_client, name="TestAgent", description="Test agent", tools=[mock_mcp_tool])
        # MCPツールを使った非同期コンテキストマネージャをテストします。
        async with agent:
            pass
