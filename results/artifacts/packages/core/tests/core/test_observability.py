# Copyright (c) Microsoft. All rights reserved.

import logging
from collections.abc import MutableSequence
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from agent_framework import (
    AGENT_FRAMEWORK_USER_AGENT,
    AgentProtocol,
    AgentRunResponse,
    AgentThread,
    BaseChatClient,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Role,
    UsageDetails,
    prepend_agent_framework_to_user_agent,
)
from agent_framework.exceptions import AgentInitializationError, ChatClientInitializationError
from agent_framework.observability import (
    OPEN_TELEMETRY_AGENT_MARKER,
    OPEN_TELEMETRY_CHAT_CLIENT_MARKER,
    ROLE_EVENT_MAP,
    ChatMessageListTimestampFilter,
    OtelAttr,
    get_function_span,
    use_agent_observability,
    use_observability,
)

# region Test constants


def test_role_event_map():
    """ROLE_EVENT_MAPに期待されるマッピングが含まれていることをテストする。"""
    assert ROLE_EVENT_MAP["system"] == OtelAttr.SYSTEM_MESSAGE
    assert ROLE_EVENT_MAP["user"] == OtelAttr.USER_MESSAGE
    assert ROLE_EVENT_MAP["assistant"] == OtelAttr.ASSISTANT_MESSAGE
    assert ROLE_EVENT_MAP["tool"] == OtelAttr.TOOL_MESSAGE


def test_enum_values():
    """OtelAttr列挙が期待される値を持つことをテストする。"""
    assert OtelAttr.OPERATION == "gen_ai.operation.name"
    assert SpanAttributes.LLM_SYSTEM == "gen_ai.system"
    assert SpanAttributes.LLM_REQUEST_MODEL == "gen_ai.request.model"
    assert OtelAttr.CHAT_COMPLETION_OPERATION == "chat"
    assert OtelAttr.TOOL_EXECUTION_OPERATION == "execute_tool"
    assert OtelAttr.AGENT_INVOKE_OPERATION == "invoke_agent"


# region Test ChatMessageListTimestampFilter


def test_filter_without_index_key():
    """レコードにINDEX_KEYがない場合のfilterメソッドをテストする。"""
    log_filter = ChatMessageListTimestampFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="test message", args=(), exc_info=None
    )
    original_created = record.created

    result = log_filter.filter(record)

    assert result is True
    assert record.created == original_created


def test_filter_with_index_key():
    """レコードにINDEX_KEYがある場合のfilterメソッドをテストする。"""
    log_filter = ChatMessageListTimestampFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="test message", args=(), exc_info=None
    )
    original_created = record.created

    # インデックスキーを追加する
    setattr(record, ChatMessageListTimestampFilter.INDEX_KEY, 5)

    result = log_filter.filter(record)

    assert result is True
    # 5マイクロ秒（5 * 1e-6）だけ増加するはず
    assert record.created == original_created + 5 * 1e-6


def test_index_key_constant():
    """INDEX_KEY定数が正しく定義されていることをテストする。"""
    assert ChatMessageListTimestampFilter.INDEX_KEY == "chat_message_index"


# region Test get_function_span


def test_start_span_basic(span_exporter: InMemorySpanExporter):
    """基本的な関数情報でスパンを開始するテスト。"""
    # モック関数を作成する
    mock_function = Mock()
    mock_function.name = "test_function"
    mock_function.description = "Test function description"
    attributes = {
        OtelAttr.OPERATION: OtelAttr.TOOL_EXECUTION_OPERATION,
        OtelAttr.TOOL_NAME: "test_function",
        OtelAttr.TOOL_DESCRIPTION: "Test function description",
        OtelAttr.TOOL_TYPE: "function",
    }
    span_exporter.clear()
    with get_function_span(attributes) as function_span:
        assert function_span is not None
        function_span.set_attribute("test_attr", "test_value")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool test_function"
    assert span.attributes["test_attr"] == "test_value"
    assert span.attributes[OtelAttr.OPERATION.value] == OtelAttr.TOOL_EXECUTION_OPERATION
    assert span.attributes[OtelAttr.TOOL_NAME] == "test_function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "Test function description"


def test_start_span_with_tool_call_id(span_exporter: InMemorySpanExporter):
    """tool_call_idを使ってスパンを開始するテスト。"""

    tool_call_id = "test_call_123"
    attributes = {
        OtelAttr.OPERATION: OtelAttr.TOOL_EXECUTION_OPERATION,
        OtelAttr.TOOL_NAME: "test_function",
        OtelAttr.TOOL_DESCRIPTION: "Test function",
        OtelAttr.TOOL_TYPE: "function",
        OtelAttr.TOOL_CALL_ID: tool_call_id,
    }

    span_exporter.clear()
    with get_function_span(attributes) as function_span:
        assert function_span is not None
        function_span.set_attribute("test_attr", "test_value")
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool test_function"
    assert span.attributes["test_attr"] == "test_value"
    assert span.attributes[OtelAttr.TOOL_CALL_ID] == tool_call_id
    # すべての属性を検証する
    assert span.attributes[OtelAttr.OPERATION.value] == OtelAttr.TOOL_EXECUTION_OPERATION
    assert span.attributes[OtelAttr.TOOL_NAME] == "test_function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "Test function"
    assert span.attributes[OtelAttr.TOOL_TYPE] == "function"


# region Test use_observability decorator


def test_decorator_with_valid_class():
    """デコレータが有効なBaseChatClientライクなクラスで動作することをテストする。"""

    # 必要なメソッドを持つモッククラスを作成する
    class MockChatClient:
        async def get_response(self, messages, **kwargs):
            return Mock()

        async def get_streaming_response(self, messages, **kwargs):
            async def gen():
                yield Mock()

            return gen()

    # デコレータを適用する
    decorated_class = use_observability(MockChatClient)
    assert hasattr(decorated_class, OPEN_TELEMETRY_CHAT_CLIENT_MARKER)


def test_decorator_with_missing_methods():
    """必要なメソッドが欠けているクラスでもデコレータが正常に処理することをテストする。"""

    class MockChatClient:
        OTEL_PROVIDER_NAME = "test_provider"

    # デコレータを適用する - エラーは発生しないはず
    with pytest.raises(ChatClientInitializationError):
        use_observability(MockChatClient)


def test_decorator_with_partial_methods():
    """メソッドが1つだけ存在する場合のデコレータのテスト。"""

    class MockChatClient:
        OTEL_PROVIDER_NAME = "test_provider"

        async def get_response(self, messages, **kwargs):
            return Mock()

    with pytest.raises(ChatClientInitializationError):
        use_observability(MockChatClient)


# region Test telemetry decorator with mock client


@pytest.fixture
def mock_chat_client():
    """テスト用のモックチャットクライアントを作成する。"""

    class MockChatClient(BaseChatClient):
        def service_url(self):
            return "https://test.example.com"

        async def _inner_get_response(
            self, *, messages: MutableSequence[ChatMessage], chat_options: ChatOptions, **kwargs: Any
        ):
            return ChatResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text="Test response")],
                usage_details=UsageDetails(input_token_count=10, output_token_count=20),
                finish_reason=None,
            )

        async def _inner_get_streaming_response(
            self, *, messages: MutableSequence[ChatMessage], chat_options: ChatOptions, **kwargs: Any
        ):
            yield ChatResponseUpdate(text="Hello", role=Role.ASSISTANT)
            yield ChatResponseUpdate(text=" world", role=Role.ASSISTANT)

    return MockChatClient


@pytest.mark.parametrize("enable_sensitive_data", [True, False], indirect=True)
async def test_chat_client_observability(mock_chat_client, span_exporter: InMemorySpanExporter, enable_sensitive_data):
    """診断が有効な場合にテレメトリが適用されることをテストする。"""
    client = use_observability(mock_chat_client)()

    messages = [ChatMessage(role=Role.USER, text="Test message")]
    span_exporter.clear()
    response = await client.get_response(messages=messages, model_id="Test")
    assert response is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat Test"
    assert span.attributes[OtelAttr.OPERATION.value] == OtelAttr.CHAT_COMPLETION_OPERATION
    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "Test"
    assert span.attributes[OtelAttr.INPUT_TOKENS] == 10
    assert span.attributes[OtelAttr.OUTPUT_TOKENS] == 20
    if enable_sensitive_data:
        assert span.attributes[OtelAttr.INPUT_MESSAGES] is not None
        assert span.attributes[OtelAttr.OUTPUT_MESSAGES] is not None


@pytest.mark.parametrize("enable_sensitive_data", [True, False], indirect=True)
async def test_chat_client_streaming_observability(
    mock_chat_client, span_exporter: InMemorySpanExporter, enable_sensitive_data
):
    """use_observabilityデコレータを通じたストリーミングテレメトリのテスト。"""
    client = use_observability(mock_chat_client)()
    messages = [ChatMessage(role=Role.USER, text="Test")]
    span_exporter.clear()
    # すべてのyieldされた更新を収集する
    updates = []
    async for update in client.get_streaming_response(messages=messages, model_id="Test"):
        updates.append(update)

    # 期待される更新を受け取ったことを検証する。これはotelに依存しないはず。
    assert len(updates) == 2
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat Test"
    assert span.attributes[OtelAttr.OPERATION.value] == OtelAttr.CHAT_COMPLETION_OPERATION
    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "Test"
    if enable_sensitive_data:
        assert span.attributes[OtelAttr.INPUT_MESSAGES] is not None
        assert span.attributes[OtelAttr.OUTPUT_MESSAGES] is not None


def test_prepend_user_agent_with_none_value():
    """ヘッダーにNone値がある場合のユーザーエージェントのプレフィックステスト。"""
    headers = {"User-Agent": None}
    result = prepend_agent_framework_to_user_agent(headers)

    # Noneを正常に処理するはずである
    assert "User-Agent" in result
    assert AGENT_FRAMEWORK_USER_AGENT in str(result["User-Agent"])


# region Test use_agent_observability decorator


def test_agent_decorator_with_valid_class():
    """有効なChatAgentライクなクラスでagentデコレータが動作することをテストする。"""

    # 必要なメソッドを持つモッククラスを作成する
    class MockChatClientAgent:
        AGENT_SYSTEM_NAME = "test_agent_system"

        def __init__(self):
            self.id = "test_agent_id"
            self.name = "test_agent"
            self.display_name = "Test Agent"
            self.description = "Test agent description"

        async def run(self, messages=None, *, thread=None, **kwargs):
            return Mock()

        async def run_stream(self, messages=None, *, thread=None, **kwargs):
            async def gen():
                yield Mock()

            return gen()

        def get_new_thread(self) -> AgentThread:
            return AgentThread()

    # デコレータを適用する
    decorated_class = use_agent_observability(MockChatClientAgent)

    assert hasattr(decorated_class, OPEN_TELEMETRY_AGENT_MARKER)


def test_agent_decorator_with_missing_methods():
    """必要なメソッドが欠けているクラスでもagentデコレータが正常に処理することをテストする。"""

    class MockAgent:
        AGENT_SYSTEM_NAME = "test_agent_system"

    # デコレータを適用する - エラーは発生しないはず
    with pytest.raises(AgentInitializationError):
        use_agent_observability(MockAgent)


def test_agent_decorator_with_partial_methods():
    """メソッドが1つだけ存在する場合のagentデコレータのテスト。"""
    from agent_framework.observability import use_agent_observability

    class MockAgent:
        AGENT_SYSTEM_NAME = "test_agent_system"

        def __init__(self):
            self.id = "test_agent_id"
            self.name = "test_agent"
            self.display_name = "Test Agent"

        async def run(self, messages=None, *, thread=None, **kwargs):
            return Mock()

    with pytest.raises(AgentInitializationError):
        use_agent_observability(MockAgent)


# region Test agent telemetry decorator with mock agent


@pytest.fixture
def mock_chat_agent():
    """テスト用のモックチャットクライアントエージェントを作成する。"""

    class MockChatClientAgent:
        AGENT_SYSTEM_NAME = "test_agent_system"

        def __init__(self):
            self.id = "test_agent_id"
            self.name = "test_agent"
            self.display_name = "Test Agent"
            self.description = "Test agent description"

        async def run(self, messages=None, *, thread=None, **kwargs):
            return AgentRunResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text="Agent response")],
                usage_details=UsageDetails(input_token_count=15, output_token_count=25),
                response_id="test_response_id",
                raw_representation=Mock(finish_reason=Mock(value="stop")),
            )

        async def run_stream(self, messages=None, *, thread=None, **kwargs):
            from agent_framework import AgentRunResponseUpdate

            yield AgentRunResponseUpdate(text="Hello", role=Role.ASSISTANT)
            yield AgentRunResponseUpdate(text=" from agent", role=Role.ASSISTANT)

    return MockChatClientAgent


@pytest.mark.parametrize("enable_sensitive_data", [True, False], indirect=True)
async def test_agent_instrumentation_enabled(
    mock_chat_agent: AgentProtocol, span_exporter: InMemorySpanExporter, enable_sensitive_data
):
    """エージェント診断が有効な場合にテレメトリが適用されることをテストする。"""

    agent = use_agent_observability(mock_chat_agent)()

    span_exporter.clear()
    response = await agent.run("Test message")
    assert response is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent Test Agent"
    assert span.attributes[OtelAttr.OPERATION.value] == OtelAttr.AGENT_INVOKE_OPERATION
    assert span.attributes[OtelAttr.AGENT_ID] == "test_agent_id"
    assert span.attributes[OtelAttr.AGENT_NAME] == "Test Agent"
    assert span.attributes[OtelAttr.AGENT_DESCRIPTION] == "Test agent description"
    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "unknown"
    assert span.attributes[OtelAttr.INPUT_TOKENS] == 15
    assert span.attributes[OtelAttr.OUTPUT_TOKENS] == 25
    if enable_sensitive_data:
        assert span.attributes[OtelAttr.OUTPUT_MESSAGES] is not None


@pytest.mark.parametrize("enable_sensitive_data", [True, False], indirect=True)
async def test_agent_streaming_response_with_diagnostics_enabled_via_decorator(
    mock_chat_agent: AgentProtocol, span_exporter: InMemorySpanExporter, enable_sensitive_data
):
    """use_agent_observabilityデコレータを通じたエージェントのストリーミングテレメトリのテスト。"""
    agent = use_agent_observability(mock_chat_agent)()
    span_exporter.clear()
    updates = []
    async for update in agent.run_stream("Test message"):
        updates.append(update)

    # 期待される更新を受け取ったことを検証する
    assert len(updates) == 2
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent Test Agent"
    assert span.attributes[OtelAttr.OPERATION.value] == OtelAttr.AGENT_INVOKE_OPERATION
    assert span.attributes[OtelAttr.AGENT_ID] == "test_agent_id"
    assert span.attributes[OtelAttr.AGENT_NAME] == "Test Agent"
    assert span.attributes[OtelAttr.AGENT_DESCRIPTION] == "Test agent description"
    assert span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "unknown"
    if enable_sensitive_data:
        assert span.attributes.get(OtelAttr.OUTPUT_MESSAGES) is not None  # ストリーミングなのでまだ使用はなし


async def test_agent_run_with_exception_handling(mock_chat_agent: AgentProtocol):
    """例外処理付きのエージェント実行をテストする。"""

    async def run_with_error(self, messages=None, *, thread=None, **kwargs):
        raise RuntimeError("Agent run error")

    mock_chat_agent.run = run_with_error

    agent = use_agent_observability(mock_chat_agent)()

    from opentelemetry.trace import Span

    with (
        patch("agent_framework.observability._get_span") as mock_get_span,
    ):
        mock_span = MagicMock(spec=Span)
        # パッチされたコンテキストマネージャがenter時にmock_spanを返すことを保証する
        mock_get_span.return_value.__enter__.return_value = mock_span
        # 例外を発生させエラーハンドラを呼び出すはずである
        with pytest.raises(RuntimeError, match="Agent run error"):
            await agent.run("Test message")

        # エラーが記録されたことを検証する スパンに両方のエラー属性が設定されたことを確認する
        mock_span.set_attribute.assert_called_with(OtelAttr.ERROR_TYPE, "RuntimeError")
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once_with(
            status=StatusCode.ERROR, description=repr(RuntimeError("Agent run error"))
        )
