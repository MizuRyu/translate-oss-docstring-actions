# Copyright (c) Microsoft. All rights reserved.

"""tau2 utilsモジュールのテストです。"""

import urllib.request
from pathlib import Path

import pytest
from agent_framework._tools import AIFunction
from agent_framework._types import ChatMessage, FunctionCallContent, FunctionResultContent, Role, TextContent
from agent_framework_lab_tau2._tau2_utils import (
    convert_agent_framework_messages_to_tau2_messages,
    convert_tau2_tool_to_ai_function,
)
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.environment.environment import Environment


@pytest.fixture(scope="session")
def tau2_airline_environment() -> Environment:
    airline_db_remote_path = "https://raw.githubusercontent.com/sierra-research/tau2-bench/5ba9e3e56db57c5e4114bf7f901291f09b2c5619/data/tau2/domains/airline/db.json"
    airline_policy_remote_path = "https://raw.githubusercontent.com/sierra-research/tau2-bench/5ba9e3e56db57c5e4114bf7f901291f09b2c5619/data/tau2/domains/airline/policy.md"

    # キャッシュディレクトリを作成する
    cache_dir = Path(__file__).parent / "data"
    cache_dir.mkdir(exist_ok=True)

    # キャッシュファイルのパスを定義する
    db_cache_path = cache_dir / "airline_db.json"
    policy_cache_path = cache_dir / "airline_policy.md"

    # キャッシュに存在しない場合のみファイルをダウンロードする
    if not db_cache_path.exists():
        urllib.request.urlretrieve(airline_db_remote_path, db_cache_path)

    if not policy_cache_path.exists():
        urllib.request.urlretrieve(airline_policy_remote_path, policy_cache_path)

    # キャッシュされたファイルからデータを読み込む
    db = FlightDB.load(str(db_cache_path))
    tools = AirlineTools(db)
    with open(policy_cache_path) as fp:
        policy = fp.read()

    yield Environment(
        domain_name="airline",
        policy=policy,
        tools=tools,
    )


def test_convert_tau2_tool_to_ai_function_basic(tau2_airline_environment):
    """tau2 toolからAIFunctionへの基本的な変換をテストします。"""
    # tau2環境から実際のツールを取得する
    tools = tau2_airline_environment.get_tools()

    # テストのために最初に利用可能なツールを使用する
    assert len(tools) > 0, "No tools available in environment"
    tau2_tool = tools[0]

    # ツールを変換する
    ai_function = convert_tau2_tool_to_ai_function(tau2_tool)

    # 変換を検証する
    assert isinstance(ai_function, AIFunction)
    assert ai_function.name == tau2_tool.name
    assert ai_function.description == tau2_tool._get_description()
    assert ai_function.input_model == tau2_tool.params

    # 関数が呼び出し可能であることをテストする（副作用を避けるため実際のパラメータで呼び出しません）
    assert callable(ai_function.func)


def test_convert_tau2_tool_to_ai_function_multiple_tools(tau2_airline_environment):
    """複数のtau2ツールでの変換をテストします。"""
    # tau2環境から実際のツールを取得する
    tools = tau2_airline_environment.get_tools()

    # 複数のツールを変換する
    ai_functions = [convert_tau2_tool_to_ai_function(tool) for tool in tools[:3]]  # 最初の3つのツールをテストする

    # すべての変換を検証する
    for ai_function, tau2_tool in zip(ai_functions, tools[:3], strict=False):
        assert isinstance(ai_function, AIFunction)
        assert ai_function.name == tau2_tool.name
        assert ai_function.description == tau2_tool._get_description()
        assert ai_function.input_model == tau2_tool.params
        assert callable(ai_function.func)


def test_convert_agent_framework_messages_to_tau2_messages_system():
    """systemメッセージの変換をテストする。"""
    messages = [ChatMessage(role=Role.SYSTEM, contents=[TextContent(text="System instruction")])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], SystemMessage)
    assert tau2_messages[0].role == "system"
    assert tau2_messages[0].content == "System instruction"


def test_convert_agent_framework_messages_to_tau2_messages_user():
    """userメッセージの変換をテストする。"""
    messages = [ChatMessage(role=Role.USER, contents=[TextContent(text="Hello assistant")])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], UserMessage)
    assert tau2_messages[0].role == "user"
    assert tau2_messages[0].content == "Hello assistant"
    assert tau2_messages[0].tool_calls is None


def test_convert_agent_framework_messages_to_tau2_messages_assistant():
    """assistantメッセージの変換をテストする。"""
    messages = [ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Hello user")])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], AssistantMessage)
    assert tau2_messages[0].role == "assistant"
    assert tau2_messages[0].content == "Hello user"
    assert tau2_messages[0].tool_calls is None


def test_convert_agent_framework_messages_to_tau2_messages_with_function_call():
    """function callを含むメッセージの変換をテストする。"""
    function_call = FunctionCallContent(call_id="call_123", name="test_function", arguments={"param": "value"})

    messages = [ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="I'll call a function"), function_call])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], AssistantMessage)
    assert tau2_messages[0].content == "I'll call a function"
    assert tau2_messages[0].tool_calls is not None
    assert len(tau2_messages[0].tool_calls) == 1

    tool_call = tau2_messages[0].tool_calls[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.id == "call_123"
    assert tool_call.name == "test_function"
    assert tool_call.arguments == {"param": "value"}
    assert tool_call.requestor == "assistant"


def test_convert_agent_framework_messages_to_tau2_messages_with_function_result():
    """function resultを含むメッセージの変換をテストする。"""
    function_result = FunctionResultContent(call_id="call_123", result={"success": True, "data": "result data"})

    messages = [ChatMessage(role=Role.TOOL, contents=[function_result])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], ToolMessage)
    assert tau2_messages[0].id == "call_123"
    assert tau2_messages[0].role == "tool"
    assert tau2_messages[0].content is not None
    assert '{"success": true, "data": "result data"}' in tau2_messages[0].content
    assert tau2_messages[0].requestor == "assistant"
    assert tau2_messages[0].error is False


def test_convert_agent_framework_messages_to_tau2_messages_with_error():
    """エラーを含むfunction resultの変換をテストする。"""
    function_result = FunctionResultContent(
        call_id="call_456", result="Error occurred", exception=Exception("Test error")
    )

    messages = [ChatMessage(role=Role.TOOL, contents=[function_result])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], ToolMessage)
    assert tau2_messages[0].error is True


def test_convert_agent_framework_messages_to_tau2_messages_multiple_text_contents():
    """複数のテキストコンテンツを含むメッセージの変換をテストする。"""
    messages = [ChatMessage(role=Role.USER, contents=[TextContent(text="First part"), TextContent(text="Second part")])]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 1
    assert isinstance(tau2_messages[0], UserMessage)
    assert tau2_messages[0].content == "First part Second part"


def test_convert_agent_framework_messages_to_tau2_messages_complex_scenario():
    """複数のメッセージタイプを含む複雑なシナリオの変換をテストする。"""
    function_call = FunctionCallContent(call_id="call_789", name="complex_tool", arguments='{"key": "value"}')

    function_result = FunctionResultContent(call_id="call_789", result={"output": "tool result"})

    messages = [
        ChatMessage(role=Role.SYSTEM, contents=[TextContent(text="System prompt")]),
        ChatMessage(role=Role.USER, contents=[TextContent(text="User request")]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="I'll help you"), function_call]),
        ChatMessage(role=Role.TOOL, contents=[function_result]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Based on the result...")]),
    ]

    tau2_messages = convert_agent_framework_messages_to_tau2_messages(messages)

    assert len(tau2_messages) == 5
    assert isinstance(tau2_messages[0], SystemMessage)
    assert isinstance(tau2_messages[1], UserMessage)
    assert isinstance(tau2_messages[2], AssistantMessage)
    assert isinstance(tau2_messages[3], ToolMessage)
    assert isinstance(tau2_messages[4], AssistantMessage)

    # tool callを含むassistantメッセージをチェックする
    assert tau2_messages[2].tool_calls is not None
    assert len(tau2_messages[2].tool_calls) == 1
    assert tau2_messages[2].tool_calls[0].name == "complex_tool"
