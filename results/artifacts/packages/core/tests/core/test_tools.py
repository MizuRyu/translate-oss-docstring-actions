# Copyright (c) Microsoft. All rights reserved.
from typing import Any
from unittest.mock import Mock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from agent_framework import (
    AIFunction,
    HostedCodeInterpreterTool,
    HostedMCPTool,
    ToolProtocol,
    ai_function,
)
from agent_framework._tools import _parse_inputs
from agent_framework.exceptions import ToolException
from agent_framework.observability import OtelAttr

# region AIFunction and ai_function decorator tests


def test_ai_function_decorator():
    """ai_function デコレーターをテストします。"""

    @ai_function(name="test_tool", description="A test tool")
    def test_tool(x: int, y: int) -> int:
        """2つの数字を加算する単純な関数。"""
        return x + y

    assert isinstance(test_tool, ToolProtocol)
    assert isinstance(test_tool, AIFunction)
    assert test_tool.name == "test_tool"
    assert test_tool.description == "A test tool"
    assert test_tool.parameters() == {
        "properties": {"x": {"title": "X", "type": "integer"}, "y": {"title": "Y", "type": "integer"}},
        "required": ["x", "y"],
        "title": "test_tool_input",
        "type": "object",
    }
    assert test_tool(1, 2) == 3


def test_ai_function_decorator_without_args():
    """ai_function デコレーターをテストします。"""

    @ai_function
    def test_tool(x: int, y: int) -> int:
        """2つの数字を加算する単純な関数。"""
        return x + y

    assert isinstance(test_tool, ToolProtocol)
    assert isinstance(test_tool, AIFunction)
    assert test_tool.name == "test_tool"
    assert test_tool.description == "A simple function that adds two numbers."
    assert test_tool.parameters() == {
        "properties": {"x": {"title": "X", "type": "integer"}, "y": {"title": "Y", "type": "integer"}},
        "required": ["x", "y"],
        "title": "test_tool_input",
        "type": "object",
    }
    assert test_tool(1, 2) == 3


async def test_ai_function_decorator_with_async():
    """非同期関数に対する ai_function デコレーターのテスト。"""

    @ai_function(name="async_test_tool", description="An async test tool")
    async def async_test_tool(x: int, y: int) -> int:
        """2つの数字を加算する非同期関数。"""
        return x + y

    assert isinstance(async_test_tool, ToolProtocol)
    assert isinstance(async_test_tool, AIFunction)
    assert async_test_tool.name == "async_test_tool"
    assert async_test_tool.description == "An async test tool"
    assert async_test_tool.parameters() == {
        "properties": {"x": {"title": "X", "type": "integer"}, "y": {"title": "Y", "type": "integer"}},
        "required": ["x", "y"],
        "title": "async_test_tool_input",
        "type": "object",
    }
    assert (await async_test_tool(1, 2)) == 3


async def test_ai_function_invoke_telemetry_enabled(span_exporter: InMemorySpanExporter):
    """テレメトリ有効時の ai_function invoke メソッドをテストします。"""

    @ai_function(
        name="telemetry_test_tool",
        description="A test tool for telemetry",
    )
    def telemetry_test_tool(x: int, y: int) -> int:
        """テレメトリテスト用の2つの数字を加算する関数。"""
        return x + y

    # ヒストグラムのモック。
    mock_histogram = Mock()
    telemetry_test_tool._invocation_duration_histogram = mock_histogram
    span_exporter.clear()
    # invoke を呼び出す。
    result = await telemetry_test_tool.invoke(x=1, y=2, tool_call_id="test_call_id")

    # 結果を検証する。
    assert result == 3

    # テレメトリ呼び出しを検証する。
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert OtelAttr.TOOL_EXECUTION_OPERATION.value in span.name
    assert "telemetry_test_tool" in span.name
    assert span.attributes[OtelAttr.TOOL_NAME] == "telemetry_test_tool"
    assert span.attributes[OtelAttr.TOOL_CALL_ID] == "test_call_id"
    assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "A test tool for telemetry"
    assert span.attributes[OtelAttr.TOOL_ARGUMENTS] == '{"x": 1, "y": 2}'
    assert span.attributes[OtelAttr.TOOL_RESULT] == "3"

    # ヒストグラムが正しい属性で呼ばれたことを検証する。
    mock_histogram.record.assert_called_once()
    call_args = mock_histogram.record.call_args
    assert call_args[0][0] > 0  # duration は正の値であるべきです。
    attributes = call_args[1]["attributes"]
    assert attributes[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] == "telemetry_test_tool"
    assert attributes[OtelAttr.TOOL_CALL_ID] == "test_call_id"


@pytest.mark.parametrize("enable_sensitive_data", [False], indirect=True)
async def test_ai_function_invoke_telemetry_sensitive_disabled(span_exporter: InMemorySpanExporter):
    """テレメトリ有効時の ai_function invoke メソッドをテストします。"""

    @ai_function(
        name="telemetry_test_tool",
        description="A test tool for telemetry",
    )
    def telemetry_test_tool(x: int, y: int) -> int:
        """テレメトリテスト用の2つの数字を加算する関数。"""
        return x + y

    # ヒストグラムのモック。
    mock_histogram = Mock()
    telemetry_test_tool._invocation_duration_histogram = mock_histogram
    span_exporter.clear()
    # invoke を呼び出す。
    result = await telemetry_test_tool.invoke(x=1, y=2, tool_call_id="test_call_id")

    # 結果を検証する。
    assert result == 3

    # テレメトリ呼び出しを検証する。
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert OtelAttr.TOOL_EXECUTION_OPERATION.value in span.name
    assert "telemetry_test_tool" in span.name
    assert span.attributes[OtelAttr.TOOL_NAME] == "telemetry_test_tool"
    assert span.attributes[OtelAttr.TOOL_CALL_ID] == "test_call_id"
    assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "A test tool for telemetry"
    assert OtelAttr.TOOL_ARGUMENTS not in span.attributes
    assert OtelAttr.TOOL_RESULT not in span.attributes

    # ヒストグラムが正しい属性で呼ばれたことを検証する。
    mock_histogram.record.assert_called_once()
    call_args = mock_histogram.record.call_args
    assert call_args[0][0] > 0  # duration は正の値であるべきです。
    attributes = call_args[1]["attributes"]
    assert attributes[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] == "telemetry_test_tool"
    assert attributes[OtelAttr.TOOL_CALL_ID] == "test_call_id"


async def test_ai_function_invoke_telemetry_with_pydantic_args(span_exporter: InMemorySpanExporter):
    """Pydantic モデル引数を用いた ai_function invoke メソッドをテストします。"""

    @ai_function(
        name="pydantic_test_tool",
        description="A test tool with Pydantic args",
    )
    def pydantic_test_tool(x: int, y: int) -> int:
        """Pydantic 引数を用いて2つの数字を加算する関数。"""
        return x + y

    # 引数を Pydantic モデルインスタンスとして作成する。
    args_model = pydantic_test_tool.input_model(x=5, y=10)

    mock_histogram = Mock()
    pydantic_test_tool._invocation_duration_histogram = mock_histogram
    span_exporter.clear()
    # Pydantic モデルで invoke を呼び出す。
    result = await pydantic_test_tool.invoke(arguments=args_model, tool_call_id="pydantic_call")

    # 結果を検証する。
    assert result == 15
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert OtelAttr.TOOL_EXECUTION_OPERATION.value in span.name
    assert "pydantic_test_tool" in span.name
    assert span.attributes[OtelAttr.TOOL_NAME] == "pydantic_test_tool"
    assert span.attributes[OtelAttr.TOOL_CALL_ID] == "pydantic_call"
    assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "A test tool with Pydantic args"
    assert span.attributes[OtelAttr.TOOL_ARGUMENTS] == '{"x":5,"y":10}'


async def test_ai_function_invoke_telemetry_with_exception(span_exporter: InMemorySpanExporter):
    """例外発生時のテレメトリを伴う ai_function invoke メソッドをテストします。"""

    @ai_function(
        name="exception_test_tool",
        description="A test tool that raises an exception",
    )
    def exception_test_tool(x: int, y: int) -> int:
        """テレメトリテスト用に例外を発生させる関数。"""
        raise ValueError("Test exception for telemetry")

    mock_histogram = Mock()
    exception_test_tool._invocation_duration_histogram = mock_histogram
    span_exporter.clear()
    # invoke を呼び出し例外を期待する。
    with pytest.raises(ValueError, match="Test exception for telemetry"):
        await exception_test_tool.invoke(x=1, y=2, tool_call_id="exception_call")
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert OtelAttr.TOOL_EXECUTION_OPERATION.value in span.name
    assert "exception_test_tool" in span.name
    assert span.attributes[OtelAttr.TOOL_NAME] == "exception_test_tool"
    assert span.attributes[OtelAttr.TOOL_CALL_ID] == "exception_call"
    assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "A test tool that raises an exception"
    assert span.attributes[OtelAttr.TOOL_ARGUMENTS] == '{"x": 1, "y": 2}'
    assert span.attributes[OtelAttr.ERROR_TYPE] == ValueError.__name__
    assert span.status.status_code == trace.StatusCode.ERROR

    # エラー属性でヒストグラムが呼ばれたことを検証する。
    mock_histogram.record.assert_called_once()
    call_args = mock_histogram.record.call_args
    attributes = call_args[1]["attributes"]
    assert attributes[OtelAttr.ERROR_TYPE] == ValueError.__name__


async def test_ai_function_invoke_telemetry_async_function(span_exporter: InMemorySpanExporter):
    """非同期関数に対するテレメトリ付き ai_function invoke メソッドをテストします。"""

    @ai_function(
        name="async_telemetry_test",
        description="An async test tool for telemetry",
    )
    async def async_telemetry_test(x: int, y: int) -> int:
        """テレメトリテスト用の非同期関数。"""
        return x * y

    mock_histogram = Mock()
    async_telemetry_test._invocation_duration_histogram = mock_histogram
    span_exporter.clear()
    # invoke を呼び出す。
    result = await async_telemetry_test.invoke(x=3, y=4, tool_call_id="async_call")

    # 結果を検証する。
    assert result == 12
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert OtelAttr.TOOL_EXECUTION_OPERATION.value in span.name
    assert "async_telemetry_test" in span.name
    assert span.attributes[OtelAttr.TOOL_NAME] == "async_telemetry_test"
    assert span.attributes[OtelAttr.TOOL_CALL_ID] == "async_call"
    assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
    assert span.attributes[OtelAttr.TOOL_DESCRIPTION] == "An async test tool for telemetry"
    assert span.attributes[OtelAttr.TOOL_ARGUMENTS] == '{"x": 3, "y": 4}'

    # ヒストグラム記録を検証する。
    mock_histogram.record.assert_called_once()
    call_args = mock_histogram.record.call_args
    attributes = call_args[1]["attributes"]
    assert attributes[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] == "async_telemetry_test"


async def test_ai_function_invoke_invalid_pydantic_args():
    """無効なPydanticモデル引数でai_functionのinvokeメソッドをテストします。"""

    @ai_function(name="invalid_args_test", description="A test tool for invalid args")
    def invalid_args_test(x: int, y: int) -> int:
        """無効なPydantic引数をテストするための関数です。"""
        return x + y

    # 異なるPydanticモデルを作成します。
    class WrongModel(BaseModel):
        a: str
        b: str

    wrong_args = WrongModel(a="hello", b="world")

    # 間違ったモデルタイプでinvokeを呼び出します。
    with pytest.raises(TypeError, match="Expected invalid_args_test_input, got WrongModel"):
        await invalid_args_test.invoke(arguments=wrong_args)


def test_ai_function_serialization():
    """AIFunctionのシリアライズとデシリアライズをテストします。"""

    def serialize_test(x: int, y: int) -> int:
        """シリアライズをテストするための関数です。"""
        return x - y

    serialize_test_ai_function = ai_function(name="serialize_test", description="A test tool for serialization")(
        serialize_test
    )

    # dictにシリアライズします。
    tool_dict = serialize_test_ai_function.to_dict()
    assert tool_dict["type"] == "ai_function"
    assert tool_dict["name"] == "serialize_test"
    assert tool_dict["description"] == "A test tool for serialization"
    assert tool_dict["input_model"] == {
        "properties": {"x": {"title": "X", "type": "integer"}, "y": {"title": "Y", "type": "integer"}},
        "required": ["x", "y"],
        "title": "serialize_test_input",
        "type": "object",
    }

    # dictからデシリアライズします。
    restored_tool = AIFunction.from_dict(tool_dict, dependencies={"ai_function": {"func": serialize_test}})
    assert isinstance(restored_tool, AIFunction)
    assert restored_tool.name == "serialize_test"
    assert restored_tool.description == "A test tool for serialization"
    assert restored_tool.parameters() == serialize_test_ai_function.parameters()
    assert restored_tool(10, 4) == 6

    # インスタンス名付きでdictからデシリアライズします。
    restored_tool_2 = AIFunction.from_dict(
        tool_dict, dependencies={"ai_function": {"name:serialize_test": {"func": serialize_test}}}
    )
    assert isinstance(restored_tool_2, AIFunction)
    assert restored_tool_2.name == "serialize_test"
    assert restored_tool_2.description == "A test tool for serialization"
    assert restored_tool_2.parameters() == serialize_test_ai_function.parameters()
    assert restored_tool_2(10, 4) == 6


# region HostedCodeInterpreterToolと_parse_inputs


def test_hosted_code_interpreter_tool_default():
    """デフォルトパラメータでHostedCodeInterpreterToolをテストします。"""
    tool = HostedCodeInterpreterTool()

    assert tool.name == "code_interpreter"
    assert tool.inputs == []
    assert tool.description == ""
    assert tool.additional_properties is None
    assert str(tool) == "HostedCodeInterpreterTool(name=code_interpreter)"


def test_hosted_code_interpreter_tool_with_description():
    """説明と追加プロパティ付きでHostedCodeInterpreterToolをテストします。"""
    tool = HostedCodeInterpreterTool(
        description="A test code interpreter",
        additional_properties={"version": "1.0", "language": "python"},
    )

    assert tool.name == "code_interpreter"
    assert tool.description == "A test code interpreter"
    assert tool.additional_properties == {"version": "1.0", "language": "python"}


def test_parse_inputs_none():
    """None入力で_parse_inputsをテストします。"""
    result = _parse_inputs(None)
    assert result == []


def test_parse_inputs_string():
    """文字列入力で_parse_inputsをテストします。"""
    from agent_framework import UriContent

    result = _parse_inputs("http://example.com")
    assert len(result) == 1
    assert isinstance(result[0], UriContent)
    assert result[0].uri == "http://example.com"
    assert result[0].media_type == "text/plain"


def test_parse_inputs_list_of_strings():
    """文字列のリストで_parse_inputsをテストします。"""
    from agent_framework import UriContent

    inputs = ["http://example.com", "https://test.org"]
    result = _parse_inputs(inputs)

    assert len(result) == 2
    assert all(isinstance(item, UriContent) for item in result)
    assert result[0].uri == "http://example.com"
    assert result[1].uri == "https://test.org"
    assert all(item.media_type == "text/plain" for item in result)


def test_parse_inputs_uri_dict():
    """URI辞書で_parse_inputsをテストします。"""
    from agent_framework import UriContent

    input_dict = {"uri": "http://example.com", "media_type": "application/json"}
    result = _parse_inputs(input_dict)

    assert len(result) == 1
    assert isinstance(result[0], UriContent)
    assert result[0].uri == "http://example.com"
    assert result[0].media_type == "application/json"


def test_parse_inputs_hosted_file_dict():
    """ホストされたファイル辞書で_parse_inputsをテストします。"""
    from agent_framework import HostedFileContent

    input_dict = {"file_id": "file-123"}
    result = _parse_inputs(input_dict)

    assert len(result) == 1
    assert isinstance(result[0], HostedFileContent)
    assert result[0].file_id == "file-123"


def test_parse_inputs_hosted_vector_store_dict():
    """ホストされたベクターストア辞書で_parse_inputsをテストします。"""
    from agent_framework import HostedVectorStoreContent

    input_dict = {"vector_store_id": "vs-789"}
    result = _parse_inputs(input_dict)

    assert len(result) == 1
    assert isinstance(result[0], HostedVectorStoreContent)
    assert result[0].vector_store_id == "vs-789"


def test_parse_inputs_data_dict():
    """データ辞書で_parse_inputsをテストします。"""
    from agent_framework import DataContent

    input_dict = {"data": b"test data", "media_type": "application/octet-stream"}
    result = _parse_inputs(input_dict)

    assert len(result) == 1
    assert isinstance(result[0], DataContent)
    assert result[0].uri == "data:application/octet-stream;base64,dGVzdCBkYXRh"
    assert result[0].media_type == "application/octet-stream"


def test_parse_inputs_ai_contents_instance():
    """Contentsインスタンスで_parse_inputsをテストします。"""
    from agent_framework import TextContent

    text_content = TextContent(text="Hello, world!")
    result = _parse_inputs(text_content)

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "Hello, world!"


def test_parse_inputs_mixed_list():
    """混合入力タイプで_parse_inputsをテストします。"""
    from agent_framework import HostedFileContent, TextContent, UriContent

    inputs = [
        "http://example.com",  # string
        {"uri": "https://test.org", "media_type": "text/html"},  # URI dict
        {"file_id": "file-456"},  # hosted file dict
        TextContent(text="Hello"),  # Contents instance
    ]

    result = _parse_inputs(inputs)

    assert len(result) == 4
    assert isinstance(result[0], UriContent)
    assert result[0].uri == "http://example.com"
    assert isinstance(result[1], UriContent)
    assert result[1].uri == "https://test.org"
    assert result[1].media_type == "text/html"
    assert isinstance(result[2], HostedFileContent)
    assert result[2].file_id == "file-456"
    assert isinstance(result[3], TextContent)
    assert result[3].text == "Hello"


def test_parse_inputs_unsupported_dict():
    """サポートされていない辞書形式で_parse_inputsをテストします。"""
    input_dict = {"unsupported_key": "value"}

    with pytest.raises(ValueError, match="Unsupported input type"):
        _parse_inputs(input_dict)


def test_parse_inputs_unsupported_type():
    """サポートされていない入力タイプで_parse_inputsをテストします。"""
    with pytest.raises(TypeError, match="Unsupported input type: int"):
        _parse_inputs(123)


def test_hosted_code_interpreter_tool_with_string_input():
    """文字列入力でHostedCodeInterpreterToolをテストします。"""
    from agent_framework import UriContent

    tool = HostedCodeInterpreterTool(inputs="http://example.com")

    assert len(tool.inputs) == 1
    assert isinstance(tool.inputs[0], UriContent)
    assert tool.inputs[0].uri == "http://example.com"


def test_hosted_code_interpreter_tool_with_dict_inputs():
    """辞書入力でHostedCodeInterpreterToolをテストします。"""
    from agent_framework import HostedFileContent, UriContent

    inputs = [{"uri": "http://example.com", "media_type": "text/html"}, {"file_id": "file-123"}]

    tool = HostedCodeInterpreterTool(inputs=inputs)

    assert len(tool.inputs) == 2
    assert isinstance(tool.inputs[0], UriContent)
    assert tool.inputs[0].uri == "http://example.com"
    assert tool.inputs[0].media_type == "text/html"
    assert isinstance(tool.inputs[1], HostedFileContent)
    assert tool.inputs[1].file_id == "file-123"


def test_hosted_code_interpreter_tool_with_ai_contents():
    """ContentsインスタンスでHostedCodeInterpreterToolをテストします。"""
    from agent_framework import DataContent, TextContent

    inputs = [TextContent(text="Hello, world!"), DataContent(data=b"test", media_type="text/plain")]

    tool = HostedCodeInterpreterTool(inputs=inputs)

    assert len(tool.inputs) == 2
    assert isinstance(tool.inputs[0], TextContent)
    assert tool.inputs[0].text == "Hello, world!"
    assert isinstance(tool.inputs[1], DataContent)
    assert tool.inputs[1].media_type == "text/plain"


def test_hosted_code_interpreter_tool_with_single_input():
    """単一入力（リストではない）でHostedCodeInterpreterToolをテストします。"""
    from agent_framework import HostedFileContent

    input_dict = {"file_id": "file-single"}
    tool = HostedCodeInterpreterTool(inputs=input_dict)

    assert len(tool.inputs) == 1
    assert isinstance(tool.inputs[0], HostedFileContent)
    assert tool.inputs[0].file_id == "file-single"


def test_hosted_code_interpreter_tool_with_unknown_input():
    """単一の未知の入力でHostedCodeInterpreterToolをテストします。"""
    with pytest.raises(ValueError, match="Unsupported input type"):
        HostedCodeInterpreterTool(inputs={"hosted_file": "file-single"})


# region HostedMCPToolのテスト


def test_hosted_mcp_tool_with_other_fields():
    """特定のapproval dict、headers、追加プロパティでHostedMCPToolを作成するテスト。"""
    tool = HostedMCPTool(
        name="mcp-tool",
        url="https://mcp.example",
        description="A test MCP tool",
        headers={"x": "y"},
        additional_properties={"p": 1},
    )

    assert tool.name == "mcp-tool"
    # pydanticのAnyUrlは文字列のように保持されます。
    assert str(tool.url).startswith("https://")
    assert tool.headers == {"x": "y"}
    assert tool.additional_properties == {"p": 1}
    assert tool.description == "A test MCP tool"


@pytest.mark.parametrize(
    "approval_mode",
    [
        "always_require",
        "never_require",
        {
            "always_require_approval": {"toolA"},
            "never_require_approval": {"toolB"},
        },
        {
            "always_require_approval": ["toolA"],
            "never_require_approval": ("toolB",),
        },
    ],
    ids=["always_require", "never_require", "specific", "specific_with_parsing"],
)
def test_hosted_mcp_tool_with_approval_mode(approval_mode: str | dict[str, Any]):
    """特定のapproval dict、headers、追加プロパティでHostedMCPToolを作成するテスト。"""
    tool = HostedMCPTool(name="mcp-tool", url="https://mcp.example", approval_mode=approval_mode)

    assert tool.name == "mcp-tool"
    # pydanticのAnyUrlは文字列のように保持されます。
    assert str(tool.url).startswith("https://")
    if not isinstance(approval_mode, dict):
        assert tool.approval_mode == approval_mode
    else:
        # approval_modeはセットにパースされます。
        assert isinstance(tool.approval_mode["always_require_approval"], set)
        assert isinstance(tool.approval_mode["never_require_approval"], set)
        assert "toolA" in tool.approval_mode["always_require_approval"]
        assert "toolB" in tool.approval_mode["never_require_approval"]


def test_hosted_mcp_tool_invalid_approval_mode_raises():
    """無効なapproval_mode文字列はServiceInitializationErrorを発生させるべきです。"""
    with pytest.raises(ToolException):
        HostedMCPTool(name="bad", url="https://x", approval_mode="invalid_mode")


@pytest.mark.parametrize(
    "tools",
    [
        {"toolA", "toolB"},
        ("toolA", "toolB"),
        ["toolA", "toolB"],
        ["toolA", "toolB", "toolA"],
    ],
    ids=[
        "set",
        "tuple",
        "list",
        "list_with_duplicates",
    ],
)
def test_hosted_mcp_tool_with_allowed_tools(tools: list[str] | tuple[str, ...] | set[str]):
    """許可されたツールのリストでHostedMCPToolを作成するテスト。"""
    tool = HostedMCPTool(
        name="mcp-tool",
        url="https://mcp.example",
        allowed_tools=tools,
    )

    assert tool.name == "mcp-tool"
    # pydanticのAnyUrlは文字列のように保持されます。
    assert str(tool.url).startswith("https://")
    # approval_modeはセットにパースされます。
    assert isinstance(tool.allowed_tools, set)
    assert tool.allowed_tools == {"toolA", "toolB"}


def test_hosted_mcp_tool_with_dict_of_allowed_tools():
    """許可されたツールの辞書でHostedMCPToolを作成するテスト。"""
    with pytest.raises(ToolException):
        HostedMCPTool(
            name="mcp-tool",
            url="https://mcp.example",
            allowed_tools={"toolA": "Tool A", "toolC": "Tool C"},
        )


# region 承認フローテスト


@pytest.fixture
def mock_chat_client():
    """承認フローをテストするためのモックチャットクライアントを作成します。"""
    from agent_framework import ChatMessage, ChatResponse, ChatResponseUpdate

    class MockChatClient:
        def __init__(self):
            self.call_count = 0
            self.responses = []

        async def get_response(self, messages, **kwargs):
            """事前定義されたレスポンスを返すモックのget_response。"""
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
            # デフォルトレスポンス。
            return ChatResponse(
                messages=[ChatMessage(role="assistant", contents=["Default response"])],
            )

        async def get_streaming_response(self, messages, **kwargs):
            """事前定義された更新を生成するモックのget_streaming_response。"""
            if self.call_count < len(self.responses):
                response = self.responses[self.call_count]
                self.call_count += 1
                # レスポンスから更新を生成します。
                for msg in response.messages:
                    for content in msg.contents:
                        yield ChatResponseUpdate(contents=[content], role=msg.role)
            else:
                # デフォルトレスポンス。
                yield ChatResponseUpdate(contents=["Default response"], role="assistant")

    return MockChatClient()


@ai_function(name="no_approval_tool", description="Tool that doesn't require approval")
def no_approval_tool(x: int) -> int:
    """承認を必要としないツール。"""
    return x * 2


@ai_function(
    name="requires_approval_tool",
    description="Tool that requires approval",
    approval_mode="always_require",
)
def requires_approval_tool(x: int) -> int:
    """承認を必要とするツール。"""
    return x * 3


async def test_non_streaming_single_function_no_approval():
    """承認を必要としない単一関数呼び出しの非ストリーミングハンドラーをテストします。"""
    from agent_framework import ChatMessage, ChatResponse, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_response

    # モッククライアントを作成します。
    mock_client = type("MockClient", (), {})()

    # レスポンスを作成します：最初は関数呼び出し、次に最終回答。
    initial_response = ChatResponse(
        messages=[
            ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="call_1", name="no_approval_tool", arguments='{"x": 5}')],
            )
        ]
    )
    final_response = ChatResponse(messages=[ChatMessage(role="assistant", contents=["The result is 10"])])

    call_count = [0]
    responses = [initial_response, final_response]

    async def mock_get_response(self, messages, **kwargs):
        result = responses[call_count[0]]
        call_count[0] += 1
        return result

    # 関数をラップします。
    wrapped = _handle_function_calls_response(mock_get_response)

    # 実行します。
    result = await wrapped(mock_client, messages=[], tools=[no_approval_tool])

    # 検証：3つのメッセージがあるはずです：関数呼び出し、関数結果、最終回答。
    assert len(result.messages) == 3
    assert isinstance(result.messages[0].contents[0], FunctionCallContent)
    from agent_framework import FunctionResultContent

    assert isinstance(result.messages[1].contents[0], FunctionResultContent)
    assert result.messages[1].contents[0].result == 10  # 5 * 2
    assert result.messages[2].contents[0] == "The result is 10"


async def test_non_streaming_single_function_requires_approval():
    """承認を必要とする単一関数呼び出しの非ストリーミングハンドラーをテストします。"""
    from agent_framework import ChatMessage, ChatResponse, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_response

    mock_client = type("MockClient", (), {})()

    # 関数呼び出しを含む初期レスポンス。
    initial_response = ChatResponse(
        messages=[
            ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="call_1", name="requires_approval_tool", arguments='{"x": 5}')],
            )
        ]
    )

    call_count = [0]
    responses = [initial_response]

    async def mock_get_response(self, messages, **kwargs):
        result = responses[call_count[0]]
        call_count[0] += 1
        return result

    wrapped = _handle_function_calls_response(mock_get_response)

    # 実行します。
    result = await wrapped(mock_client, messages=[], tools=[requires_approval_tool])

    # 検証：関数呼び出しと承認要求を含む1つのメッセージを返すはずです。
    from agent_framework import FunctionApprovalRequestContent

    assert len(result.messages) == 1
    assert len(result.messages[0].contents) == 2
    assert isinstance(result.messages[0].contents[0], FunctionCallContent)
    assert isinstance(result.messages[0].contents[1], FunctionApprovalRequestContent)
    assert result.messages[0].contents[1].function_call.name == "requires_approval_tool"


async def test_non_streaming_two_functions_both_no_approval():
    """承認を必要としない2つの関数呼び出しの非ストリーミングハンドラーをテストします。"""
    from agent_framework import ChatMessage, ChatResponse, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_response

    mock_client = type("MockClient", (), {})()

    # 同じツールへの2つの関数呼び出しを含む初期レスポンス。
    initial_response = ChatResponse(
        messages=[
            ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="call_1", name="no_approval_tool", arguments='{"x": 5}'),
                    FunctionCallContent(call_id="call_2", name="no_approval_tool", arguments='{"x": 3}'),
                ],
            )
        ]
    )
    final_response = ChatResponse(
        messages=[ChatMessage(role="assistant", contents=["Both tools executed successfully"])]
    )

    call_count = [0]
    responses = [initial_response, final_response]

    async def mock_get_response(self, messages, **kwargs):
        result = responses[call_count[0]]
        call_count[0] += 1
        return result

    wrapped = _handle_function_calls_response(mock_get_response)

    # 実行します。
    result = await wrapped(mock_client, messages=[], tools=[no_approval_tool])

    # 検証：関数呼び出し、結果、最終回答があるはずです。
    from agent_framework import FunctionResultContent

    assert len(result.messages) == 3
    # 最初のメッセージには両方の関数呼び出しがあります。
    assert len(result.messages[0].contents) == 2
    # 2番目のメッセージには両方の結果があります。
    assert len(result.messages[1].contents) == 2
    assert all(isinstance(c, FunctionResultContent) for c in result.messages[1].contents)
    assert result.messages[1].contents[0].result == 10  # 5 * 2
    assert result.messages[1].contents[1].result == 6  # 3 * 2


async def test_non_streaming_two_functions_both_require_approval():
    """承認を必要とする2つの関数呼び出しの非ストリーミングハンドラーをテストします。"""
    from agent_framework import ChatMessage, ChatResponse, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_response

    mock_client = type("MockClient", (), {})()

    # 同じツールへの2つの関数呼び出しを含む初期レスポンス。
    initial_response = ChatResponse(
        messages=[
            ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="call_1", name="requires_approval_tool", arguments='{"x": 5}'),
                    FunctionCallContent(call_id="call_2", name="requires_approval_tool", arguments='{"x": 3}'),
                ],
            )
        ]
    )

    call_count = [0]
    responses = [initial_response]

    async def mock_get_response(self, messages, **kwargs):
        result = responses[call_count[0]]
        call_count[0] += 1
        return result

    wrapped = _handle_function_calls_response(mock_get_response)

    # 実行します。
    result = await wrapped(mock_client, messages=[], tools=[requires_approval_tool])

    # 検証：関数呼び出しと承認要求を含む1つのメッセージを返すはずです。
    from agent_framework import FunctionApprovalRequestContent

    assert len(result.messages) == 1
    assert len(result.messages[0].contents) == 4  # 2つの関数呼び出し + 2つの承認要求。
    function_calls = [c for c in result.messages[0].contents if isinstance(c, FunctionCallContent)]
    approval_requests = [c for c in result.messages[0].contents if isinstance(c, FunctionApprovalRequestContent)]
    assert len(function_calls) == 2
    assert len(approval_requests) == 2
    assert approval_requests[0].function_call.name == "requires_approval_tool"
    assert approval_requests[1].function_call.name == "requires_approval_tool"


async def test_non_streaming_two_functions_mixed_approval():
    """1つが承認を必要とする2つの関数呼び出しの非ストリーミングハンドラーをテストします。"""
    from agent_framework import ChatMessage, ChatResponse, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_response

    mock_client = type("MockClient", (), {})()

    # 2つの関数呼び出しを含む初期レスポンス。
    initial_response = ChatResponse(
        messages=[
            ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="call_1", name="no_approval_tool", arguments='{"x": 5}'),
                    FunctionCallContent(call_id="call_2", name="requires_approval_tool", arguments='{"x": 3}'),
                ],
            )
        ]
    )

    call_count = [0]
    responses = [initial_response]

    async def mock_get_response(self, messages, **kwargs):
        result = responses[call_count[0]]
        call_count[0] += 1
        return result

    wrapped = _handle_function_calls_response(mock_get_response)

    # 実行します。
    result = await wrapped(mock_client, messages=[], tools=[no_approval_tool, requires_approval_tool])

    # 検証：両方に承認要求を返すはずです（1つが承認を必要とするときはすべて承認待ちになります）。
    from agent_framework import FunctionApprovalRequestContent

    assert len(result.messages) == 1
    assert len(result.messages[0].contents) == 4  # 2つの関数呼び出し + 2つの承認要求。
    approval_requests = [c for c in result.messages[0].contents if isinstance(c, FunctionApprovalRequestContent)]
    assert len(approval_requests) == 2


async def test_streaming_single_function_no_approval():
    """承認を必要としない単一関数呼び出しのストリーミングハンドラーをテストします。"""
    from agent_framework import ChatResponseUpdate, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_streaming_response

    mock_client = type("MockClient", (), {})()

    # 関数呼び出しの初期レスポンス、その後関数実行後の最終レスポンス。
    initial_updates = [
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_1", name="no_approval_tool", arguments='{"x": 5}')],
            role="assistant",
        )
    ]
    final_updates = [ChatResponseUpdate(contents=["The result is 10"], role="assistant")]

    call_count = [0]
    updates_list = [initial_updates, final_updates]

    async def mock_get_streaming_response(self, messages, **kwargs):
        updates = updates_list[call_count[0]]
        call_count[0] += 1
        for update in updates:
            yield update

    wrapped = _handle_function_calls_streaming_response(mock_get_streaming_response)

    # 実行して更新を収集します。
    updates = []
    async for update in wrapped(mock_client, messages=[], tools=[no_approval_tool]):
        updates.append(update)

    # 検証：関数呼び出しの更新、ツール結果の更新（ラッパーによって注入）、最終更新があるはずです。
    from agent_framework import FunctionResultContent, Role

    assert len(updates) >= 3
    # 最初の更新は関数呼び出しです。
    assert isinstance(updates[0].contents[0], FunctionCallContent)
    # 2番目の更新はツール結果（ラッパーによって注入）であるべきです。
    assert updates[1].role == Role.TOOL
    assert isinstance(updates[1].contents[0], FunctionResultContent)
    assert updates[1].contents[0].result == 10  # 5 * 2
    # 最後の更新は最終メッセージです。
    assert updates[-1].contents[0] == "The result is 10"


async def test_streaming_single_function_requires_approval():
    """承認を必要とする単一関数呼び出しのストリーミングハンドラーをテストします。"""
    from agent_framework import ChatResponseUpdate, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_streaming_response

    mock_client = type("MockClient", (), {})()

    # 関数呼び出しの初期レスポンス。
    initial_updates = [
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_1", name="requires_approval_tool", arguments='{"x": 5}')],
            role="assistant",
        )
    ]

    call_count = [0]
    updates_list = [initial_updates]

    async def mock_get_streaming_response(self, messages, **kwargs):
        updates = updates_list[call_count[0]]
        call_count[0] += 1
        for update in updates:
            yield update

    wrapped = _handle_function_calls_streaming_response(mock_get_streaming_response)

    # 実行して更新を収集します。
    updates = []
    async for update in wrapped(mock_client, messages=[], tools=[requires_approval_tool]):
        updates.append(update)

    # 検証：関数呼び出しとその後の承認要求を生成するはずです。
    from agent_framework import FunctionApprovalRequestContent, Role

    assert len(updates) == 2
    assert isinstance(updates[0].contents[0], FunctionCallContent)
    assert updates[1].role == Role.ASSISTANT
    assert isinstance(updates[1].contents[0], FunctionApprovalRequestContent)


async def test_streaming_two_functions_both_no_approval():
    """承認を必要としない2つの関数呼び出しのストリーミングハンドラーをテストします。"""
    from agent_framework import ChatResponseUpdate, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_streaming_response

    mock_client = type("MockClient", (), {})()

    # 同じツールへの2つの関数呼び出しを含む初期レスポンス。
    initial_updates = [
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_1", name="no_approval_tool", arguments='{"x": 5}')],
            role="assistant",
        ),
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_2", name="no_approval_tool", arguments='{"x": 3}')],
            role="assistant",
        ),
    ]
    final_updates = [ChatResponseUpdate(contents=["Both tools executed successfully"], role="assistant")]

    call_count = [0]
    updates_list = [initial_updates, final_updates]

    async def mock_get_streaming_response(self, messages, **kwargs):
        updates = updates_list[call_count[0]]
        call_count[0] += 1
        for update in updates:
            yield update

    wrapped = _handle_function_calls_streaming_response(mock_get_streaming_response)

    # 実行して更新を収集します。
    updates = []
    async for update in wrapped(mock_client, messages=[], tools=[no_approval_tool]):
        updates.append(update)

    # 検証：両方の関数呼び出し、両方の結果を含む1つのツール結果更新、最終メッセージがあるはずです。
    from agent_framework import FunctionResultContent, Role

    assert len(updates) >= 3
    # 最初の2つの更新は関数呼び出しです。
    assert isinstance(updates[0].contents[0], FunctionCallContent)
    assert isinstance(updates[1].contents[0], FunctionCallContent)
    # 両方の結果を含むツール結果の更新があるはずです。
    tool_updates = [u for u in updates if u.role == Role.TOOL]
    assert len(tool_updates) == 1
    assert len(tool_updates[0].contents) == 2
    assert all(isinstance(c, FunctionResultContent) for c in tool_updates[0].contents)


async def test_streaming_two_functions_both_require_approval():
    """承認を必要とする2つの関数呼び出しのストリーミングハンドラーをテストします。"""
    from agent_framework import ChatResponseUpdate, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_streaming_response

    mock_client = type("MockClient", (), {})()

    # 同じツールへの2つの関数呼び出しを含む初期レスポンス。
    initial_updates = [
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_1", name="requires_approval_tool", arguments='{"x": 5}')],
            role="assistant",
        ),
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_2", name="requires_approval_tool", arguments='{"x": 3}')],
            role="assistant",
        ),
    ]

    call_count = [0]
    updates_list = [initial_updates]

    async def mock_get_streaming_response(self, messages, **kwargs):
        updates = updates_list[call_count[0]]
        call_count[0] += 1
        for update in updates:
            yield update

    wrapped = _handle_function_calls_streaming_response(mock_get_streaming_response)

    # 実行して更新を収集します。
    updates = []
    async for update in wrapped(mock_client, messages=[], tools=[requires_approval_tool]):
        updates.append(update)

    # 検証：両方の関数呼び出しとその後の承認要求を生成するはずです。
    from agent_framework import FunctionApprovalRequestContent, Role

    assert len(updates) == 3
    assert isinstance(updates[0].contents[0], FunctionCallContent)
    assert isinstance(updates[1].contents[0], FunctionCallContent)
    # 両方の承認要求を含むAssistantの更新。
    assert updates[2].role == Role.ASSISTANT
    assert len(updates[2].contents) == 2
    assert all(isinstance(c, FunctionApprovalRequestContent) for c in updates[2].contents)


async def test_streaming_two_functions_mixed_approval():
    """1つが承認を必要とする2つの関数呼び出しのストリーミングハンドラーをテストします。"""
    from agent_framework import ChatResponseUpdate, FunctionCallContent
    from agent_framework._tools import _handle_function_calls_streaming_response

    mock_client = type("MockClient", (), {})()

    # 2つの関数呼び出しを含む初期レスポンス。
    initial_updates = [
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_1", name="no_approval_tool", arguments='{"x": 5}')],
            role="assistant",
        ),
        ChatResponseUpdate(
            contents=[FunctionCallContent(call_id="call_2", name="requires_approval_tool", arguments='{"x": 3}')],
            role="assistant",
        ),
    ]

    call_count = [0]
    updates_list = [initial_updates]

    async def mock_get_streaming_response(self, messages, **kwargs):
        updates = updates_list[call_count[0]]
        call_count[0] += 1
        for update in updates:
            yield update

    wrapped = _handle_function_calls_streaming_response(mock_get_streaming_response)

    # 実行して更新を収集します。
    updates = []
    async for update in wrapped(mock_client, messages=[], tools=[no_approval_tool, requires_approval_tool]):
        updates.append(update)

    # 検証：両方の関数呼び出しとその後の承認要求を生成するはずです（1つが承認を必要とするときはすべて待機します）。
    from agent_framework import FunctionApprovalRequestContent, Role

    assert len(updates) == 3
    assert isinstance(updates[0].contents[0], FunctionCallContent)
    assert isinstance(updates[1].contents[0], FunctionCallContent)
    # 両方の承認要求を含むAssistantの更新。
    assert updates[2].role == Role.ASSISTANT
    assert len(updates[2].contents) == 2
    assert all(isinstance(c, FunctionApprovalRequestContent) for c in updates[2].contents)
