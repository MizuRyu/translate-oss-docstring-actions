# Copyright (c) Microsoft. All rights reserved.
# type: ignore[reportPrivateUsage]
import os
from contextlib import _AsyncGeneratorContextManager  # type: ignore
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl, ValidationError

from agent_framework import (
    ChatMessage,
    DataContent,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    MCPWebsocketTool,
    Role,
    TextContent,
    ToolProtocol,
    UriContent,
)
from agent_framework._mcp import (
    MCPTool,
    _ai_content_to_mcp_types,
    _chat_message_to_mcp_types,
    _get_input_model_from_mcp_prompt,
    _get_input_model_from_mcp_tool,
    _mcp_call_tool_result_to_ai_contents,
    _mcp_prompt_message_to_chat_message,
    _mcp_type_to_ai_content,
    _normalize_mcp_name,
)
from agent_framework.exceptions import ToolException, ToolExecutionException

# 統合テストのスキップ条件
skip_if_mcp_integration_tests_disabled = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "false").lower() != "true" or os.getenv("LOCAL_MCP_URL", "") == "",
    reason="No LOCAL_MCP_URL provided; skipping integration tests."
    if os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    else "Integration tests are disabled.",
)


# ヘルパー関数のテスト
def test_normalize_mcp_name():
    """MCP名の正規化をテストします。"""
    assert _normalize_mcp_name("valid_name") == "valid_name"
    assert _normalize_mcp_name("name-with-dashes") == "name-with-dashes"
    assert _normalize_mcp_name("name.with.dots") == "name.with.dots"
    assert _normalize_mcp_name("name with spaces") == "name-with-spaces"
    assert _normalize_mcp_name("name@with#special$chars") == "name-with-special-chars"
    assert _normalize_mcp_name("name/with\\slashes") == "name-with-slashes"


def test_mcp_prompt_message_to_ai_content():
    """MCPのpromptメッセージからAIコンテンツへの変換をテストします。"""
    mcp_message = types.PromptMessage(role="user", content=types.TextContent(type="text", text="Hello, world!"))
    ai_content = _mcp_prompt_message_to_chat_message(mcp_message)

    assert isinstance(ai_content, ChatMessage)
    assert ai_content.role.value == "user"
    assert len(ai_content.contents) == 1
    assert isinstance(ai_content.contents[0], TextContent)
    assert ai_content.contents[0].text == "Hello, world!"
    assert ai_content.raw_representation == mcp_message


def test_mcp_call_tool_result_to_ai_contents():
    """MCPのtool結果からAIコンテンツへの変換をテストします。"""
    mcp_result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="Result text"),
            types.ImageContent(type="image", data="data:image/png;base64,xyz", mimeType="image/png"),
        ]
    )
    ai_contents = _mcp_call_tool_result_to_ai_contents(mcp_result)

    assert len(ai_contents) == 2
    assert isinstance(ai_contents[0], TextContent)
    assert ai_contents[0].text == "Result text"
    assert isinstance(ai_contents[1], DataContent)
    assert ai_contents[1].uri == "data:image/png;base64,xyz"
    assert ai_contents[1].media_type == "image/png"


def test_mcp_content_types_to_ai_content_text():
    """MCPのテキストコンテンツからAIコンテンツへの変換をテストします。"""
    mcp_content = types.TextContent(type="text", text="Sample text")
    ai_content = _mcp_type_to_ai_content(mcp_content)

    assert isinstance(ai_content, TextContent)
    assert ai_content.text == "Sample text"
    assert ai_content.raw_representation == mcp_content


def test_mcp_content_types_to_ai_content_image():
    """MCPの画像コンテンツからAIコンテンツへの変換をテストします。"""
    mcp_content = types.ImageContent(type="image", data="data:image/jpeg;base64,abc", mimeType="image/jpeg")
    ai_content = _mcp_type_to_ai_content(mcp_content)

    assert isinstance(ai_content, DataContent)
    assert ai_content.uri == "data:image/jpeg;base64,abc"
    assert ai_content.media_type == "image/jpeg"
    assert ai_content.raw_representation == mcp_content


def test_mcp_content_types_to_ai_content_audio():
    """MCPのオーディオコンテンツからAIコンテンツへの変換をテストします。"""
    mcp_content = types.AudioContent(type="audio", data="data:audio/wav;base64,def", mimeType="audio/wav")
    ai_content = _mcp_type_to_ai_content(mcp_content)

    assert isinstance(ai_content, DataContent)
    assert ai_content.uri == "data:audio/wav;base64,def"
    assert ai_content.media_type == "audio/wav"
    assert ai_content.raw_representation == mcp_content


def test_mcp_content_types_to_ai_content_resource_link():
    """MCPのリソースリンクからAIコンテンツへの変換をテストします。"""
    mcp_content = types.ResourceLink(
        type="resource_link",
        uri=AnyUrl("https://example.com/resource"),
        name="test_resource",
        mimeType="application/json",
    )
    ai_content = _mcp_type_to_ai_content(mcp_content)

    assert isinstance(ai_content, UriContent)
    assert ai_content.uri == "https://example.com/resource"
    assert ai_content.media_type == "application/json"
    assert ai_content.raw_representation == mcp_content


def test_mcp_content_types_to_ai_content_embedded_resource_text():
    """MCPの埋め込みテキストリソースからAIコンテンツへの変換をテストします。"""
    text_resource = types.TextResourceContents(
        uri=AnyUrl("file://test.txt"), mimeType="text/plain", text="Embedded text content"
    )
    mcp_content = types.EmbeddedResource(type="resource", resource=text_resource)
    ai_content = _mcp_type_to_ai_content(mcp_content)

    assert isinstance(ai_content, TextContent)
    assert ai_content.text == "Embedded text content"
    assert ai_content.raw_representation == mcp_content


def test_mcp_content_types_to_ai_content_embedded_resource_blob():
    """MCPの埋め込みblobリソースからAIコンテンツへの変換をテストします。"""
    # blobフィールドには適切なdata URIを使用します。これはMCPの実装が期待している形式です。
    blob_resource = types.BlobResourceContents(
        uri=AnyUrl("file://test.bin"),
        mimeType="application/octet-stream",
        blob="data:application/octet-stream;base64,dGVzdCBkYXRh",
    )
    mcp_content = types.EmbeddedResource(type="resource", resource=blob_resource)
    ai_content = _mcp_type_to_ai_content(mcp_content)

    assert isinstance(ai_content, DataContent)
    assert ai_content.uri == "data:application/octet-stream;base64,dGVzdCBkYXRh"
    assert ai_content.media_type == "application/octet-stream"
    assert ai_content.raw_representation == mcp_content


def test_ai_content_to_mcp_content_types_text():
    """AIのテキストコンテンツからMCPコンテンツへの変換をテストします。"""
    ai_content = TextContent(text="Sample text")
    mcp_content = _ai_content_to_mcp_types(ai_content)

    assert isinstance(mcp_content, types.TextContent)
    assert mcp_content.type == "text"
    assert mcp_content.text == "Sample text"


def test_ai_content_to_mcp_content_types_data_image():
    """AIのデータコンテンツからMCPコンテンツへの変換をテストします。"""
    ai_content = DataContent(uri="data:image/png;base64,xyz", media_type="image/png")
    mcp_content = _ai_content_to_mcp_types(ai_content)

    assert isinstance(mcp_content, types.ImageContent)
    assert mcp_content.type == "image"
    assert mcp_content.data == "data:image/png;base64,xyz"
    assert mcp_content.mimeType == "image/png"


def test_ai_content_to_mcp_content_types_data_audio():
    """AIのデータコンテンツからMCPコンテンツへの変換をテストします。"""
    ai_content = DataContent(uri="data:audio/mpeg;base64,xyz", media_type="audio/mpeg")
    mcp_content = _ai_content_to_mcp_types(ai_content)

    assert isinstance(mcp_content, types.AudioContent)
    assert mcp_content.type == "audio"
    assert mcp_content.data == "data:audio/mpeg;base64,xyz"
    assert mcp_content.mimeType == "audio/mpeg"


def test_ai_content_to_mcp_content_types_data_binary():
    """AIのデータコンテンツからMCPコンテンツへの変換をテストします。"""
    ai_content = DataContent(uri="data:application/octet-stream;base64,xyz", media_type="application/octet-stream")
    mcp_content = _ai_content_to_mcp_types(ai_content)

    assert isinstance(mcp_content, types.EmbeddedResource)
    assert mcp_content.type == "resource"
    assert mcp_content.resource.blob == "data:application/octet-stream;base64,xyz"
    assert mcp_content.resource.mimeType == "application/octet-stream"


def test_ai_content_to_mcp_content_types_uri():
    """AIのURIコンテンツからMCPコンテンツへの変換をテストします。"""
    ai_content = UriContent(uri="https://example.com/resource", media_type="application/json")
    mcp_content = _ai_content_to_mcp_types(ai_content)

    assert isinstance(mcp_content, types.ResourceLink)
    assert mcp_content.type == "resource_link"
    assert str(mcp_content.uri) == "https://example.com/resource"
    assert mcp_content.mimeType == "application/json"


def test_chat_message_to_mcp_types():
    message = ChatMessage(
        role="user",
        contents=[TextContent(text="test"), DataContent(uri="data:image/png;base64,xyz", media_type="image/png")],
    )
    mcp_contents = _chat_message_to_mcp_types(message)
    assert len(mcp_contents) == 2
    assert isinstance(mcp_contents[0], types.TextContent)
    assert isinstance(mcp_contents[1], types.ImageContent)


def test_get_input_model_from_mcp_tool():
    """MCPツールからの入力モデルの作成をテストします。"""
    tool = types.Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {"param1": {"type": "string"}, "param2": {"type": "number"}},
            "required": ["param1"],
        },
    )
    model = _get_input_model_from_mcp_tool(tool)

    # モデルが正しく動作することを検証するためにインスタンスを作成します。
    instance = model(param1="test", param2=42)
    assert instance.param1 == "test"
    assert instance.param2 == 42

    # バリデーションをテストします。
    with pytest.raises(ValidationError):  # Missing required param1
        model(param2=42)


def test_get_input_model_from_mcp_tool_with_nested_object():
    """ネストされたオブジェクトプロパティを持つMCPツールからの入力モデルの作成をテストします。"""
    tool = types.Tool(
        name="get_customer_detail",
        description="Get customer details",
        inputSchema={
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer"}},
                    "required": ["customer_id"],
                }
            },
            "required": ["params"],
        },
    )
    model = _get_input_model_from_mcp_tool(tool)

    # ネストされたオブジェクトを持つモデルが正しく動作することを検証するためにインスタンスを作成します。
    instance = model(params={"customer_id": 251})
    assert instance.params == {"customer_id": 251}
    assert isinstance(instance.params, dict)

    # model_dumpが正しいネスト構造を生成することを検証します。
    dumped = instance.model_dump()
    assert dumped == {"params": {"customer_id": 251}}


def test_get_input_model_from_mcp_tool_with_ref_schema():
    """$refスキーマを持つMCPツールからの入力モデルの作成をテストします。

    これは、スキーマに$refを含むPydanticモデルを使用するFastMCPツールをシミュレートします。
    スキーマは解決され、ネストされたオブジェクトは保持されるべきです。
    """
    # これはFastMCPが以下のようなコードを生成する場合に似ています: async def get_customer_detail(params:
    # CustomerIdParam) -> CustomerDetail
    tool = types.Tool(
        name="get_customer_detail",
        description="Get customer details",
        inputSchema={
            "type": "object",
            "properties": {"params": {"$ref": "#/$defs/CustomerIdParam"}},
            "required": ["params"],
            "$defs": {
                "CustomerIdParam": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "integer"}},
                    "required": ["customer_id"],
                }
            },
        },
    )
    model = _get_input_model_from_mcp_tool(tool)

    # $refスキーマを持つモデルが正しく動作することを検証するためにインスタンスを作成します。
    instance = model(params={"customer_id": 251})
    assert instance.params == {"customer_id": 251}
    assert isinstance(instance.params, dict)

    # model_dumpが正しいネスト構造を生成することを検証します。
    dumped = instance.model_dump()
    assert dumped == {"params": {"customer_id": 251}}


def test_get_input_model_from_mcp_prompt():
    """MCPのpromptからの入力モデルの作成をテストします。"""
    prompt = types.Prompt(
        name="test_prompt",
        description="A test prompt",
        arguments=[
            types.PromptArgument(name="arg1", description="First argument", required=True),
            types.PromptArgument(name="arg2", description="Second argument", required=False),
        ],
    )
    model = _get_input_model_from_mcp_prompt(prompt)

    # モデルが正しく動作することを検証するためにインスタンスを作成します。
    instance = model(arg1="test", arg2="optional")
    assert instance.arg1 == "test"
    assert instance.arg2 == "optional"

    # バリデーションをテストします。
    with pytest.raises(ValidationError):  # Missing required arg1
        model(arg2="optional")


# MCPToolのテスト
async def test_local_mcp_server_initialization():
    """MCPToolの初期化をテストします。"""
    server = MCPTool(name="test_server")
    assert isinstance(server, ToolProtocol)
    assert server.name == "test_server"
    assert server.session is None
    assert server.functions == []


async def test_local_mcp_server_context_manager():
    """コンテキストマネージャとしてのMCPToolをテストします。"""

    class TestServer(MCPTool):
        async def connect(self):
            # 接続のモック
            self.session = Mock(spec=ClientSession)

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server")
    async with server:
        assert server.session is not None

    assert server.session is None


async def test_local_mcp_server_load_functions():
    """MCPサーバーからの関数の読み込みをテストします。"""

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            # ツールリスト応答のモック
            self.session.list_tools = AsyncMock(
                return_value=types.ListToolsResult(
                    tools=[
                        types.Tool(
                            name="test_tool",
                            description="Test tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                                "required": ["param"],
                            },
                        )
                    ]
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server")
    assert isinstance(server, ToolProtocol)
    async with server:
        await server.load_tools()
        assert len(server.functions) == 1
        assert server.functions[0].name == "test_tool"


async def test_local_mcp_server_load_prompts():
    """MCPサーバーからのpromptの読み込みをテストします。"""

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            # promptリスト応答のモック
            self.session.list_prompts = AsyncMock(
                return_value=types.ListPromptsResult(
                    prompts=[
                        types.Prompt(
                            name="test_prompt",
                            description="Test prompt",
                            arguments=[types.PromptArgument(name="arg", description="Test arg", required=True)],
                        )
                    ]
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server")
    async with server:
        await server.load_prompts()
        assert len(server.functions) == 1
        assert server.functions[0].name == "test_prompt"


async def test_local_mcp_server_function_execution():
    """MCPサーバーを介した関数実行をテストします。"""

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            self.session.list_tools = AsyncMock(
                return_value=types.ListToolsResult(
                    tools=[
                        types.Tool(
                            name="test_tool",
                            description="Test tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                                "required": ["param"],
                            },
                        )
                    ]
                )
            )
            self.session.call_tool = AsyncMock(
                return_value=types.CallToolResult(
                    content=[types.TextContent(type="text", text="Tool executed successfully")]
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server")
    async with server:
        await server.load_tools()
        func = server.functions[0]
        result = await func.invoke(param="test_value")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Tool executed successfully"


async def test_local_mcp_server_function_execution_with_nested_object():
    """ネストされたオブジェクト引数を使ったMCPサーバーを介した関数実行をテストします。"""

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            self.session.list_tools = AsyncMock(
                return_value=types.ListToolsResult(
                    tools=[
                        types.Tool(
                            name="get_customer_detail",
                            description="Get customer details",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "params": {
                                        "type": "object",
                                        "properties": {"customer_id": {"type": "integer"}},
                                        "required": ["customer_id"],
                                    }
                                },
                                "required": ["params"],
                            },
                        )
                    ]
                )
            )
            self.session.call_tool = AsyncMock(
                return_value=types.CallToolResult(
                    content=[types.TextContent(type="text", text='{"name": "John Doe", "id": 251}')]
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server")
    async with server:
        await server.load_tools()
        func = server.functions[0]

        # ネストされたオブジェクトを使って呼び出し
        result = await func.invoke(params={"customer_id": 251})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        # session.call_toolが正しいネスト構造で呼び出されたことを検証します。
        server.session.call_tool.assert_called_once()
        call_args = server.session.call_tool.call_args
        assert call_args.kwargs["arguments"] == {"params": {"customer_id": 251}}


async def test_local_mcp_server_function_execution_error():
    """関数実行のエラー処理をテストします。"""

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            self.session.list_tools = AsyncMock(
                return_value=types.ListToolsResult(
                    tools=[
                        types.Tool(
                            name="test_tool",
                            description="Test tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                                "required": ["param"],
                            },
                        )
                    ]
                )
            )
            # MCPエラーを発生させるツール呼び出しのモック
            self.session.call_tool = AsyncMock(
                side_effect=McpError(types.ErrorData(code=-1, message="Tool execution failed"))
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server")
    async with server:
        await server.load_tools()
        func = server.functions[0]

        with pytest.raises(ToolExecutionException):
            await func.invoke(param="test_value")


async def test_local_mcp_server_prompt_execution():
    """MCPサーバーを介したprompt実行をテストします。"""

    class TestMCPTool(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            self.session.list_prompts = AsyncMock(
                return_value=types.ListPromptsResult(
                    prompts=[
                        types.Prompt(
                            name="test_prompt",
                            description="Test prompt",
                            arguments=[types.PromptArgument(name="arg", description="Test arg", required=True)],
                        )
                    ]
                )
            )
            self.session.get_prompt = AsyncMock(
                return_value=types.GetPromptResult(
                    description="Generated prompt",
                    messages=[
                        types.PromptMessage(role="user", content=types.TextContent(type="text", text="Test message"))
                    ],
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestMCPTool(name="test_server")
    async with server:
        await server.load_prompts()
        prompt = server.functions[0]
        result = await prompt.invoke(arg="test_value")

        assert len(result) == 1
        assert isinstance(result[0], ChatMessage)
        assert result[0].role == Role.USER
        assert len(result[0].contents) == 1
        assert result[0].contents[0].text == "Test message"


@pytest.mark.parametrize(
    "approval_mode,expected_approvals",
    [
        ("always_require", {"tool_one": "always_require", "tool_two": "always_require"}),
        ("never_require", {"tool_one": "never_require", "tool_two": "never_require"}),
        (
            {"always_require_approval": ["tool_one"], "never_require_approval": ["tool_two"]},
            {"tool_one": "always_require", "tool_two": "never_require"},
        ),
    ],
)
async def test_mcp_tool_approval_mode(approval_mode, expected_approvals):
    """MCPToolのapproval_modeパラメータを様々な設定でテストします。

    approval_modeパラメータはツールの実行前承認が必要かどうかを制御します。
    グローバル設定（"always_require"または"never_require"）か、ツールごとにdictで設定可能です。
    """

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            self.session.list_tools = AsyncMock(
                return_value=types.ListToolsResult(
                    tools=[
                        types.Tool(
                            name="tool_one",
                            description="First tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                            },
                        ),
                        types.Tool(
                            name="tool_two",
                            description="Second tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                            },
                        ),
                    ]
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server", approval_mode=approval_mode)
    async with server:
        await server.load_tools()
        assert len(server.functions) == 2

        # 各ツールが期待されるapproval modeを持つことを検証します。
        for func in server.functions:
            assert func.approval_mode == expected_approvals[func.name]


@pytest.mark.parametrize(
    "allowed_tools,expected_count,expected_names",
    [
        (None, 3, ["tool_one", "tool_two", "tool_three"]),  # None means all tools are allowed
        (["tool_one"], 1, ["tool_one"]),  # Only tool_one is allowed
        (["tool_one", "tool_three"], 2, ["tool_one", "tool_three"]),  # Two tools allowed
        (["nonexistent_tool"], 0, []),  # No matching tools
    ],
)
async def test_mcp_tool_allowed_tools(allowed_tools, expected_count, expected_names):
    """MCPToolのallowed_toolsパラメータを様々な設定でテストします。

    allowed_toolsパラメータはfunctionsプロパティで公開されるツールをフィルタリングします。
    Noneの場合は全ての読み込まれたツールが利用可能です。リストの場合は、その名前がリストにあるツールのみが公開されます。
    """

    class TestServer(MCPTool):
        async def connect(self):
            self.session = Mock(spec=ClientSession)
            self.session.list_tools = AsyncMock(
                return_value=types.ListToolsResult(
                    tools=[
                        types.Tool(
                            name="tool_one",
                            description="First tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                            },
                        ),
                        types.Tool(
                            name="tool_two",
                            description="Second tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                            },
                        ),
                        types.Tool(
                            name="tool_three",
                            description="Third tool",
                            inputSchema={
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                            },
                        ),
                    ]
                )
            )

        def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
            return None

    server = TestServer(name="test_server", allowed_tools=allowed_tools)
    async with server:
        await server.load_tools()
        # _functionsは全てのツールを含むべきです。
        assert len(server._functions) == 3

        # functionsプロパティはallowed_toolsに基づいてフィルタリングされるべきです。
        assert len(server.functions) == expected_count
        actual_names = [func.name for func in server.functions]
        assert sorted(actual_names) == sorted(expected_names)


# サーバー実装のテスト
def test_local_mcp_stdio_tool_init():
    """MCPStdioToolの初期化をテストします。"""
    tool = MCPStdioTool(name="test", command="echo", args=["hello"])
    assert tool.name == "test"
    assert tool.command == "echo"
    assert tool.args == ["hello"]


def test_local_mcp_websocket_tool_init():
    """MCPWebsocketToolの初期化をテストします。"""
    tool = MCPWebsocketTool(name="test", url="ws://localhost:8080")
    assert tool.name == "test"
    assert tool.url == "ws://localhost:8080"


def test_local_mcp_streamable_http_tool_init():
    """MCPStreamableHTTPToolの初期化をテストします。"""
    tool = MCPStreamableHTTPTool(name="test", url="http://localhost:8080")
    assert tool.name == "test"
    assert tool.url == "http://localhost:8080"


# 統合テスト
@pytest.mark.flaky
@skip_if_mcp_integration_tests_disabled
async def test_streamable_http_integration():
    """MCP StreamableHTTPの統合をテストします。"""
    url = os.environ.get("LOCAL_MCP_URL", "")
    if not url.startswith("http"):
        pytest.skip("LOCAL_MCP_URL is not an HTTP URL")

    tool = MCPStreamableHTTPTool(name="integration_test", url=url)

    async with tool:
        # 接続してツールを読み込めることをテストします。
        assert tool.session is not None
        assert isinstance(tool.functions, list)

        # 利用可能な関数があれば、その情報を取得しようとします。
        assert tool.functions, "The MCP server should have at least one function."

        func = tool.functions[0]

        assert hasattr(func, "name")
        assert hasattr(func, "description")

        result = await func.invoke(query="What is Agent Framework?")
        assert result[0].text is not None


async def test_mcp_tool_message_handler_notification():
    """message_handlerがtools/list_changedとprompts/list_changed通知を正しく処理することをテストします。"""
    tool = MCPStdioTool(name="test_tool", command="python")

    # load_toolsとload_promptsメソッドのモック
    tool.load_tools = AsyncMock()
    tool.load_prompts = AsyncMock()

    # toolsリスト変更通知のテスト
    tools_notification = Mock(spec=types.ServerNotification)
    tools_notification.root = Mock()
    tools_notification.root.method = "notifications/tools/list_changed"

    result = await tool.message_handler(tools_notification)
    assert result is None
    tool.load_tools.assert_called_once()

    # モックをリセット
    tool.load_tools.reset_mock()

    # promptsリスト変更通知のテスト
    prompts_notification = Mock(spec=types.ServerNotification)
    prompts_notification.root = Mock()
    prompts_notification.root.method = "notifications/prompts/list_changed"

    result = await tool.message_handler(prompts_notification)
    assert result is None
    tool.load_prompts.assert_called_once()

    # 未処理通知のテスト
    unknown_notification = Mock(spec=types.ServerNotification)
    unknown_notification.root = Mock()
    unknown_notification.root.method = "notifications/unknown"

    result = await tool.message_handler(unknown_notification)
    assert result is None


async def test_mcp_tool_message_handler_error():
    """message_handlerが例外をログに記録しNoneを返して正常に処理することをテストします。"""
    tool = MCPStdioTool(name="test_tool", command="python")

    # 例外メッセージを使ったテスト
    test_exception = RuntimeError("Test error message")

    # message handlerはエラーをログに記録しNoneを返すべきです。
    result = await tool.message_handler(test_exception)
    assert result is None


async def test_mcp_tool_sampling_callback_no_client():
    """チャットClientが利用できない場合のsampling callbackのエラー経路をテストします。"""
    tool = MCPStdioTool(name="test_tool", command="python")

    # 最小限のparamsモックを作成します。
    params = Mock()
    params.messages = []

    result = await tool.sampling_callback(Mock(), params)

    assert isinstance(result, types.ErrorData)
    assert result.code == types.INTERNAL_ERROR
    assert "No chat client available" in result.message


async def test_mcp_tool_sampling_callback_chat_client_exception():
    """チャットClientが例外を発生させる場合のsampling callbackをテストします。"""
    tool = MCPStdioTool(name="test_tool", command="python")

    # 例外を発生させるチャットClientのモック
    mock_chat_client = AsyncMock()
    mock_chat_client.get_response.side_effect = RuntimeError("Chat client error")

    tool.chat_client = mock_chat_client

    # モックparamsを作成します。
    params = Mock()
    mock_message = Mock()
    mock_message.role = "user"
    mock_message.content = Mock()
    mock_message.content.text = "Test question"
    params.messages = [mock_message]
    params.temperature = None
    params.maxTokens = None
    params.stopSequences = None

    result = await tool.sampling_callback(Mock(), params)

    assert isinstance(result, types.ErrorData)
    assert result.code == types.INTERNAL_ERROR
    assert "Failed to get chat message content: Chat client error" in result.message


async def test_mcp_tool_sampling_callback_no_valid_content():
    """応答に有効なコンテンツタイプがない場合のsampling callbackをテストします。"""
    from agent_framework import ChatMessage, DataContent, Role

    tool = MCPStdioTool(name="test_tool", command="python")

    # 無効なコンテンツタイプのみを含む応答を持つチャットClientのモック
    mock_chat_client = AsyncMock()
    mock_response = Mock()
    mock_response.messages = [
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[DataContent(uri="data:application/json;base64,e30K", media_type="application/json")],
        )
    ]
    mock_response.model_id = "test-model"
    mock_chat_client.get_response.return_value = mock_response

    tool.chat_client = mock_chat_client

    # モックparamsを作成します。
    params = Mock()
    mock_message = Mock()
    mock_message.role = "user"
    mock_message.content = Mock()
    mock_message.content.text = "Test question"
    params.messages = [mock_message]
    params.temperature = None
    params.maxTokens = None
    params.stopSequences = None

    result = await tool.sampling_callback(Mock(), params)

    assert isinstance(result, types.ErrorData)
    assert result.code == types.INTERNAL_ERROR
    assert "Failed to get right content types from the response." in result.message


# connect()メソッドのエラー処理をテストします。


async def test_connect_session_creation_failure():
    """ClientSessionの作成に失敗した場合、connect()がToolExceptionを発生させることをテストします。"""
    tool = MCPStdioTool(name="test", command="test-command")

    # 成功したtransport作成のモック
    mock_transport = (Mock(), Mock())  # (read_stream, write_stream)
    mock_context_manager = Mock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_transport)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    tool.get_mcp_client = Mock(return_value=mock_context_manager)

    # 例外を発生させるClientSessionのモック
    with patch("agent_framework._mcp.ClientSession") as mock_session_class:
        mock_session_class.side_effect = RuntimeError("Session creation failed")

        with pytest.raises(ToolException) as exc_info:
            await tool.connect()

        assert "Failed to create MCP session" in str(exc_info.value)
        assert "Session creation failed" in str(exc_info.value.__cause__)


async def test_connect_initialization_failure_http_no_command():
    """HTTPツールでsession.initialize()が失敗した場合のconnect()をテストします（command属性なし）。"""
    tool = MCPStreamableHTTPTool(name="test", url="http://example.com")

    # 成功したtransport作成のモック
    mock_transport = (Mock(), Mock())
    mock_context_manager = Mock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_transport)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    tool.get_mcp_client = Mock(return_value=mock_context_manager)

    # 成功したsession作成だが初期化に失敗したモック
    mock_session = Mock()
    mock_session.initialize = AsyncMock(side_effect=ConnectionError("Server not ready"))

    with patch("agent_framework._mcp.ClientSession") as mock_session_class:
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ToolException) as exc_info:
            await tool.connect()

        # HTTPツールにcommandがないため、一般的なエラーメッセージを使用すべきです。
        assert "MCP server failed to initialize" in str(exc_info.value)
        assert "Server not ready" in str(exc_info.value)


async def test_connect_cleanup_on_transport_failure():
    """transport作成に失敗した場合に_exit_stack.aclose()が呼ばれることをテストします。"""
    tool = MCPStdioTool(name="test", command="test-command")

    # 呼び出しを検証するために_exit_stack.acloseをモックします。
    tool._exit_stack.aclose = AsyncMock()

    # 例外を発生させるget_mcp_clientのモック
    tool.get_mcp_client = Mock(side_effect=RuntimeError("Transport failed"))

    with pytest.raises(ToolException):
        await tool.connect()

    # クリーンアップが呼ばれたことを検証します。
    tool._exit_stack.aclose.assert_called_once()


async def test_connect_cleanup_on_initialization_failure():
    """初期化に失敗した場合に_exit_stack.aclose()が呼ばれることをテストします。"""
    tool = MCPStdioTool(name="test", command="test-command")

    # 呼び出しを検証するために_exit_stack.acloseをモックします。
    tool._exit_stack.aclose = AsyncMock()

    # 成功したtransport作成のモック
    mock_transport = (Mock(), Mock())
    mock_context_manager = Mock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_transport)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    tool.get_mcp_client = Mock(return_value=mock_context_manager)

    # 成功したsession作成だが初期化に失敗したモック
    mock_session = Mock()
    mock_session.initialize = AsyncMock(side_effect=RuntimeError("Init failed"))

    with patch("agent_framework._mcp.ClientSession") as mock_session_class:
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ToolException):
            await tool.connect()

        # クリーンアップが呼ばれたことを検証します。
        tool._exit_stack.aclose.assert_called_once()


def test_mcp_stdio_tool_get_mcp_client_with_env_and_kwargs():
    """環境変数とclient kwargsを使ったMCPStdioTool.get_mcp_client()のテスト。"""
    env_vars = {"PATH": "/usr/bin", "DEBUG": "1"}
    tool = MCPStdioTool(name="test", command="test-command", env=env_vars, custom_param="value1", another_param=42)

    with patch("agent_framework._mcp.stdio_client"), patch("agent_framework._mcp.StdioServerParameters") as mock_params:
        tool.get_mcp_client()

        # カスタムkwargsを含む全てのパラメータが渡されたことを検証します。
        mock_params.assert_called_once_with(
            command="test-command", args=[], env=env_vars, custom_param="value1", another_param=42
        )


def test_mcp_streamable_http_tool_get_mcp_client_all_params():
    """全てのパラメータを使ったMCPStreamableHTTPTool.get_mcp_client()のテスト。"""
    tool = MCPStreamableHTTPTool(
        name="test",
        url="http://example.com",
        headers={"Auth": "token"},
        timeout=30.0,
        sse_read_timeout=10.0,
        terminate_on_close=True,
        custom_param="test",
    )

    with patch("agent_framework._mcp.streamablehttp_client") as mock_http_client:
        tool.get_mcp_client()

        # 全てのパラメータが渡されたことを検証します。
        mock_http_client.assert_called_once_with(
            url="http://example.com",
            headers={"Auth": "token"},
            timeout=30.0,
            sse_read_timeout=10.0,
            terminate_on_close=True,
            custom_param="test",
        )


def test_mcp_websocket_tool_get_mcp_client_with_kwargs():
    """client kwargsを使ったMCPWebsocketTool.get_mcp_client()のテスト。"""
    tool = MCPWebsocketTool(
        name="test", url="wss://example.com", max_size=1024, ping_interval=30, compression="deflate"
    )

    with patch("agent_framework._mcp.websocket_client") as mock_ws_client:
        tool.get_mcp_client()

        # 全てのkwargsが渡されたことを検証します。
        mock_ws_client.assert_called_once_with(
            url="wss://example.com", max_size=1024, ping_interval=30, compression="deflate"
        )
