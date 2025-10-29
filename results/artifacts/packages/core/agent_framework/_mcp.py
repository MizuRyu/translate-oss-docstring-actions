# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import re
import sys
from abc import abstractmethod
from collections.abc import Collection
from contextlib import AsyncExitStack, _AsyncGeneratorContextManager  # type: ignore
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from mcp.shared.context import RequestContext
from mcp.shared.exceptions import McpError
from mcp.shared.session import RequestResponder
from pydantic import BaseModel, create_model

from ._tools import AIFunction, HostedMCPSpecificApproval
from ._types import ChatMessage, Contents, DataContent, Role, TextContent, UriContent
from .exceptions import ToolException, ToolExecutionException

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover

if TYPE_CHECKING:
    from ._clients import ChatClientProtocol

logger = logging.getLogger(__name__)

# region: Helpers

LOG_LEVEL_MAPPING: dict[types.LoggingLevel, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}

__all__ = [
    "MCPStdioTool",
    "MCPStreamableHTTPTool",
    "MCPWebsocketTool",
]


def _mcp_prompt_message_to_chat_message(
    mcp_type: types.PromptMessage | types.SamplingMessage,
) -> ChatMessage:
    """MCPのコンテナタイプをAgent Frameworkのタイプに変換します。"""
    return ChatMessage(
        role=Role(value=mcp_type.role),
        contents=[_mcp_type_to_ai_content(mcp_type.content)],
        raw_representation=mcp_type,
    )


def _mcp_call_tool_result_to_ai_contents(
    mcp_type: types.CallToolResult,
) -> list[Contents]:
    """MCPのコンテナタイプをAgent Frameworkのタイプに変換します。"""
    return [_mcp_type_to_ai_content(item) for item in mcp_type.content]


def _mcp_type_to_ai_content(
    mcp_type: types.ImageContent | types.TextContent | types.AudioContent | types.EmbeddedResource | types.ResourceLink,
) -> Contents:
    """MCPのタイプをAgent Frameworkのタイプに変換します。"""
    match mcp_type:
        case types.TextContent():
            return TextContent(text=mcp_type.text, raw_representation=mcp_type)
        case types.ImageContent() | types.AudioContent():
            return DataContent(uri=mcp_type.data, media_type=mcp_type.mimeType, raw_representation=mcp_type)
        case types.ResourceLink():
            return UriContent(
                uri=str(mcp_type.uri), media_type=mcp_type.mimeType or "application/json", raw_representation=mcp_type
            )
        case _:
            match mcp_type.resource:
                case types.TextResourceContents():
                    return TextContent(
                        text=mcp_type.resource.text,
                        raw_representation=mcp_type,
                        additional_properties=mcp_type.annotations.model_dump() if mcp_type.annotations else None,
                    )
                case types.BlobResourceContents():
                    return DataContent(
                        uri=mcp_type.resource.blob,
                        media_type=mcp_type.resource.mimeType,
                        raw_representation=mcp_type,
                        additional_properties=mcp_type.annotations.model_dump() if mcp_type.annotations else None,
                    )


def _ai_content_to_mcp_types(
    content: Contents,
) -> types.TextContent | types.ImageContent | types.AudioContent | types.EmbeddedResource | types.ResourceLink | None:
    """BaseContentタイプをMCPタイプに変換します。"""
    match content:
        case TextContent():
            return types.TextContent(type="text", text=content.text)
        case DataContent():
            if content.media_type and content.media_type.startswith("image/"):
                return types.ImageContent(type="image", data=content.uri, mimeType=content.media_type)
            if content.media_type and content.media_type.startswith("audio/"):
                return types.AudioContent(type="audio", data=content.uri, mimeType=content.media_type)
            if content.media_type and content.media_type.startswith("application/"):
                return types.EmbeddedResource(
                    type="resource",
                    resource=types.BlobResourceContents(
                        blob=content.uri,
                        mimeType=content.media_type,
                        # uriはMCPでは制限されませんが、設定する必要があります。 data contentのuriはdata uriを含み、
                        # ここで意味するuriとは異なります。UriContentがこれに該当します。
                        uri=content.additional_properties.get("uri", "af://binary")
                        if content.additional_properties
                        else "af://binary",  # type: ignore[reportArgumentType]
                    ),
                )
            return None
        case UriContent():
            return types.ResourceLink(
                type="resource_link",
                uri=content.uri,  # type: ignore[reportArgumentType]
                mimeType=content.media_type,
                name=content.additional_properties.get("name", "Unknown")
                if content.additional_properties
                else "Unknown",
            )
        case _:
            return None


def _chat_message_to_mcp_types(
    content: ChatMessage,
) -> list[types.TextContent | types.ImageContent | types.AudioContent | types.EmbeddedResource | types.ResourceLink]:
    """ChatMessageをMCPタイプのリストに変換します。"""
    messages: list[
        types.TextContent | types.ImageContent | types.AudioContent | types.EmbeddedResource | types.ResourceLink
    ] = []
    for item in content.contents:
        mcp_content = _ai_content_to_mcp_types(item)
        if mcp_content:
            messages.append(mcp_content)
    return messages


def _get_input_model_from_mcp_prompt(prompt: types.Prompt) -> type[BaseModel]:
    """プロンプトのパラメータからPydanticモデルを作成します。"""
    # 'arguments'が欠落または空であるかをチェックします。
    if not prompt.arguments:
        return create_model(f"{prompt.name}_input")

    field_definitions: dict[str, Any] = {}
    for prompt_argument in prompt.arguments:
        # プロンプトでは、すべての引数は通常必須で文字列型です。 ただしプロンプト引数で別途指定がある場合を除きます。
        python_type = str  # プロンプト引数のデフォルトタイプ。

        # create_model用のフィールド定義を作成します。
        if prompt_argument.required:
            field_definitions[prompt_argument.name] = (python_type, ...)
        else:
            field_definitions[prompt_argument.name] = (python_type, None)

    return create_model(f"{prompt.name}_input", **field_definitions)


def _get_input_model_from_mcp_tool(tool: types.Tool) -> type[BaseModel]:
    """ツールのパラメータからPydanticモデルを作成します。"""
    properties = tool.inputSchema.get("properties", None)
    required = tool.inputSchema.get("required", [])
    definitions = tool.inputSchema.get("$defs", {})

    # 'properties'が欠落しているか辞書でないかをチェックします。
    if not properties:
        return create_model(f"{tool.name}_input")

    def resolve_type(prop_details: dict[str, Any]) -> type:
        """JSON SchemaのタイプをPythonのタイプに解決し、$refを処理します。"""
        # $refを参照を解決して処理します。
        if "$ref" in prop_details:
            ref = prop_details["$ref"]
            # 参照パスを抽出します（例: "#/$defs/CustomerIdParam" -> "CustomerIdParam"）。
            if ref.startswith("#/$defs/"):
                def_name = ref.split("/")[-1]
                if def_name in definitions:
                    # 参照を解決し、そのタイプを使用します。
                    resolved = definitions[def_name]
                    return resolve_type(resolved)
            # 参照を解決できない場合は安全のためdictをデフォルトとします。
            return dict

        # JSON SchemaのタイプをPythonのタイプにマッピングします。
        json_type = prop_details.get("type", "string")
        match json_type:
            case "integer":
                return int
            case "number":
                return float
            case "boolean":
                return bool
            case "array":
                return list
            case "object":
                return dict
            case _:
                return str  # default

    field_definitions: dict[str, Any] = {}
    for prop_name, prop_details in properties.items():
        prop_details = json.loads(prop_details) if isinstance(prop_details, str) else prop_details

        python_type = resolve_type(prop_details)

        # create_model用のフィールド定義を作成します。
        if prop_name in required:
            field_definitions[prop_name] = (python_type, ...)
        else:
            default_value = prop_details.get("default", None)
            field_definitions[prop_name] = (python_type, default_value)

    return create_model(f"{tool.name}_input", **field_definitions)


def _normalize_mcp_name(name: str) -> str:
    """MCPのツール/プロンプト名を許可された識別子パターン（A-Za-z0-9_.-）に正規化します。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)


# region: MCP Plugin


class MCPTool:
    """Model Context Protocolサーバーに接続するためのメインMCPクラス。

    これはMCPツール実装の基底クラスです。接続管理、
    ツールおよびプロンプトの読み込み、MCPサーバーとの通信を処理します。

    注意:
        MCPToolは直接インスタンス化できません。以下のサブクラスを使用してください:
        MCPStdioTool、MCPStreamableHTTPTool、またはMCPWebsocketTool。

    Examples:
        使用例はサブクラスのドキュメントを参照してください:

        - stdioベースのMCPサーバー用 :class:`MCPStdioTool`
        - HTTPベースのMCPサーバー用 :class:`MCPStreamableHTTPTool`
        - WebSocketベースのMCPサーバー用 :class:`MCPWebsocketTool`

    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        approval_mode: Literal["always_require", "never_require"] | HostedMCPSpecificApproval | None = None,
        allowed_tools: Collection[str] | None = None,
        load_tools: bool = True,
        load_prompts: bool = True,
        session: ClientSession | None = None,
        request_timeout: int | None = None,
        chat_client: "ChatClientProtocol | None" = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        """MCP Toolの基底を初期化します。

        注意:
            このメソッドは使用せず、サブクラスのMCPStreamableHTTPTool、MCPWebsocketTool
            またはMCPStdioToolを使用してください。

        """
        self.name = name
        self.description = description or ""
        self.approval_mode = approval_mode
        self.allowed_tools = allowed_tools
        self.additional_properties = additional_properties
        self.load_tools_flag = load_tools
        self.load_prompts_flag = load_prompts
        self._exit_stack = AsyncExitStack()
        self.session = session
        self.request_timeout = request_timeout
        self.chat_client = chat_client
        self._functions: list[AIFunction[Any, Any]] = []
        self.is_connected: bool = False

    def __str__(self) -> str:
        return f"MCPTool(name={self.name}, description={self.description})"

    @property
    def functions(self) -> list[AIFunction[Any, Any]]:
        """許可されている関数のリストを取得します。"""
        if not self.allowed_tools:
            return self._functions
        return [func for func in self._functions if func.name in self.allowed_tools]

    async def connect(self) -> None:
        """MCPサーバーに接続します。

        MCPサーバーへの接続を確立し、セッションを初期化し、
        設定されていればツールとプロンプトを読み込みます。

        Raises:
            ToolException: 接続またはセッション初期化に失敗した場合。

        """
        if not self.session:
            try:
                transport = await self._exit_stack.enter_async_context(self.get_mcp_client())
            except Exception as ex:
                await self._exit_stack.aclose()
                command = getattr(self, "command", None)
                if command:
                    error_msg = f"Failed to start MCP server '{command}': {ex}"
                else:
                    error_msg = f"Failed to connect to MCP server: {ex}"
                raise ToolException(error_msg, inner_exception=ex) from ex
            try:
                session = await self._exit_stack.enter_async_context(
                    ClientSession(
                        read_stream=transport[0],
                        write_stream=transport[1],
                        read_timeout_seconds=timedelta(seconds=self.request_timeout) if self.request_timeout else None,
                        message_handler=self.message_handler,
                        logging_callback=self.logging_callback,
                        sampling_callback=self.sampling_callback,
                    )
                )
            except Exception as ex:
                await self._exit_stack.aclose()
                raise ToolException(
                    message="Failed to create MCP session. Please check your configuration.", inner_exception=ex
                ) from ex
            try:
                await session.initialize()
            except Exception as ex:
                await self._exit_stack.aclose()
                # 初期化失敗に関するコンテキストを提供します。
                command = getattr(self, "command", None)
                if command:
                    args_str = " ".join(getattr(self, "args", []))
                    full_command = f"{command} {args_str}".strip()
                    error_msg = f"MCP server '{full_command}' failed to initialize: {ex}"
                else:
                    error_msg = f"MCP server failed to initialize: {ex}"
                raise ToolException(error_msg, inner_exception=ex) from ex
            self.session = session
        elif self.session._request_id == 0:  # type: ignore[reportPrivateUsage]
            # セッションが初期化されていない場合、再初期化が必要です。
            await self.session.initialize()
        logger.debug("Connected to MCP server: %s", self.session)
        self.is_connected = True
        if self.load_tools_flag:
            await self.load_tools()
        if self.load_prompts_flag:
            await self.load_prompts()

        if logger.level != logging.NOTSET:
            try:
                await self.session.set_logging_level(
                    next(level for level, value in LOG_LEVEL_MAPPING.items() if value == logger.level)
                )
            except Exception as exc:
                logger.warning("Failed to set log level to %s", logger.level, exc_info=exc)

    async def sampling_callback(
        self, context: RequestContext[ClientSession, Any], params: types.CreateMessageRequestParams
    ) -> types.CreateMessageResult | types.ErrorData:
        """サンプリング用のコールバック関数。

        MCPサーバーがメッセージの生成を必要とするときに呼び出されます。
        設定されたチャットクライアントを使用してレスポンスを生成します。

        注意:
            これはこの関数の簡易版です。より複雑なサンプリングを許可するためにオーバーライド可能です。
            初期化時にセッションに追加されるため、カスタマイズする最良の方法です。

        Args:
            context: MCPサーバーからのリクエストコンテキスト。
            params: メッセージ作成リクエストのパラメータ。

        Returns:
            生成されたメッセージを含むCreateMessageResult、または生成失敗時のErrorData。

        """
        if not self.chat_client:
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message="No chat client available. Please set a chat client.",
            )
        logger.debug("Sampling callback called with params: %s", params)
        messages: list[ChatMessage] = []
        for msg in params.messages:
            messages.append(_mcp_prompt_message_to_chat_message(msg))
        try:
            response = await self.chat_client.get_response(
                messages,
                temperature=params.temperature,
                max_tokens=params.maxTokens,
                stop=params.stopSequences,
            )
        except Exception as ex:
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Failed to get chat message content: {ex}",
            )
        if not response or not response.messages:
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message="Failed to get chat message content.",
            )
        mcp_contents = _chat_message_to_mcp_types(response.messages[0])
        # TextContentまたはImageContentタイプの最初のコンテンツを取得します。
        mcp_content = next(
            (content for content in mcp_contents if isinstance(content, (types.TextContent, types.ImageContent))),
            None,
        )
        if not mcp_content:
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message="Failed to get right content types from the response.",
            )
        return types.CreateMessageResult(
            role="assistant",
            content=mcp_content,
            model=response.model_id or "unknown",
        )

    async def logging_callback(self, params: types.LoggingMessageNotificationParams) -> None:
        """ログ用のコールバック関数。

        MCPサーバーがログメッセージを送信するときに呼び出されます。
        デフォルトでは、paramsで設定されたレベルでロガーにメッセージを記録します。

        注意:
            MCPToolをサブクラス化し、この関数をオーバーライドして動作を適応可能です。

        Args:
            params: MCPサーバーからのログメッセージ通知のパラメータ。

        """
        logger.log(LOG_LEVEL_MAPPING[params.level], params.data)

    async def message_handler(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
    ) -> None:
        """MCPサーバーからのメッセージを処理します。

        デフォルトでは、この関数はサーバー上の例外をログに記録して処理し、
        リスト変更通知を受け取った際にツールとプロンプトのリロードをトリガーします。

        注意:
            この動作を拡張したい場合は、MCPToolをサブクラス化して
            この関数をオーバーライドしてください。デフォルトの動作を保持したい場合は、
            ``super().message_handler(message)``を呼び出すようにしてください。

        引数:
            message: MCPサーバーからのメッセージ（リクエストレスポンダー、通知、または例外）。

        """
        if isinstance(message, Exception):
            logger.error("Error from MCP server: %s", message, exc_info=message)
            return
        if isinstance(message, types.ServerNotification):
            match message.root.method:
                case "notifications/tools/list_changed":
                    await self.load_tools()
                case "notifications/prompts/list_changed":
                    await self.load_prompts()
                case _:
                    logger.debug("Unhandled notification: %s", message.root.method)

    def _determine_approval_mode(
        self,
        local_name: str,
    ) -> Literal["always_require", "never_require"] | None:
        if isinstance(self.approval_mode, dict):
            if (always_require := self.approval_mode.get("always_require_approval")) and local_name in always_require:
                return "always_require"
            if (never_require := self.approval_mode.get("never_require_approval")) and local_name in never_require:
                return "never_require"
            return None
        return self.approval_mode  # type: ignore[reportReturnType]

    async def load_prompts(self) -> None:
        """MCPサーバーからプロンプトをロードします。

        接続されたMCPサーバーから利用可能なプロンプトを取得し、
        それらをAIFunctionインスタンスに変換します。

        例外:
            ToolExecutionException: MCPサーバーに接続されていない場合。

        """
        if not self.session:
            raise ToolExecutionException("MCP server not connected, please call connect() before using this method.")
        try:
            prompt_list = await self.session.list_prompts()
        except Exception as exc:
            logger.info(
                "Prompt could not be loaded, you can exclude trying to load, by setting: load_prompts=False",
                exc_info=exc,
            )
            prompt_list = None
        for prompt in prompt_list.prompts if prompt_list else []:
            local_name = _normalize_mcp_name(prompt.name)
            input_model = _get_input_model_from_mcp_prompt(prompt)
            approval_mode = self._determine_approval_mode(local_name)
            func: AIFunction[BaseModel, list[ChatMessage]] = AIFunction(
                func=partial(self.get_prompt, prompt.name),
                name=local_name,
                description=prompt.description or "",
                approval_mode=approval_mode,
                input_model=input_model,
            )
            self._functions.append(func)

    async def load_tools(self) -> None:
        """MCPサーバーからツールをロードします。

        接続されたMCPサーバーから利用可能なツールを取得し、
        それらをAIFunctionインスタンスに変換します。

        例外:
            ToolExecutionException: MCPサーバーに接続されていない場合。

        """
        if not self.session:
            raise ToolExecutionException("MCP server not connected, please call connect() before using this method.")
        try:
            tool_list = await self.session.list_tools()
        except Exception as exc:
            logger.info(
                "Tools could not be loaded, you can exclude trying to load, by setting: load_tools=False",
                exc_info=exc,
            )
            tool_list = None
        for tool in tool_list.tools if tool_list else []:
            local_name = _normalize_mcp_name(tool.name)
            input_model = _get_input_model_from_mcp_tool(tool)
            approval_mode = self._determine_approval_mode(local_name)
            # 各ツールからAIFunctionを作成します
            func: AIFunction[BaseModel, list[Contents]] = AIFunction(
                func=partial(self.call_tool, tool.name),
                name=local_name,
                description=tool.description or "",
                approval_mode=approval_mode,
                input_model=input_model,
            )
            self._functions.append(func)

    async def close(self) -> None:
        """MCPサーバーから切断します。

        接続を閉じてリソースをクリーンアップします。

        """
        await self._exit_stack.aclose()
        self.session = None
        self.is_connected = False

    @abstractmethod
    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """MCPクライアントを取得します。

        戻り値:
            MCPクライアントトランスポートの非同期コンテキストマネージャ。

        """
        pass

    async def call_tool(self, tool_name: str, **kwargs: Any) -> list[Contents]:
        """指定された引数でツールを呼び出します。

        引数:
            tool_name: 呼び出すツールの名前。

        キーワード引数:
            kwargs: ツールに渡す引数。

        戻り値:
            ツールから返されたコンテンツアイテムのリスト。

        例外:
            ToolExecutionException: MCPサーバーに接続されていない、ツールがロードされていない、
                またはツール呼び出しが失敗した場合。

        """
        if not self.session:
            raise ToolExecutionException("MCP server not connected, please call connect() before using this method.")
        if not self.load_tools_flag:
            raise ToolExecutionException(
                "Tools are not loaded for this server, please set load_tools=True in the constructor."
            )
        try:
            return _mcp_call_tool_result_to_ai_contents(await self.session.call_tool(tool_name, arguments=kwargs))
        except McpError as mcp_exc:
            raise ToolExecutionException(mcp_exc.error.message, inner_exception=mcp_exc) from mcp_exc
        except Exception as ex:
            raise ToolExecutionException(f"Failed to call tool '{tool_name}'.", inner_exception=ex) from ex

    async def get_prompt(self, prompt_name: str, **kwargs: Any) -> list[ChatMessage]:
        """指定された引数でプロンプトを呼び出します。

        引数:
            prompt_name: 取得するプロンプトの名前。

        キーワード引数:
            kwargs: プロンプトに渡す引数。

        戻り値:
            プロンプトから返されたチャットメッセージのリスト。

        例外:
            ToolExecutionException: MCPサーバーに接続されていない、プロンプトがロードされていない、
                またはプロンプト呼び出しが失敗した場合。

        """
        if not self.session:
            raise ToolExecutionException("MCP server not connected, please call connect() before using this method.")
        if not self.load_prompts_flag:
            raise ToolExecutionException(
                "Prompts are not loaded for this server, please set load_prompts=True in the constructor."
            )
        try:
            prompt_result = await self.session.get_prompt(prompt_name, arguments=kwargs)
            return [_mcp_prompt_message_to_chat_message(message) for message in prompt_result.messages]
        except McpError as mcp_exc:
            raise ToolExecutionException(mcp_exc.error.message, inner_exception=mcp_exc) from mcp_exc
        except Exception as ex:
            raise ToolExecutionException(f"Failed to call prompt '{prompt_name}'.", inner_exception=ex) from ex

    async def __aenter__(self) -> Self:
        """非同期コンテキストマネージャに入ります。

        MCPサーバーに自動的に接続します。

        戻り値:
            MCPToolインスタンス。

        例外:
            ToolException: 接続に失敗した場合。
            ToolExecutionException: コンテキストマネージャのセットアップに失敗した場合。

        """
        try:
            await self.connect()
            return self
        except ToolException:
            raise
        except Exception as ex:
            await self._exit_stack.aclose()
            raise ToolExecutionException("Failed to enter context manager.", inner_exception=ex) from ex

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> None:
        """非同期コンテキストマネージャを終了します。

        接続を閉じてリソースをクリーンアップします。

        引数:
            exc_type: 例外が発生した場合の例外タイプ、そうでなければNone。
            exc_value: 例外が発生した場合の例外値、そうでなければNone。
            traceback: 例外が発生した場合のトレースバック、そうでなければNone。

        """
        await self.close()


# region: MCP Plugin Implementations


class MCPStdioTool(MCPTool):
    """stdioベースのMCPサーバーに接続するためのMCPツール。

    このクラスは標準入力/出力を介して通信するMCPサーバーに接続します。
    通常はローカルプロセスで使用されます。

    Examples:
        .. code-block:: python

            from agent_framework import MCPStdioTool, ChatAgent

            # MCP stdioツールを作成
            mcp_tool = MCPStdioTool(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                description="File system operations",
            )

            # チャットエージェントと共に使用
            async with mcp_tool:
                agent = ChatAgent(chat_client=client, name="assistant", tools=mcp_tool)
                response = await agent.run("List files in the directory")

    """

    def __init__(
        self,
        name: str,
        command: str,
        *,
        load_tools: bool = True,
        load_prompts: bool = True,
        request_timeout: int | None = None,
        session: ClientSession | None = None,
        description: str | None = None,
        approval_mode: Literal["always_require", "never_require"] | HostedMCPSpecificApproval | None = None,
        allowed_tools: Collection[str] | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        encoding: str | None = None,
        chat_client: "ChatClientProtocol | None" = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """MCP stdioツールを初期化します。

        注意:
            引数はStdioServerParametersオブジェクトを作成するために使用され、
            それを用いてstdioクライアントが作成されます。
            詳細は``mcp.client.stdio.stdio_client``および
            ``mcp.client.stdio.stdio_server_parameters``を参照してください。

        引数:
            name: ツールの名前。
            command: MCPサーバーを実行するコマンド。

        キーワード引数:
            load_tools: MCPサーバーからツールをロードするかどうか。
            load_prompts: MCPサーバーからプロンプトをロードするかどうか。
            request_timeout: すべてのリクエストのデフォルトタイムアウト（秒）。
            session: MCP接続に使用するセッション。
            description: ツールの説明。
            approval_mode: ツールの承認モード。以下のいずれか:
                - "always_require": ツールは常に使用前に承認が必要。
                - "never_require": ツールは使用前に承認を必要としない。
                - `always_require_approval`または`never_require_approval`キーを持つ辞書で、
                  関連ツールの名前のシーケンスを指定。
                ツールは両方にリストされるべきではなく、もしそうなら承認が必要になります。
            allowed_tools: このツールの使用を許可されたツールのリスト。
            additional_properties: 追加のプロパティ。
            args: コマンドに渡す引数。
            env: コマンドの環境変数。
            encoding: コマンド出力に使用するエンコーディング。
            chat_client: サンプリングに使用するチャットクライアント。
            kwargs: stdioクライアントに渡す追加の引数。

        """
        super().__init__(
            name=name,
            description=description,
            approval_mode=approval_mode,
            allowed_tools=allowed_tools,
            additional_properties=additional_properties,
            session=session,
            chat_client=chat_client,
            load_tools=load_tools,
            load_prompts=load_prompts,
            request_timeout=request_timeout,
        )
        self.command = command
        self.args = args or []
        self.env = env
        self.encoding = encoding
        self._client_kwargs = kwargs

    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """MCP stdioクライアントを取得します。

        戻り値:
            stdioクライアントトランスポートの非同期コンテキストマネージャ。

        """
        args: dict[str, Any] = {
            "command": self.command,
            "args": self.args,
            "env": self.env,
        }
        if self.encoding:
            args["encoding"] = self.encoding
        if self._client_kwargs:
            args.update(self._client_kwargs)
        return stdio_client(server=StdioServerParameters(**args))


class MCPStreamableHTTPTool(MCPTool):
    """HTTPベースのMCPサーバーに接続するためのMCPツール。

    このクラスはストリーム可能なHTTP/SSEを介して通信するMCPサーバーに接続します。

    Examples:
        .. code-block:: python

            from agent_framework import MCPStreamableHTTPTool, ChatAgent

            # MCP HTTPツールを作成
            mcp_tool = MCPStreamableHTTPTool(
                name="web-api",
                url="https://api.example.com/mcp",
                headers={"Authorization": "Bearer token"},
                description="Web API operations",
            )

            # チャットエージェントと共に使用
            async with mcp_tool:
                agent = ChatAgent(chat_client=client, name="assistant", tools=mcp_tool)
                response = await agent.run("Fetch data from the API")

    """

    def __init__(
        self,
        name: str,
        url: str,
        *,
        load_tools: bool = True,
        load_prompts: bool = True,
        request_timeout: int | None = None,
        session: ClientSession | None = None,
        description: str | None = None,
        approval_mode: Literal["always_require", "never_require"] | HostedMCPSpecificApproval | None = None,
        allowed_tools: Collection[str] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | None = None,
        sse_read_timeout: float | None = None,
        terminate_on_close: bool | None = None,
        chat_client: "ChatClientProtocol | None" = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """MCPストリーム可能HTTPツールを初期化します。

        注意:
            引数はストリーム可能なHTTPクライアントを作成するために使用されます。
            詳細は``mcp.client.streamable_http.streamablehttp_client``を参照してください。
            コンストラクタに渡された追加の引数はすべて
            ストリーム可能HTTPクライアントのコンストラクタに渡されます。

        引数:
            name: ツールの名前。
            url: MCPサーバーのURL。

        キーワード引数:
            load_tools: MCPサーバーからツールをロードするかどうか。
            load_prompts: MCPサーバーからプロンプトをロードするかどうか。
            request_timeout: すべてのリクエストのデフォルトタイムアウト（秒）。
            session: MCP接続に使用するセッション。
            description: ツールの説明。
            approval_mode: ツールの承認モード。以下のいずれか:
                - "always_require": ツールは常に使用前に承認が必要。
                - "never_require": ツールは使用前に承認を必要としない。
                - `always_require_approval`または`never_require_approval`キーを持つ辞書で、
                  関連ツールの名前のシーケンスを指定。
                ツールは両方にリストされるべきではなく、もしそうなら承認が必要になります。
            allowed_tools: このツールの使用を許可されたツールのリスト。
            additional_properties: 追加のプロパティ。
            headers: リクエストに送信するヘッダー。
            timeout: リクエストのタイムアウト。
            sse_read_timeout: SSEストリームの読み取りタイムアウト。
            terminate_on_close: MCPクライアント終了時にトランスポートを閉じるかどうか。
            chat_client: サンプリングに使用するチャットクライアント。
            kwargs: SSEクライアントに渡す追加の引数。

        """
        super().__init__(
            name=name,
            description=description,
            approval_mode=approval_mode,
            allowed_tools=allowed_tools,
            additional_properties=additional_properties,
            session=session,
            chat_client=chat_client,
            load_tools=load_tools,
            load_prompts=load_prompts,
            request_timeout=request_timeout,
        )
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.terminate_on_close = terminate_on_close
        self._client_kwargs = kwargs

    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """MCPストリーム可能HTTPクライアントを取得します。

        戻り値:
            ストリーム可能HTTPクライアントトランスポートの非同期コンテキストマネージャ。

        """
        args: dict[str, Any] = {
            "url": self.url,
        }
        if self.headers:
            args["headers"] = self.headers
        if self.timeout is not None:
            args["timeout"] = self.timeout
        if self.sse_read_timeout is not None:
            args["sse_read_timeout"] = self.sse_read_timeout
        if self.terminate_on_close is not None:
            args["terminate_on_close"] = self.terminate_on_close
        if self._client_kwargs:
            args.update(self._client_kwargs)
        return streamablehttp_client(**args)


class MCPWebsocketTool(MCPTool):
    """WebSocketベースのMCPサーバーに接続するためのMCPツール。

    このクラスはWebSocketを介して通信するMCPサーバーに接続します。

    Examples:
        .. code-block:: python

            from agent_framework import MCPWebsocketTool, ChatAgent

            # MCP WebSocketツールを作成
            mcp_tool = MCPWebsocketTool(
                name="realtime-service", url="wss://service.example.com/mcp", description="Real-time service operations"
            )

            # チャットエージェントと共に使用
            async with mcp_tool:
                agent = ChatAgent(chat_client=client, name="assistant", tools=mcp_tool)
                response = await agent.run("Connect to the real-time service")

    """

    def __init__(
        self,
        name: str,
        url: str,
        *,
        load_tools: bool = True,
        load_prompts: bool = True,
        request_timeout: int | None = None,
        session: ClientSession | None = None,
        description: str | None = None,
        approval_mode: Literal["always_require", "never_require"] | HostedMCPSpecificApproval | None = None,
        allowed_tools: Collection[str] | None = None,
        chat_client: "ChatClientProtocol | None" = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """MCP WebSocketツールを初期化します。

        注意:
            引数はWebSocketクライアントを作成するために使用されます。
            詳細は``mcp.client.websocket.websocket_client``を参照してください。
            コンストラクタに渡された追加の引数はすべて
            WebSocketクライアントのコンストラクタに渡されます。

        引数:
            name: ツールの名前。
            url: MCPサーバーのURL。

        キーワード引数:
            load_tools: MCPサーバーからツールをロードするかどうか。
            load_prompts: MCPサーバーからプロンプトをロードするかどうか。
            request_timeout: すべてのリクエストのデフォルトタイムアウト（秒）。
            session: MCP接続に使用するセッション。
            description: ツールの説明。
            approval_mode: ツールの承認モード。以下のいずれか:
                - "always_require": ツールは常に使用前に承認が必要。
                - "never_require": ツールは使用前に承認を必要としない。
                - `always_require_approval`または`never_require_approval`キーを持つ辞書で、
                  関連ツールの名前のシーケンスを指定。
                ツールは両方にリストされるべきではなく、もしそうなら承認が必要になります。
            allowed_tools: このツールの使用を許可されたツールのリスト。
            additional_properties: 追加のプロパティ。
            chat_client: サンプリングに使用するチャットクライアント。
            kwargs: WebSocketクライアントに渡す追加の引数。

        """
        super().__init__(
            name=name,
            description=description,
            approval_mode=approval_mode,
            allowed_tools=allowed_tools,
            additional_properties=additional_properties,
            session=session,
            chat_client=chat_client,
            load_tools=load_tools,
            load_prompts=load_prompts,
            request_timeout=request_timeout,
        )
        self.url = url
        self._client_kwargs = kwargs

    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """MCP WebSocketクライアントを取得します。

        戻り値:
            WebSocketクライアントトランスポートの非同期コンテキストマネージャ。

        """
        args: dict[str, Any] = {
            "url": self.url,
        }
        if self._client_kwargs:
            args.update(self._client_kwargs)
        return websocket_client(**args)
