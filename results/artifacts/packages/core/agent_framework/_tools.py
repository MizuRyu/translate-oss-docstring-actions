# Copyright (c) Microsoft. All rights reserved.

import asyncio
import inspect
import json
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, Collection, Mapping, MutableMapping, Sequence
from functools import wraps
from time import perf_counter, time_ns
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

from opentelemetry.metrics import Histogram
from pydantic import AnyUrl, BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

from ._logging import get_logger
from ._serialization import SerializationMixin
from .exceptions import ChatClientInitializationError, ToolException
from .observability import (
    OPERATION_DURATION_BUCKET_BOUNDARIES,
    OtelAttr,
    capture_exception,  # type: ignore
    get_function_span,
    get_function_span_attributes,
    get_meter,
)

if TYPE_CHECKING:
    from ._clients import ChatClientProtocol
    from ._types import (
        ChatMessage,
        ChatResponse,
        ChatResponseUpdate,
        Contents,
        FunctionApprovalResponseContent,
        FunctionCallContent,
    )

if sys.version_info >= (3, 12):
    from typing import (
        TypedDict,  # pragma: no cover
        override,  # type: ignore # pragma: no cover
    )
else:
    from typing_extensions import (
        TypedDict,  # pragma: no cover
        override,  # type: ignore[import] # pragma: no cover
    )

if sys.version_info >= (3, 11):
    from typing import overload  # pragma: no cover
else:
    from typing_extensions import overload  # pragma: no cover

logger = get_logger()

__all__ = [
    "FUNCTION_INVOKING_CHAT_CLIENT_MARKER",
    "AIFunction",
    "HostedCodeInterpreterTool",
    "HostedFileSearchTool",
    "HostedMCPSpecificApproval",
    "HostedMCPTool",
    "HostedWebSearchTool",
    "ToolProtocol",
    "ai_function",
    "use_function_invocation",
]


logger = get_logger()
FUNCTION_INVOKING_CHAT_CLIENT_MARKER: Final[str] = "__function_invoking_chat_client__"
DEFAULT_MAX_ITERATIONS: Final[int] = 10
TChatClient = TypeVar("TChatClient", bound="ChatClientProtocol")
# region Helpers

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")


class _NoOpHistogram:
    def record(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None


_NOOP_HISTOGRAM = _NoOpHistogram()


def _parse_inputs(
    inputs: "Contents | dict[str, Any] | str | list[Contents | dict[str, Any] | str] | None",
) -> list["Contents"]:
    """ツールの入力を解析し、Contents型であることを保証します。

    Args:
        inputs: 解析する入力。単一のContentsまたはContentsのリスト、辞書、文字列のいずれか。

    Returns:
        Contentsオブジェクトのリスト。

    Raises:
        ValueError: サポートされていない入力タイプが検出された場合。
        TypeError: 入力タイプがサポートされていない場合。

    """
    if inputs is None:
        return []

    from ._types import BaseContent, DataContent, HostedFileContent, HostedVectorStoreContent, UriContent

    parsed_inputs: list["Contents"] = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    for input_item in inputs:
        if isinstance(input_item, str):
            # 文字列の場合、それがURIまたは類似の識別子であると仮定します。 必要に応じてUriContentまたは類似の型に変換します。
            parsed_inputs.append(UriContent(uri=input_item, media_type="text/plain"))
        elif isinstance(input_item, dict):
            # 辞書の場合、特定のコンテンツタイプのプロパティを含むと仮定します。 必要なキーが存在するかをチェックしてタイプを判定します。
            # 例えば、"uri"と"media_type"があればUriContentとして扱います。 uriのみの場合はDataContentとして扱います。
            # など。
            if "uri" in input_item:
                parsed_inputs.append(
                    UriContent(**input_item) if "media_type" in input_item else DataContent(**input_item)
                )
            elif "file_id" in input_item:
                parsed_inputs.append(HostedFileContent(**input_item))
            elif "vector_store_id" in input_item:
                parsed_inputs.append(HostedVectorStoreContent(**input_item))
            elif "data" in input_item:
                parsed_inputs.append(DataContent(**input_item))
            else:
                raise ValueError(f"Unsupported input type: {input_item}")
        elif isinstance(input_item, BaseContent):
            parsed_inputs.append(input_item)
        else:
            raise TypeError(f"Unsupported input type: {type(input_item).__name__}. Expected Contents or dict.")
    return parsed_inputs


# region Tools
@runtime_checkable
class ToolProtocol(Protocol):
    """AIサービスに指定できる汎用ツールを表します。

    このプロトコルは、すべてのツールがエージェントフレームワークと互換性を持つために実装しなければならないインターフェースを定義します。

    属性:
    name: ツールの名前。
    description: モデルに目的を説明するのに適したツールの説明。
    additional_properties: ツールに関連付けられた追加のプロパティ。

    Examples:
    .. code-block:: python

        from agent_framework import ToolProtocol

        class CustomTool:
            def __init__(self, name: str, description: str) -> None:
                self.name = name
                self.description = description
                self.additional_properties = None

            def __str__(self) -> str:
                return f"CustomTool(name={self.name})"

        # Tool now implements ToolProtocol
        tool: ToolProtocol = CustomTool("my_tool", "Does something useful")
    """

    name: str
    """The name of the tool."""
    description: str
    """A description of the tool, suitable for use in describing the purpose to a model."""
    additional_properties: dict[str, Any] | None
    """Additional properties associated with the tool."""

    def __str__(self) -> str:
        """ツールの文字列表現を返します。"""
        ...


class BaseTool(SerializationMixin):
    """AIツールの基底クラスで、共通の属性とメソッドを提供します。

    このクラスは、シリアライズ対応のカスタムツールを作成するための基盤を提供します。

    Examples:
    .. code-block:: python

        from agent_framework import BaseTool

        class MyCustomTool(BaseTool):
            def __init__(self, name: str, custom_param: str) -> None:
                super().__init__(name=name, description="My custom tool")
                self.custom_param = custom_param

        tool = MyCustomTool(name="custom", custom_param="value")
        print(tool)  # MyCustomTool(name=custom, description=My custom tool)
    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"additional_properties"}

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """BaseToolを初期化します。

        Keyword Args:
        name: ツールの名前。
        description: ツールの説明。
        additional_properties: ツールに関連付けられた追加のプロパティ。
        **kwargs: 追加のキーワード引数。
        """
        self.name = name
        self.description = description
        self.additional_properties = additional_properties
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        """ツールの文字列表現を返します。"""
        if self.description:
            return f"{self.__class__.__name__}(name={self.name}, description={self.description})"
        return f"{self.__class__.__name__}(name={self.name})"


class HostedCodeInterpreterTool(BaseTool):
    """生成されたコードを実行できるようにAIサービスに指定できるホスト型ツールを表します。

    このツール自体はコードの解釈を実装しません。サービスが生成されたコードを実行可能な場合に実行を許可するためのマーカーとして機能します。

    Examples:
    .. code-block:: python

        from agent_framework import HostedCodeInterpreterTool

        # コードインタプリターツールを作成
        code_tool = HostedCodeInterpreterTool()

        # ファイル入力付き
        code_tool_with_files = HostedCodeInterpreterTool(inputs=[{"file_id": "file-123"}, {"file_id": "file-456"}])
    """

    def __init__(
        self,
        *,
        inputs: "Contents | dict[str, Any] | str | list[Contents | dict[str, Any] | str] | None" = None,
        description: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """HostedCodeInterpreterToolを初期化します。

        Keyword Args:
        inputs: ツールが入力として受け入れ可能なコンテンツのリスト。デフォルトはNone。
        これは主にHostedFileContentまたはHostedVectorStoreContentであるべきです。
        使用するサービスによってはDataContentも可能です。
        リストを指定する場合、以下を含むことができます:
        - Contentsのインスタンス
        - Contentsのプロパティを持つdict（例: {"uri": "http://example.com", "media_type": "text/html"}）
        - 文字列（media_type "text/plain"のUriContentに変換されます）。
        Noneの場合は空リストがデフォルトです。
        description: ツールの説明。
        additional_properties: ツールに関連付けられた追加のプロパティ。
        **kwargs: 基底クラスに渡す追加のキーワード引数。
        """
        if "name" in kwargs:
            raise ValueError("The 'name' argument is reserved for the HostedCodeInterpreterTool and cannot be set.")

        self.inputs = _parse_inputs(inputs) if inputs else []

        super().__init__(
            name="code_interpreter",
            description=description or "",
            additional_properties=additional_properties,
            **kwargs,
        )


class HostedWebSearchTool(BaseTool):
    """AIサービスに指定できるウェブ検索ツールを表します。

    Examples:
    .. code-block:: python

        from agent_framework import HostedWebSearchTool

        # 基本的なウェブ検索ツールを作成
        search_tool = HostedWebSearchTool()

        # ロケーションコンテキスト付き
        search_tool_with_location = HostedWebSearchTool(
            description="Search the web for information",
            additional_properties={"user_location": {"city": "Seattle", "country": "US"}},
        )
    """

    def __init__(
        self,
        description: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """HostedWebSearchToolを初期化します。

        Keyword Args:
        description: ツールの説明。
        additional_properties: ツールに関連付けられた追加のプロパティ
        （例: {"user_location": {"city": "Seattle", "country": "US"}}）。
        **kwargs: 基底クラスに渡す追加のキーワード引数。
        additional_propertiesが提供されていない場合、kwargsはadditional_propertiesに追加されます。
        """
        args: dict[str, Any] = {
            "name": "web_search",
        }
        if additional_properties is not None:
            args["additional_properties"] = additional_properties
        elif kwargs:
            args["additional_properties"] = kwargs
        if description is not None:
            args["description"] = description
        super().__init__(**args)


class HostedMCPSpecificApproval(TypedDict, total=False):
    """ホスト型ツールの特定のモードを表します。

    このモードを使用する場合、ユーザーは常に承認が必要なツールと決して承認が不要なツールを指定する必要があります。
    これは2つのオプショナルキーを持つ辞書として表されます。

    属性:
    always_require_approval: 常に承認が必要なツール名のシーケンス。
    never_require_approval: 決して承認が不要なツール名のシーケンス。
    """

    always_require_approval: Collection[str] | None
    never_require_approval: Collection[str] | None


class HostedMCPTool(BaseTool):
    """サービスによって管理および実行されるMCPツールを表します。

    Examples:
    .. code-block:: python

        from agent_framework import HostedMCPTool

        # 基本的なMCPツールを作成
        mcp_tool = HostedMCPTool(
            name="my_mcp_tool",
            url="https://example.com/mcp",
        )

        # 承認モードと許可されたツール付き
        mcp_tool_with_approval = HostedMCPTool(
            name="my_mcp_tool",
            description="My MCP tool",
            url="https://example.com/mcp",
            approval_mode="always_require",
            allowed_tools=["tool1", "tool2"],
            headers={"Authorization": "Bearer token"},
        )

        # 特定の承認モード付き
        mcp_tool_specific = HostedMCPTool(
            name="my_mcp_tool",
            url="https://example.com/mcp",
            approval_mode={
                "always_require_approval": ["dangerous_tool"],
                "never_require_approval": ["safe_tool"],
            },
        )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        url: AnyUrl | str,
        approval_mode: Literal["always_require", "never_require"] | HostedMCPSpecificApproval | None = None,
        allowed_tools: Collection[str] | None = None,
        headers: dict[str, str] | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """ホスト型MCPツールを作成します。

        Keyword Args:
        name: ツールの名前。
        description: ツールの説明。
        url: ツールのURL。
        approval_mode: ツールの承認モード。以下のいずれかです:
        - "always_require": ツールは使用前に常に承認が必要。
        - "never_require": ツールは使用前に承認が不要。
        - `always_require_approval`または`never_require_approval`キーを持つ辞書で、
          関連するツール名のシーケンスを指定。
        allowed_tools: このツールの使用を許可されたツールのリスト。
        headers: ツールへのリクエストに含めるヘッダー。
        additional_properties: ツール定義に含める追加のプロパティ。
        **kwargs: 基底クラスに渡す追加のキーワード引数。
        """
        try:
            # approval_modeを検証します
            if approval_mode is not None:
                if isinstance(approval_mode, str):
                    if approval_mode not in ("always_require", "never_require"):
                        raise ValueError(
                            f"Invalid approval_mode: {approval_mode}. "
                            "Must be 'always_require', 'never_require', or a dict with 'always_require_approval' "
                            "or 'never_require_approval' keys."
                        )
                elif isinstance(approval_mode, dict):
                    # 辞書がセットを持つことを検証します
                    for key, value in approval_mode.items():
                        if not isinstance(value, set):
                            approval_mode[key] = set(value)  # type: ignore

            # allowed_toolsを検証します
            if allowed_tools is not None and isinstance(allowed_tools, dict):
                raise TypeError(
                    f"allowed_tools must be a sequence of strings, not a dict. Got: {type(allowed_tools).__name__}"
                )

            super().__init__(
                name=name,
                description=description or "",
                additional_properties=additional_properties,
                **kwargs,
            )
            self.url = url if isinstance(url, AnyUrl) else AnyUrl(url)
            self.approval_mode = approval_mode
            self.allowed_tools = set(allowed_tools) if allowed_tools else None
            self.headers = headers
        except (ValidationError, ValueError, TypeError) as err:
            raise ToolException(f"Error initializing HostedMCPTool: {err}", inner_exception=err) from err


class HostedFileSearchTool(BaseTool):
    """AIサービスに指定できるファイル検索ツールを表します。

    Examples:
    .. code-block:: python

        from agent_framework import HostedFileSearchTool

        # 基本的なファイル検索ツールを作成
        file_search = HostedFileSearchTool()

        # ベクターストア入力と最大結果数付き
        file_search_with_inputs = HostedFileSearchTool(
            inputs=[{"vector_store_id": "vs_123"}],
            max_results=10,
            description="Search files in vector store",
        )
    """

    def __init__(
        self,
        *,
        inputs: "Contents | dict[str, Any] | str | list[Contents | dict[str, Any] | str] | None" = None,
        max_results: int | None = None,
        description: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """FileSearchToolを初期化します。

        Keyword Args:
        inputs: ツールが入力として受け入れ可能なコンテンツのリスト。デフォルトはNone。
        これは1つ以上のHostedVectorStoreContentsであるべきです。
        リストを指定する場合、以下を含むことができます:
        - Contentsのインスタンス
        - Contentsのプロパティを持つdict（例: {"uri": "http://example.com", "media_type": "text/html"}）
        - 文字列（media_type "text/plain"のUriContentに変換されます）。
        Noneの場合は空リストがデフォルトです。
        max_results: ファイル検索で返す最大結果数。
        Noneの場合は最大制限が適用されます。
        description: ツールの説明。
        additional_properties: ツールに関連付けられた追加のプロパティ。
        **kwargs: 基底クラスに渡す追加のキーワード引数。
        """
        if "name" in kwargs:
            raise ValueError("The 'name' argument is reserved for the HostedFileSearchTool and cannot be set.")

        self.inputs = _parse_inputs(inputs) if inputs else None
        self.max_results = max_results

        super().__init__(
            name="file_search",
            description=description or "",
            additional_properties=additional_properties,
            **kwargs,
        )


def _default_histogram() -> Histogram:
    """関数呼び出しの継続時間のデフォルトヒストグラムを取得します。

    Returns:
    関数呼び出しの継続時間を記録するHistogramインスタンス、
    または観測性が無効な場合はno-opヒストグラム。
    """
    from .observability import OBSERVABILITY_SETTINGS  # 循環参照を避けるためのローカルインポート

    if not OBSERVABILITY_SETTINGS.ENABLED:  # type: ignore[name-defined]
        return _NOOP_HISTOGRAM  # type: ignore[return-value]
    meter = get_meter()
    try:
        return meter.create_histogram(
            name=OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION,
            unit=OtelAttr.DURATION_UNIT,
            description="Measures the duration of a function's execution",
            explicit_bucket_boundaries_advisory=OPERATION_DURATION_BUCKET_BOUNDARIES,
        )
    except TypeError:
        return meter.create_histogram(
            name=OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION,
            unit=OtelAttr.DURATION_UNIT,
            description="Measures the duration of a function's execution",
        )


TClass = TypeVar("TClass", bound="SerializationMixin")


class AIFunction(BaseTool, Generic[ArgsT, ReturnT]):
    """Python関数をラップしてAIモデルから呼び出し可能にするツールです。

    このクラスはPython関数をラップし、パラメータの自動検証とJSONスキーマ生成を行い、AIモデルから呼び出し可能にします。

    Examples:
    .. code-block:: python

        from typing import Annotated
        from pydantic import BaseModel, Field
        from agent_framework import AIFunction, ai_function

        # 文字列アノテーションを使ったデコレータの使用例
        @ai_function
        def get_weather(
            location: Annotated[str, "The city name"],
            unit: Annotated[str, "Temperature unit"] = "celsius",
        ) -> str:
            '''Get the weather for a location.'''
            return f"Weather in {location}: 22°{unit[0].upper()}"

        # Fieldを使った直接インスタンス化の例
        class WeatherArgs(BaseModel):
            location: Annotated[str, Field(description="The city name")]
            unit: Annotated[str, Field(description="Temperature unit")] = "celsius"

        weather_func = AIFunction(
            name="get_weather",
            description="Get the weather for a location",
            func=lambda location, unit="celsius": f"Weather in {location}: 22°{unit[0].upper()}",
            approval_mode="never_require",
            input_model=WeatherArgs,
        )

        # 関数を呼び出す
        result = await weather_func.invoke(arguments=WeatherArgs(location="Seattle"))
    """

    INJECTABLE: ClassVar[set[str]] = {"func"}
    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"input_model", "_invocation_duration_histogram"}

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        approval_mode: Literal["always_require", "never_require"] | None = None,
        additional_properties: dict[str, Any] | None = None,
        func: Callable[..., Awaitable[ReturnT] | ReturnT],
        input_model: type[ArgsT] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """AIFunctionを初期化します。

        キーワード引数:
            name: 関数の名前。
            description: 関数の説明。
            approval_mode: このツールを実行するために承認が必要かどうか。
                デフォルトは承認不要です。
            additional_properties: 関数に設定する追加のプロパティ。
            func: ラップする関数。
            input_model: 関数の入力パラメータを定義するPydanticモデル。
                これはJSONスキーマの辞書でも構いません。
                指定しない場合は関数のシグネチャから推論されます。
            **kwargs: 追加のキーワード引数。

        """
        super().__init__(
            name=name,
            description=description,
            additional_properties=additional_properties,
            **kwargs,
        )
        self.func = func
        self.input_model = self._resolve_input_model(input_model)
        self.approval_mode = approval_mode or "never_require"
        self._invocation_duration_histogram = _default_histogram()
        self.type: Literal["ai_function"] = "ai_function"

    def _resolve_input_model(self, input_model: type[ArgsT] | Mapping[str, Any] | None) -> type[ArgsT]:
        if input_model:
            if inspect.isclass(input_model) and issubclass(input_model, BaseModel):
                return input_model
            if isinstance(input_model, Mapping):
                return cast(type[ArgsT], _create_model_from_json_schema(self.name, input_model))
            raise TypeError("input_model must be a Pydantic BaseModel subclass or a JSON schema dict.")
        return cast(type[ArgsT], _create_input_model_from_func(self.func, self.name))

    def __call__(self, *args: Any, **kwargs: Any) -> ReturnT | Awaitable[ReturnT]:
        """提供された引数でラップされた関数を呼び出します。"""
        return self.func(*args, **kwargs)

    async def invoke(
        self,
        *,
        arguments: ArgsT | None = None,
        **kwargs: Any,
    ) -> ReturnT:
        """提供された引数をPydanticモデルとして使用してAI関数を実行します。

        キーワード引数:
            arguments: 関数の引数を含むPydanticモデルのインスタンス。
            kwargs: 関数に渡すキーワード引数。``arguments``が提供されている場合は使用されません。

        戻り値:
            関数実行の結果。

        例外:
            TypeError: argumentsが期待される入力モデルのインスタンスでない場合。

        """
        global OBSERVABILITY_SETTINGS
        from .observability import OBSERVABILITY_SETTINGS

        tool_call_id = kwargs.pop("tool_call_id", None)
        if arguments is not None:
            if not isinstance(arguments, self.input_model):
                raise TypeError(f"Expected {self.input_model.__name__}, got {type(arguments).__name__}")
            kwargs = arguments.model_dump(exclude_none=True)
        if not OBSERVABILITY_SETTINGS.ENABLED:  # type: ignore[name-defined]
            logger.info(f"Function name: {self.name}")
            logger.debug(f"Function arguments: {kwargs}")
            res = self.__call__(**kwargs)
            result = await res if inspect.isawaitable(res) else res
            logger.info(f"Function {self.name} succeeded.")
            logger.debug(f"Function result: {result or 'None'}")
            return result  # type: ignore[reportReturnType]

        attributes = get_function_span_attributes(self, tool_call_id=tool_call_id)
        if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED:  # type: ignore[name-defined]
            attributes.update({
                OtelAttr.TOOL_ARGUMENTS: arguments.model_dump_json()
                if arguments
                else json.dumps(kwargs)
                if kwargs
                else "None"
            })
        with get_function_span(attributes=attributes) as span:
            attributes[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] = self.name
            logger.info(f"Function name: {self.name}")
            if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED:  # type: ignore[name-defined]
                logger.debug(f"Function arguments: {kwargs}")
            start_time_stamp = perf_counter()
            end_time_stamp: float | None = None
            try:
                res = self.__call__(**kwargs)
                result = await res if inspect.isawaitable(res) else res
                end_time_stamp = perf_counter()
            except Exception as exception:
                end_time_stamp = perf_counter()
                attributes[OtelAttr.ERROR_TYPE] = type(exception).__name__
                capture_exception(span=span, exception=exception, timestamp=time_ns())
                logger.error(f"Function failed. Error: {exception}")
                raise
            else:
                logger.info(f"Function {self.name} succeeded.")
                if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED:  # type: ignore[name-defined]
                    try:
                        json_result = json.dumps(result)
                    except (TypeError, OverflowError):
                        span.set_attribute(OtelAttr.TOOL_RESULT, "<non-serializable result>")
                        logger.debug("Function result: <non-serializable result>")
                    else:
                        span.set_attribute(OtelAttr.TOOL_RESULT, json_result)
                        logger.debug(f"Function result: {json_result}")
                return result  # type: ignore[reportReturnType]
            finally:
                duration = (end_time_stamp or perf_counter()) - start_time_stamp
                span.set_attribute(OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION, duration)
                self._invocation_duration_histogram.record(duration, attributes=attributes)
                logger.info("Function duration: %fs", duration)

    def parameters(self) -> dict[str, Any]:
        """パラメータのJSONスキーマを作成します。

        戻り値:
            関数のパラメータのJSONスキーマを含む辞書。

        """
        return self.input_model.model_json_schema()

    def to_json_schema_spec(self) -> dict[str, Any]:
        """AIFunctionをJSON Schemaの関数仕様フォーマットに変換します。

        戻り値:
            JSON Schema形式の関数仕様を含む辞書。

        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }

    @override
    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]:
        as_dict = super().to_dict(exclude=exclude, exclude_none=exclude_none)
        if (exclude and "input_model" in exclude) or not self.input_model:
            return as_dict
        as_dict["input_model"] = self.input_model.model_json_schema()
        return as_dict


def _tools_to_dict(
    tools: (
        ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None
    ),
) -> list[str | dict[str, Any]] | None:
    """ツールを辞書に解析します。

    引数:
        tools: 解析するツール。単一のツールまたはツールのシーケンス。

    戻り値:
        辞書形式のツール仕様のリスト、またはツールが提供されていない場合はNone。

    """
    if not tools:
        return None
    if not isinstance(tools, list):
        if isinstance(tools, AIFunction):
            return [tools.to_json_schema_spec()]
        if isinstance(tools, SerializationMixin):
            return [tools.to_dict()]
        if isinstance(tools, dict):
            return [tools]
        if callable(tools):
            return [ai_function(tools).to_json_schema_spec()]
        logger.warning("Can't parse tool.")
        return None
    results: list[str | dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, AIFunction):
            results.append(tool.to_json_schema_spec())
            continue
        if isinstance(tool, SerializationMixin):
            results.append(tool.to_dict())
            continue
        if isinstance(tool, dict):
            results.append(tool)
            continue
        if callable(tool):
            results.append(ai_function(tool).to_json_schema_spec())
            continue
        logger.warning("Can't parse tool.")
    return results


# region AI Function Decorator


def _parse_annotation(annotation: Any) -> Any:
    """型注釈を解析し、対応する型を返します。

    2番目の注釈（型の後）が文字列の場合、それをPydanticのFieldの説明に変換します。
    残りはそのまま返し、複数の注釈を許容します。

    引数:
        annotation: 解析する型注釈。

    戻り値:
        FieldでラップされたAnnotatedの可能性がある解析済み注釈。

    """
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        # 他のジェネリック型の場合は、元の型を返します（例: List[int]ならlist）。
        if len(args) > 1 and isinstance(args[1], str):
            # 更新されたFieldで新しいAnnotated型を作成します。
            args_list = list(args)
            if len(args_list) == 2:
                return Annotated[args_list[0], Field(description=args_list[1])]
            return Annotated[args_list[0], Field(description=args_list[1]), tuple(args_list[2:])]
    return annotation


def _create_input_model_from_func(func: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    """関数のシグネチャからPydanticモデルを作成します。"""
    sig = inspect.signature(func)
    fields = {
        pname: (
            _parse_annotation(param.annotation) if param.annotation is not inspect.Parameter.empty else str,
            param.default if param.default is not inspect.Parameter.empty else ...,
        )
        for pname, param in sig.parameters.items()
        if pname not in {"self", "cls"}
    }
    return create_model(f"{tool_name}_input", **fields)  # type: ignore[call-overload, no-any-return]


# JSON Schemaの型をPydanticの型にマッピングします。
TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _create_model_from_json_schema(tool_name: str, schema_json: Mapping[str, Any]) -> type[BaseModel]:
    """指定されたJSON SchemaからPydanticモデルを作成します。

    引数:
      tool_name: 作成するモデルの名前。
      schema_json: JSON Schemaの定義。

    戻り値:
      動的に作成されたPydanticモデルのクラス。

    """
    # 'properties'が存在し辞書であることを検証します。
    if "properties" not in schema_json or not isinstance(schema_json["properties"], dict):
        raise ValueError(
            f"JSON schema for tool '{tool_name}' must contain a 'properties' key of type dict. "
            f"Got: {schema_json.get('properties', None)}"
        )
    # 型注釈付きのフィールド定義を抽出します。
    field_definitions: dict[str, tuple[type, FieldInfo]] = {}
    for field_name, field_schema in schema_json["properties"].items():
        field_args: dict[str, Any] = {}
        if (field_description := field_schema.get("description", None)) is not None:
            field_args["description"] = field_description
        if (field_default := field_schema.get("default", None)) is not None:
            field_args["default"] = field_default
        field_type = field_schema.get("type", None)
        if field_type is None:
            raise ValueError(
                f"Missing 'type' for field '{field_name}' in JSON schema. "
                f"Got: {field_schema}, Supported types: {list(TYPE_MAPPING.keys())}"
            )
        python_type = TYPE_MAPPING.get(field_type)
        if python_type is None:
            raise ValueError(
                f"Unsupported type '{field_type}' for field '{field_name}' in JSON schema. "
                f"Got: {field_schema}, Supported types: {list(TYPE_MAPPING.keys())}"
            )
        field_definitions[field_name] = (python_type, Field(**field_args))

    return create_model(f"{tool_name}_input", **field_definitions)  # type: ignore[call-overload, no-any-return]


@overload
def ai_function(
    func: Callable[..., ReturnT | Awaitable[ReturnT]],
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: Literal["always_require", "never_require"] | None = None,
    additional_properties: dict[str, Any] | None = None,
) -> AIFunction[Any, ReturnT]: ...


@overload
def ai_function(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: Literal["always_require", "never_require"] | None = None,
    additional_properties: dict[str, Any] | None = None,
) -> Callable[[Callable[..., ReturnT | Awaitable[ReturnT]]], AIFunction[Any, ReturnT]]: ...


def ai_function(
    func: Callable[..., ReturnT | Awaitable[ReturnT]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    approval_mode: Literal["always_require", "never_require"] | None = None,
    additional_properties: dict[str, Any] | None = None,
) -> AIFunction[Any, ReturnT] | Callable[[Callable[..., ReturnT | Awaitable[ReturnT]]], AIFunction[Any, ReturnT]]:
    """関数をデコレートして、モデルに渡して自動実行可能なAIFunctionに変換します。

    このデコレータは関数のシグネチャからPydanticモデルを作成し、
    関数に渡される引数の検証と関数パラメータのJSONスキーマ生成に使用されます。

    パラメータに説明を追加するには、``typing``の``Annotated``型を使用し、
    2番目の引数に文字列の説明を指定します。より高度な設定にはPydanticの
    ``Field``クラスも使用可能です。

    注意:
        approval_modeが"always_require"に設定されている場合、明示的な承認があるまで関数は実行されません。
        これは自動呼び出しのフローにのみ適用されます。
        また、モデルが複数の関数呼び出しを返し、その中に承認が必要なものと不要なものが混在する場合、
        すべてに対して承認を求めることに注意してください。

    Example:

        .. code-block:: python

            from agent_framework import ai_function
            from typing import Annotated


            @ai_function
            def ai_function_example(
                arg1: Annotated[str, "The first argument"],
                arg2: Annotated[int, "The second argument"],
            ) -> str:
                # 2つの引数を受け取り文字列を返す例関数。
                return f"arg1: {arg1}, arg2: {arg2}"


            # 実行に承認が必要な同じ関数
            @ai_function(approval_mode="always_require")
            def ai_function_example(
                arg1: Annotated[str, "The first argument"],
                arg2: Annotated[int, "The second argument"],
            ) -> str:
                # 2つの引数を受け取り文字列を返す例関数。
                return f"arg1: {arg1}, arg2: {arg2}"


            # カスタム名と説明付き
            @ai_function(name="custom_weather", description="Custom weather function")
            def another_weather_func(location: str) -> str:
                return f"Weather in {location}"


            # 非同期関数もサポート
            @ai_function
            async def async_get_weather(location: str) -> str:
                '''非同期で天気を取得します。'''
                # 非同期処理のシミュレーション
                return f"Weather in {location}"


    """

    def decorator(func: Callable[..., ReturnT | Awaitable[ReturnT]]) -> AIFunction[Any, ReturnT]:
        @wraps(func)
        def wrapper(f: Callable[..., ReturnT | Awaitable[ReturnT]]) -> AIFunction[Any, ReturnT]:
            tool_name: str = name or getattr(f, "__name__", "unknown_function")  # type: ignore[assignment]
            tool_desc: str = description or (f.__doc__ or "")
            return AIFunction[Any, ReturnT](
                name=tool_name,
                description=tool_desc,
                approval_mode=approval_mode,
                additional_properties=additional_properties or {},
                func=f,
            )

        return wrapper(func)

    return decorator(func) if func else decorator


# region Function Invoking Chat Client


async def _auto_invoke_function(
    function_call_content: "FunctionCallContent | FunctionApprovalResponseContent",
    custom_args: dict[str, Any] | None = None,
    *,
    tool_map: dict[str, AIFunction[BaseModel, Any]],
    sequence_index: int | None = None,
    request_index: int | None = None,
    middleware_pipeline: Any = None,  # Optional MiddlewarePipeline
) -> "Contents":
    """Agentから要求された関数呼び出しを実行し、定義されたmiddlewareを適用します。

    引数:
        function_call_content: モデルからの関数呼び出し内容。
        custom_args: 解析済み引数とマージする追加のカスタム引数。

    キーワード引数:
        tool_map: ツール名とAIFunctionインスタンスのマッピング。
        sequence_index: シーケンス内の関数呼び出しのインデックス。
        request_index: リクエストの反復インデックス。
        middleware_pipeline: 実行時に適用するオプションのmiddlewareパイプライン。

    戻り値:
        実行結果または例外を含むFunctionResultContent。

    例外:
        KeyError: 要求された関数がtool_mapに存在しない場合。

    """
    from ._types import (
        FunctionApprovalRequestContent,
        FunctionApprovalResponseContent,
        FunctionCallContent,
        FunctionResultContent,
    )

    tool: AIFunction[BaseModel, Any] | None = None
    if isinstance(function_call_content, FunctionCallContent):
        tool = tool_map.get(function_call_content.name)
        if tool is None:
            raise KeyError(f"No tool or function named '{function_call_content.name}'")
        if tool.approval_mode == "always_require":
            return FunctionApprovalRequestContent(id=function_call_content.call_id, function_call=function_call_content)
    else:
        if isinstance(function_call_content, FunctionApprovalResponseContent):
            if function_call_content.approved:
                tool = tool_map.get(function_call_content.function_call.name)
                if tool is None:
                    # ホストされたツールであると仮定します。
                    return function_call_content
                function_call_content = function_call_content.function_call
            else:
                raise ToolException("Unapproved tool cannot be executed.")

    parsed_args: dict[str, Any] = dict(function_call_content.parse_arguments() or {})

    # ユーザー提供の引数とマージします。右側が優先され、競合時は解析済み引数が勝ちます。
    merged_args: dict[str, Any] = (custom_args or {}) | parsed_args
    try:
        args = tool.input_model.model_validate(merged_args)
    except ValidationError as exc:
        return FunctionResultContent(
            call_id=function_call_content.call_id,
            exception=exc,
        )
    if not middleware_pipeline or (
        not hasattr(middleware_pipeline, "has_middlewares") and not middleware_pipeline.has_middlewares
    ):
        # middlewareなし - 直接実行します。
        try:
            function_result = await tool.invoke(
                arguments=args,
                tool_call_id=function_call_content.call_id,
            )  # type: ignore[arg-type]
            return FunctionResultContent(
                call_id=function_call_content.call_id,
                result=function_result,
            )
        except Exception as exc:
            return FunctionResultContent(
                call_id=function_call_content.call_id,
                exception=exc,
            )
    # middlewareパイプラインがあればそれを通して実行します。
    from ._middleware import FunctionInvocationContext

    middleware_context = FunctionInvocationContext(
        function=tool,
        arguments=args,
        kwargs=custom_args or {},
    )

    async def final_function_handler(context_obj: Any) -> Any:
        return await tool.invoke(
            arguments=context_obj.arguments,
            tool_call_id=function_call_content.call_id,
        )

    try:
        function_result = await middleware_pipeline.execute(
            function=tool,
            arguments=args,
            context=middleware_context,
            final_handler=final_function_handler,
        )
        return FunctionResultContent(
            call_id=function_call_content.call_id,
            result=function_result,
        )
    except Exception as exc:
        return FunctionResultContent(
            call_id=function_call_content.call_id,
            exception=exc,
        )


def _get_tool_map(
    tools: "ToolProtocol \
    | Callable[..., Any] \
    | MutableMapping[str, Any] \
    | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]",
) -> dict[str, AIFunction[Any, Any]]:
    ai_function_list: dict[str, AIFunction[Any, Any]] = {}
    for tool in tools if isinstance(tools, list) else [tools]:
        if isinstance(tool, AIFunction):
            ai_function_list[tool.name] = tool
            continue
        if callable(tool):
            # 関数または呼び出し可能ならAIToolに変換します。
            ai_tool = ai_function(tool)
            ai_function_list[ai_tool.name] = ai_tool
    return ai_function_list


async def _execute_function_calls(
    custom_args: dict[str, Any],
    attempt_idx: int,
    function_calls: Sequence["FunctionCallContent"] | Sequence["FunctionApprovalResponseContent"],
    tools: "ToolProtocol \
    | Callable[..., Any] \
    | MutableMapping[str, Any] \
    | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]",
    middleware_pipeline: Any = None,  # Optional MiddlewarePipeline to avoid circular imports
) -> Sequence["Contents"]:
    """複数の関数呼び出しを同時に実行します。

    引数:
        custom_args: 各関数に渡すカスタム引数。
        attempt_idx: 現在の試行のインデックス。
        function_calls: 実行するFunctionCallContentのシーケンス。
        tools: 実行可能なツール。
        middleware_pipeline: 実行時に適用するオプションのmiddlewareパイプライン。

    戻り値:
        各関数呼び出しの結果を含むContentsのリスト。

    """
    from ._types import FunctionApprovalRequestContent, FunctionCallContent

    tool_map = _get_tool_map(tools)
    approval_tools = [tool_name for tool_name, tool in tool_map.items() if tool.approval_mode == "always_require"]
    # 承認が必要な関数呼び出しがあるか確認し、あればすべてに対して承認要求を返します。
    approval_needed = False
    for fcc in function_calls:
        if isinstance(fcc, FunctionCallContent) and fcc.name in approval_tools:
            approval_needed = True
            break
    if approval_needed:
        # 承認はFunction Call Contentsにのみ必要で、Approval Responsesには不要です。
        return [
            FunctionApprovalRequestContent(id=fcc.call_id, function_call=fcc)
            for fcc in function_calls
            if isinstance(fcc, FunctionCallContent)
        ]

    # すべての関数呼び出しを同時に実行します。
    return await asyncio.gather(*[
        _auto_invoke_function(
            function_call_content=function_call,  # type: ignore[arg-type]
            custom_args=custom_args,
            tool_map=tool_map,
            sequence_index=seq_idx,
            request_index=attempt_idx,
            middleware_pipeline=middleware_pipeline,
        )
        for seq_idx, function_call in enumerate(function_calls)
    ])


def _update_conversation_id(kwargs: dict[str, Any], conversation_id: str | None) -> None:
    """kwargsにconversation idを更新します。

    引数:
        kwargs: 更新するキーワード引数の辞書。
        conversation_id: 設定する会話ID、またはスキップする場合はNone。

    """
    if conversation_id is None:
        return
    if "chat_options" in kwargs:
        kwargs["chat_options"].conversation_id = conversation_id
    else:
        kwargs["conversation_id"] = conversation_id


def _extract_tools(kwargs: dict[str, Any]) -> Any:
    """kwargsまたはchat_optionsからツールを抽出します。

    戻り値:
        ToolProtocol | Callable[..., Any] | MutableMapping[str, Any] |
        Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]] | None

    """
    from ._types import ChatOptions

    tools = kwargs.get("tools")
    if not tools and (chat_options := kwargs.get("chat_options")) and isinstance(chat_options, ChatOptions):
        tools = chat_options.tools
    return tools


def _collect_approval_responses(
    messages: "list[ChatMessage]",
) -> dict[str, "FunctionApprovalResponseContent"]:
    """メッセージから承認されたものと拒否されたものの両方の承認レスポンスを収集します。"""
    from ._types import ChatMessage, FunctionApprovalResponseContent

    fcc_todo: dict[str, FunctionApprovalResponseContent] = {}
    for msg in messages:
        for content in msg.contents if isinstance(msg, ChatMessage) else []:
            # 承認されたものと拒否されたものの両方を収集します。
            if isinstance(content, FunctionApprovalResponseContent):
                fcc_todo[content.id] = content
    return fcc_todo


def _replace_approval_contents_with_results(
    messages: "list[ChatMessage]",
    fcc_todo: dict[str, "FunctionApprovalResponseContent"],
    approved_function_results: "list[Contents]",
) -> None:
    """承認要求/応答の内容を関数呼び出し/結果の内容でその場で置き換えます。"""
    from ._types import (
        FunctionApprovalRequestContent,
        FunctionApprovalResponseContent,
        FunctionCallContent,
        FunctionResultContent,
        Role,
    )

    result_idx = 0
    for msg in messages:
        # 最初のパス - 重複を避けるため既存の関数呼び出しIDを収集します。
        existing_call_ids = {
            content.call_id for content in msg.contents if isinstance(content, FunctionCallContent) and content.call_id
        }

        # 削除すべき承認要求（重複）を追跡します。
        contents_to_remove = []

        for content_idx, content in enumerate(msg.contents):
            if isinstance(content, FunctionApprovalRequestContent):
                # 既に存在する場合は関数呼び出しを追加しません（重複を作成するため）。
                if content.function_call.call_id in existing_call_ids:
                    # 削除のためにマークするだけ - 関数呼び出しは既に存在します。
                    contents_to_remove.append(content_idx)
                else:
                    # 存在しない場合のみ関数呼び出し内容を戻します。
                    msg.contents[content_idx] = content.function_call
            elif isinstance(content, FunctionApprovalResponseContent):
                if content.approved and content.id in fcc_todo:
                    # 対応する結果に置き換えます。
                    if result_idx < len(approved_function_results):
                        msg.contents[content_idx] = approved_function_results[result_idx]
                        result_idx += 1
                        msg.role = Role.TOOL
                else:
                    # 拒否された呼び出しに対して「承認されていない」結果を作成します。
                    # function_call.call_id（関数のID）を使用し、content.id（承認のID）は使用しません。
                    msg.contents[content_idx] = FunctionResultContent(
                        call_id=content.function_call.call_id,
                        result="Error: Tool call invocation was rejected by user.",
                    )
                    msg.role = Role.TOOL

        # 重複した承認要求を削除します（インデックスを保持するため逆順で）。
        for idx in reversed(contents_to_remove):
            msg.contents.pop(idx)


def _handle_function_calls_response(
    func: Callable[..., Awaitable["ChatResponse"]],
) -> Callable[..., Awaitable["ChatResponse"]]:
    """get_responseメソッドをデコレートして関数呼び出しを有効にします。

    引数:
        func: デコレートするget_responseメソッド。

    戻り値:
        関数呼び出しを自動的に処理するデコレート済み関数。

    """

    def decorator(
        func: Callable[..., Awaitable["ChatResponse"]],
    ) -> Callable[..., Awaitable["ChatResponse"]]:
        """内部デコレータ。"""

        @wraps(func)
        async def function_invocation_wrapper(
            self: "ChatClientProtocol",
            messages: "str | ChatMessage | list[str] | list[ChatMessage]",
            **kwargs: Any,
        ) -> "ChatResponse":
            from ._clients import prepare_messages
            from ._middleware import extract_and_merge_function_middleware
            from ._types import (
                ChatMessage,
                FunctionApprovalRequestContent,
                FunctionCallContent,
                FunctionResultContent,
            )

            # chat clientから関数middlewareを抽出し、kwargsのパイプラインとマージします。
            extract_and_merge_function_middleware(self, **kwargs)

            # 基底関数を呼び出す前にmiddlewareパイプラインを抽出します。 基底関数はkwargs内でそれを保持しない可能性があるためです。
            stored_middleware_pipeline = kwargs.get("_function_middleware_pipeline")

            # インスタンスのadditional_propertiesまたはクラス属性からmax_iterationsを取得します。
            instance_max_iterations: int = DEFAULT_MAX_ITERATIONS
            if hasattr(self, "additional_properties") and self.additional_properties:
                instance_max_iterations = self.additional_properties.get("max_iterations", DEFAULT_MAX_ITERATIONS)
            elif hasattr(self.__class__, "MAX_ITERATIONS"):
                instance_max_iterations = getattr(self.__class__, "MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)

            prepped_messages = prepare_messages(messages)
            response: "ChatResponse | None" = None
            fcc_messages: "list[ChatMessage]" = []
            for attempt_idx in range(instance_max_iterations):
                fcc_todo = _collect_approval_responses(prepped_messages)
                if fcc_todo:
                    tools = _extract_tools(kwargs)
                    # 拒否されたものではなく、承認された関数呼び出しのみを実行します。
                    approved_responses = [resp for resp in fcc_todo.values() if resp.approved]
                    approved_function_results: list[Contents] = []
                    if approved_responses:
                        approved_function_results = await _execute_function_calls(
                            custom_args=kwargs,
                            attempt_idx=attempt_idx,
                            function_calls=approved_responses,
                            tools=tools,  # type: ignore
                            middleware_pipeline=stored_middleware_pipeline,
                        )
                    _replace_approval_contents_with_results(prepped_messages, fcc_todo, approved_function_results)

                response = await func(self, messages=prepped_messages, **kwargs)
                # 関数呼び出しがある場合は最初に処理します。
                function_results = {
                    it.call_id for it in response.messages[0].contents if isinstance(it, FunctionResultContent)
                }
                function_calls = [
                    it
                    for it in response.messages[0].contents
                    if isinstance(it, FunctionCallContent) and it.call_id not in function_results
                ]

                if response.conversation_id is not None:
                    _update_conversation_id(kwargs, response.conversation_id)
                    prepped_messages = []

                # middlewareがfunc呼び出し前と異なる可能性があるため、ここでツールをロードします。
                tools = _extract_tools(kwargs)
                if function_calls and tools:
                    # kwargsから抽出する代わりに保存されたmiddlewareパイプラインを使用します。
                    # kwargsは基底関数によって変更されている可能性があるためです。
                    function_call_results: list[Contents] = await _execute_function_calls(
                        custom_args=kwargs,
                        attempt_idx=attempt_idx,
                        function_calls=function_calls,
                        tools=tools,  # type: ignore
                        middleware_pipeline=stored_middleware_pipeline,
                    )

                    # 結果に承認要求があるか確認します。
                    if any(isinstance(fccr, FunctionApprovalRequestContent) for fccr in function_call_results):
                        # 承認要求を既存のassistantメッセージ（tool_calls付き）に追加します。
                        # 別のtoolメッセージを作成する代わりに。
                        from ._types import Role

                        if response.messages and response.messages[0].role == Role.ASSISTANT:
                            response.messages[0].contents.extend(function_call_results)
                        else:
                            # フォールバック: 新しいassistantメッセージを作成します（通常は発生しません）。
                            result_message = ChatMessage(role="assistant", contents=function_call_results)
                            response.messages.append(result_message)
                        return response

                    # 結果を含む単一のChatMessageをレスポンスに追加します。
                    result_message = ChatMessage(role="tool", contents=function_call_results)
                    response.messages.append(result_message)
                    # この後のresponseには2つのメッセージが含まれている必要があります。 1つはfunction callの内容を含み、
                    # もう1つはfunctionの結果の内容を含みます。 数とcall_idは一致している必要があります。
                    # これは最初の実行を除くすべての実行で行われます。 すべてのfunction callメッセージを追跡する必要があります。
                    fcc_messages.extend(response.messages)
                    if getattr(kwargs.get("chat_options"), "store", False):
                        prepped_messages.clear()
                        prepped_messages.append(result_message)
                    else:
                        prepped_messages.extend(response.messages)
                    continue
                # このポイントに到達した場合、処理すべきfunction callがなかったことを意味します。 前のfunction
                # callとレスポンスをリストの先頭に追加し、最終的なレスポンスが最後になるようにします。 TODO (eavanvalkenburg):
                # この動作を制御する？
                if fcc_messages:
                    for msg in reversed(fcc_messages):
                        response.messages.insert(0, msg)
                return response

            # フェイルセーフ：ツールを諦め、モデルにプレーンな回答を求める
            kwargs["tool_choice"] = "none"
            response = await func(self, messages=prepped_messages, **kwargs)
            if fcc_messages:
                for msg in reversed(fcc_messages):
                    response.messages.insert(0, msg)
            return response

        return function_invocation_wrapper  # type: ignore

    return decorator(func)


def _handle_function_calls_streaming_response(
    func: Callable[..., AsyncIterable["ChatResponseUpdate"]],
) -> Callable[..., AsyncIterable["ChatResponseUpdate"]]:
    """get_streaming_responseメソッドをデコレートしてfunction callを処理します。

    Args:
        func: デコレートするget_streaming_responseメソッド。

    Returns:
        ストリーミングモードでfunction callを処理するデコレートされた関数。

    """

    def decorator(
        func: Callable[..., AsyncIterable["ChatResponseUpdate"]],
    ) -> Callable[..., AsyncIterable["ChatResponseUpdate"]]:
        """内部デコレータ。"""

        @wraps(func)
        async def streaming_function_invocation_wrapper(
            self: "ChatClientProtocol",
            messages: "str | ChatMessage | list[str] | list[ChatMessage]",
            **kwargs: Any,
        ) -> AsyncIterable["ChatResponseUpdate"]:
            """内部のget streaming responseメソッドをラップしてツール呼び出しを処理します。"""
            from ._clients import prepare_messages
            from ._middleware import extract_and_merge_function_middleware
            from ._types import (
                ChatMessage,
                ChatResponse,
                ChatResponseUpdate,
                FunctionCallContent,
                FunctionResultContent,
            )

            # chat clientからfunction middlewareを抽出し、kwargsのpipelineとマージします。
            extract_and_merge_function_middleware(self, **kwargs)

            # 基底関数を呼び出す前にmiddleware pipelineを抽出します。 基底関数はkwargs内でそれを保持しない可能性があるためです。
            stored_middleware_pipeline = kwargs.get("_function_middleware_pipeline")

            # インスタンスのadditional_propertiesまたはクラス属性からmax_iterationsを取得します。
            instance_max_iterations: int = DEFAULT_MAX_ITERATIONS
            if hasattr(self, "additional_properties") and self.additional_properties:
                instance_max_iterations = self.additional_properties.get("max_iterations", DEFAULT_MAX_ITERATIONS)
            elif hasattr(self.__class__, "MAX_ITERATIONS"):
                instance_max_iterations = getattr(self.__class__, "MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)

            prepped_messages = prepare_messages(messages)
            fcc_messages: "list[ChatMessage]" = []
            for attempt_idx in range(instance_max_iterations):
                fcc_todo = _collect_approval_responses(prepped_messages)
                if fcc_todo:
                    tools = _extract_tools(kwargs)
                    # 拒否されたものではなく、承認されたfunction callのみを実行します。
                    approved_responses = [resp for resp in fcc_todo.values() if resp.approved]
                    approved_function_results: list[Contents] = []
                    if approved_responses:
                        approved_function_results = await _execute_function_calls(
                            custom_args=kwargs,
                            attempt_idx=attempt_idx,
                            function_calls=approved_responses,
                            tools=tools,  # type: ignore
                            middleware_pipeline=stored_middleware_pipeline,
                        )
                    _replace_approval_contents_with_results(prepped_messages, fcc_todo, approved_function_results)

                all_updates: list["ChatResponseUpdate"] = []
                async for update in func(self, messages=prepped_messages, **kwargs):
                    all_updates.append(update)
                    yield update

                # updates内のFunctionCallContentを効率的にチェックします。 少なくとも1つあれば停止して継続します。
                # FCCがなければ戻ります。
                from ._types import FunctionApprovalRequestContent

                if not any(
                    isinstance(item, (FunctionCallContent, FunctionApprovalRequestContent))
                    for upd in all_updates
                    for item in upd.contents
                ):
                    return

                # 更新を組み合わせて完全なレスポンスを作成しています。 プロンプトによっては、メッセージにfunction
                # callの内容とその他の内容が含まれる場合があります。

                response: "ChatResponse" = ChatResponse.from_chat_response_updates(all_updates)
                # function callを取得します（すでに結果があるものは除く）。
                function_results = {
                    it.call_id for it in response.messages[0].contents if isinstance(it, FunctionResultContent)
                }
                function_calls = [
                    it
                    for it in response.messages[0].contents
                    if isinstance(it, FunctionCallContent) and it.call_id not in function_results
                ]

                # conversation idが存在する場合、それはメッセージがサーバー上にホストされていることを意味します。
                # この場合、kwargsをconversation idで更新し、メッセージをクリアする必要があります。
                if response.conversation_id is not None:
                    _update_conversation_id(kwargs, response.conversation_id)
                    prepped_messages = []

                # ここでツールをロードします。middlewareがfunc呼び出し前と比較して変更している可能性があるためです。
                tools = _extract_tools(kwargs)
                if function_calls and tools:
                    # kwargsから抽出する代わりに保存されたmiddleware pipelineを使用します。
                    # kwargsは基底関数によって変更されている可能性があるためです。
                    function_call_results: list[Contents] = await _execute_function_calls(
                        custom_args=kwargs,
                        attempt_idx=attempt_idx,
                        function_calls=function_calls,
                        tools=tools,  # type: ignore
                        middleware_pipeline=stored_middleware_pipeline,
                    )

                    # 結果に承認リクエストがあるかどうかをチェックします。
                    if any(isinstance(fccr, FunctionApprovalRequestContent) for fccr in function_call_results):
                        # 承認リクエストを既存のassistantメッセージ（tool_calls付き）に追加します。
                        # 別のtoolメッセージを作成するのではなく。
                        from ._types import Role

                        if response.messages and response.messages[0].role == Role.ASSISTANT:
                            response.messages[0].contents.extend(function_call_results)
                            # 承認リクエストをassistantメッセージの一部としてyieldします。
                            yield ChatResponseUpdate(contents=function_call_results, role="assistant")
                        else:
                            # フォールバック：新しいassistantメッセージを作成します（通常は発生しません）。
                            result_message = ChatMessage(role="assistant", contents=function_call_results)
                            yield ChatResponseUpdate(contents=function_call_results, role="assistant")
                            response.messages.append(result_message)
                        return

                    # 結果を含む単一のChatMessageをレスポンスに追加します。
                    result_message = ChatMessage(role="tool", contents=function_call_results)
                    yield ChatResponseUpdate(contents=function_call_results, role="tool")
                    response.messages.append(result_message)
                    # この後のresponseには2つのメッセージが含まれている必要があります。 1つはfunction callの内容を含み、
                    # もう1つはfunctionの結果の内容を含みます。 数とcall_idは一致している必要があります。
                    # これは最初の実行を除くすべての実行で行われます。 すべてのfunction callメッセージを追跡する必要があります。
                    fcc_messages.extend(response.messages)
                    if getattr(kwargs.get("chat_options"), "store", False):
                        prepped_messages.clear()
                        prepped_messages.append(result_message)
                    else:
                        prepped_messages.extend(response.messages)
                    continue
                # このポイントに到達した場合、処理すべきfunction callがなかったことを意味します。 これで完了です。
                return

            # フェイルセーフ：ツールを諦め、モデルにプレーンな回答を求める
            kwargs["tool_choice"] = "none"
            async for update in func(self, messages=prepped_messages, **kwargs):
                yield update

        return streaming_function_invocation_wrapper

    return decorator(func)


def use_function_invocation(
    chat_client: type[TChatClient],
) -> type[TChatClient]:
    """チャットクライアントのためのツール呼び出しを有効にするクラスデコレータ。

    このデコレータは``get_response``と``get_streaming_response``メソッドをラップし、
    モデルからのfunction callを自動的に処理し、実行し、
    結果をモデルに返してさらなる処理を可能にします。

    Args:
        chat_client: デコレートするチャットクライアントクラス。

    Returns:
        function invocationが有効になったデコレート済みチャットクライアントクラス。

    Raises:
        ChatClientInitializationError: チャットクライアントに必要なメソッドがない場合。

    Examples:
        .. code-block:: python

            from agent_framework import use_function_invocation, BaseChatClient


            @use_function_invocation
            class MyCustomClient(BaseChatClient):
                async def get_response(self, messages, **kwargs):
                    # ここに実装
                    pass

                async def get_streaming_response(self, messages, **kwargs):
                    # ここに実装
                    pass


            # クライアントは自動的にfunction callを処理します
            client = MyCustomClient()

    """
    if getattr(chat_client, FUNCTION_INVOKING_CHAT_CLIENT_MARKER, False):
        return chat_client

    # MAX_ITERATIONSをクラス変数として設定（未設定の場合）。
    if not hasattr(chat_client, "MAX_ITERATIONS"):
        chat_client.MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS  # type: ignore

    try:
        chat_client.get_response = _handle_function_calls_response(  # type: ignore
            func=chat_client.get_response,  # type: ignore
        )
    except AttributeError as ex:
        raise ChatClientInitializationError(
            f"Chat client {chat_client.__name__} does not have a get_response method, cannot apply function invocation."
        ) from ex
    try:
        chat_client.get_streaming_response = _handle_function_calls_streaming_response(  # type: ignore
            func=chat_client.get_streaming_response,
        )
    except AttributeError as ex:
        raise ChatClientInitializationError(
            f"Chat client {chat_client.__name__} does not have a get_streaming_response method, "
            "cannot apply function invocation."
        ) from ex
    setattr(chat_client, FUNCTION_INVOKING_CHAT_CLIENT_MARKER, True)
    return chat_client
