# Copyright (c) Microsoft. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, MutableMapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from ._logging import get_logger
from ._mcp import MCPTool
from ._memory import AggregateContextProvider, ContextProvider
from ._middleware import (
    ChatMiddleware,
    ChatMiddlewareCallable,
    FunctionMiddleware,
    FunctionMiddlewareCallable,
    Middleware,
)
from ._serialization import SerializationMixin
from ._threads import ChatMessageStoreProtocol
from ._tools import ToolProtocol
from ._types import (
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    ToolMode,
)

if TYPE_CHECKING:
    from ._agents import ChatAgent


TInput = TypeVar("TInput", contravariant=True)
TEmbedding = TypeVar("TEmbedding")
TBaseChatClient = TypeVar("TBaseChatClient", bound="BaseChatClient")

logger = get_logger()

__all__ = [
    "BaseChatClient",
    "ChatClientProtocol",
]


# region ChatClientProtocol Protocol


@runtime_checkable
class ChatClientProtocol(Protocol):
    """レスポンスを生成できるチャットクライアントのプロトコル。

    このプロトコルはすべてのチャットクライアントが実装すべきインターフェースを定義し、
    ストリーミングおよび非ストリーミングのレスポンス生成メソッドを含みます。

    Note:
        プロトコルは構造的サブタイピング（ダックタイピング）を使用します。
        クラスはこのプロトコルを明示的に継承する必要はありません。

    Examples:
        .. code-block:: python

            from agent_framework import ChatClientProtocol, ChatResponse, ChatMessage


            # 必要なメソッドを実装する任意のクラスは互換性があります
            class CustomChatClient:
                @property
                def additional_properties(self) -> dict[str, Any]:
                    return {}

                async def get_response(self, messages, **kwargs):
                    # カスタム実装
                    return ChatResponse(messages=[], response_id="custom")

                def get_streaming_response(self, messages, **kwargs):
                    async def _stream():
                        from agent_framework import ChatResponseUpdate

                        yield ChatResponseUpdate()

                    return _stream()


            # インスタンスがプロトコルを満たすことを検証
            client = CustomChatClient()
            assert isinstance(client, ChatClientProtocol)

    """

    @property
    def additional_properties(self) -> dict[str, Any]:
        """クライアントに関連付けられた追加のプロパティを取得します。"""
        ...

    async def get_response(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str | int, float] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        model_id: str | None = None,
        presence_penalty: float | None = None,
        response_format: type[BaseModel] | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        store: bool | None = None,
        temperature: float | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = "auto",
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """入力を送信し、レスポンスを返します。

        Args:
            messages: 送信する入力メッセージのシーケンス。

        Keyword Args:
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: Agentに使用するmodel_id。
            presence_penalty: 使用するpresence penalty。
            response_format: レスポンスのフォーマット。
            seed: 使用するランダムシード。
            stop: リクエストの停止シーケンス。
            store: レスポンスを保存するかどうか。
            temperature: 使用するサンプリング温度。
            tool_choice: リクエストのtool choice。
            tools: リクエストに使用するtools。
            top_p: 使用するnucleus sampling確率。
            user: リクエストに関連付けるユーザー。
            additional_properties: リクエストに含める追加のプロパティ。
            kwargs: その他の追加キーワード引数。
                呼び出される関数にのみ渡されます。

        Returns:
            クライアントによって生成されたレスポンスメッセージ。

        Raises:
            ValueError: 入力メッセージシーケンスが``None``の場合。
        """
        ...

    def get_streaming_response(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str | int, float] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        model_id: str | None = None,
        presence_penalty: float | None = None,
        response_format: type[BaseModel] | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        store: bool | None = None,
        temperature: float | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = "auto",
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """入力メッセージを送信し、レスポンスをストリームします。

        Args:
            messages: 送信する入力メッセージのシーケンス。

        Keyword Args:
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: Agentに使用するmodel_id。
            presence_penalty: 使用するpresence penalty。
            response_format: レスポンスのフォーマット。
            seed: 使用するランダムシード。
            stop: リクエストの停止シーケンス。
            store: レスポンスを保存するかどうか。
            temperature: 使用するサンプリング温度。
            tool_choice: リクエストのtool choice。
            tools: リクエストに使用するtools。
            top_p: 使用するnucleus sampling確率。
            user: リクエストに関連付けるユーザー。
            additional_properties: リクエストに含める追加のプロパティ。
            kwargs: その他の追加キーワード引数。
                呼び出される関数にのみ渡されます。

        Yields:
            ChatResponseUpdate: クライアントによって生成されたレスポンスメッセージの内容を含む非同期イテラブルのチャットレスポンス更新。

        Raises:
            ValueError: 入力メッセージシーケンスが``None``の場合。
        """
        ...


# region ChatClientBase


def prepare_messages(messages: str | ChatMessage | list[str] | list[ChatMessage]) -> list[ChatMessage]:
    """さまざまなメッセージ入力形式をChatMessageオブジェクトのリストに変換します。

    Args:
        messages: 対応するさまざまな形式の入力メッセージ。

    Returns:
        ChatMessageオブジェクトのリスト。

    """
    if isinstance(messages, str):
        return [ChatMessage(role="user", text=messages)]
    if isinstance(messages, ChatMessage):
        return [messages]
    return_messages: list[ChatMessage] = []
    for msg in messages:
        if isinstance(msg, str):
            msg = ChatMessage(role="user", text=msg)
        return_messages.append(msg)
    return return_messages


def merge_chat_options(
    *,
    base_chat_options: ChatOptions | Any | None,
    model_id: str | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict[str | int, float] | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    presence_penalty: float | None = None,
    response_format: type[BaseModel] | None = None,
    seed: int | None = None,
    stop: str | Sequence[str] | None = None,
    store: bool | None = None,
    temperature: float | None = None,
    tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = "auto",
    tools: list[ToolProtocol | dict[str, Any] | Callable[..., Any]] | None = None,
    top_p: float | None = None,
    user: str | None = None,
    additional_properties: dict[str, Any] | None = None,
) -> ChatOptions:
    """base chat optionsと直接パラメータをマージして新しいChatOptionsインスタンスを作成します。

    base_chat_optionsと個別パラメータの両方が提供された場合、個別パラメータが優先され、base_chat_optionsの対応する値を上書きします。
    両方のソースからのtoolsは単一のリストに結合されます。

    Keyword Args:
        base_chat_options: 直接パラメータとマージするオプションのbase ChatOptions。
        model_id: Agentに使用するmodel_id。
        frequency_penalty: 使用するfrequency penalty。
        logit_bias: 使用するlogit bias。
        max_tokens: 生成する最大トークン数。
        metadata: リクエストに含める追加のメタデータ。
        presence_penalty: 使用するpresence penalty。
        response_format: レスポンスのフォーマット。
        seed: 使用するランダムシード。
        stop: リクエストの停止シーケンス。
        store: レスポンスを保存するかどうか。
        temperature: 使用するサンプリング温度。
        tool_choice: リクエストのtool choice。
        tools: リクエストに使用する正規化されたtools。
        top_p: 使用するnucleus sampling確率。
        user: リクエストに関連付けるユーザー。
        additional_properties: リクエストに含める追加のプロパティ。

    Returns:
        マージされた値を持つ新しいChatOptionsインスタンス。

    Raises:
        TypeError: base_chat_optionsがNoneでなく、ChatOptionsのインスタンスでない場合。
    """
    # base_chat_optionsの型を検証します（提供されている場合）
    if base_chat_options is not None and not isinstance(base_chat_options, ChatOptions):
        raise TypeError("chat_options must be an instance of ChatOptions")

    if base_chat_options is None:
        base_chat_options = ChatOptions()

    return base_chat_options & ChatOptions(
        model_id=model_id,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        max_tokens=max_tokens,
        metadata=metadata,
        presence_penalty=presence_penalty,
        response_format=response_format,
        seed=seed,
        stop=stop,
        store=store,
        temperature=temperature,
        top_p=top_p,
        tool_choice=tool_choice,
        tools=tools,
        user=user,
        additional_properties=additional_properties,
    )


class BaseChatClient(SerializationMixin, ABC):
    """チャットクライアントの基底クラス。

    この抽象基底クラスは、ミドルウェアサポート、メッセージ準備、ツールの正規化など、チャットクライアント実装のコア機能を提供します。

    注意:
        BaseChatClientは抽象基底クラスであるため直接インスタンス化できません。
        サブクラスは``_inner_get_response()``および``_inner_get_streaming_response()``を実装する必要があります。

    Examples:
        .. code-block:: python

            from agent_framework import BaseChatClient, ChatResponse, ChatMessage
            from collections.abc import AsyncIterable


            class CustomChatClient(BaseChatClient):
                async def _inner_get_response(self, *, messages, chat_options, **kwargs):
                    # カスタム実装
                    return ChatResponse(
                        messages=[ChatMessage(role="assistant", text="Hello!")], response_id="custom-response"
                    )

                async def _inner_get_streaming_response(self, *, messages, chat_options, **kwargs):
                    # カスタムストリーミング実装
                    from agent_framework import ChatResponseUpdate

                    yield ChatResponseUpdate(role="assistant", contents=[{"type": "text", "text": "Hello!"}])


            # カスタムクライアントのインスタンスを作成
            client = CustomChatClient()

            # クライアントを使用してレスポンスを取得
            response = await client.get_response("Hello, how are you?")

    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "unknown"
    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"additional_properties"}
    # これはOTelセットアップに使用され、サブクラスでオーバーライドされるべきです

    def __init__(
        self,
        *,
        middleware: (
            ChatMiddleware
            | ChatMiddlewareCallable
            | FunctionMiddleware
            | FunctionMiddlewareCallable
            | list[ChatMiddleware | ChatMiddlewareCallable | FunctionMiddleware | FunctionMiddlewareCallable]
            | None
        ) = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """BaseChatClientインスタンスを初期化します。

        Keyword Args:
            middleware: クライアント用のMiddleware。
            additional_properties: クライアント用の追加プロパティ。
            kwargs: 追加のキーワード引数（additional_propertiesにマージされます）。

        """
        # kwargsをadditional_propertiesにマージします
        self.additional_properties = additional_properties or {}
        self.additional_properties.update(kwargs)

        self.middleware = middleware

    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]:
        """インスタンスを辞書に変換します。

        additional_propertiesのフィールドをルートレベルに抽出します。

        Keyword Args:
            exclude: シリアライズから除外するフィールド名のセット。
            exclude_none: None値を出力から除外するかどうか。デフォルトはTrue。

        Returns:
            インスタンスの辞書表現。

        """
        # SerializationMixinから基本の辞書を取得します
        result = super().to_dict(exclude=exclude, exclude_none=exclude_none)

        # additional_propertiesをルートレベルに抽出します
        if self.additional_properties:
            result.update(self.additional_properties)

        return result

    def prepare_messages(
        self, messages: str | ChatMessage | list[str] | list[ChatMessage], chat_options: ChatOptions
    ) -> MutableSequence[ChatMessage]:
        """さまざまなメッセージ入力形式をChatMessageオブジェクトのリストに変換します。

        chat_optionsにシステム指示がある場合は先頭に追加します。

        Args:
            messages: 対応するさまざまな形式の入力メッセージ。
            chat_options: 指示やその他の設定を含むチャットオプション。

        Returns:
            変更可能なChatMessageオブジェクトのシーケンス。

        """
        if chat_options.instructions:
            system_msg = ChatMessage(role="system", text=chat_options.instructions)
            return [system_msg, *prepare_messages(messages)]
        return prepare_messages(messages)

    def _filter_internal_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """チャットクライアント実装に渡すべきでない内部フレームワークパラメータをフィルタリングします。

        Keyword Args:
            kwargs: 元のkwargs辞書。

        Returns:
            内部パラメータを除いたフィルタリング済みkwargs辞書。

        """
        return {k: v for k, v in kwargs.items() if not k.startswith("_")}

    @staticmethod
    async def _normalize_tools(
        tools: ToolProtocol
        | MutableMapping[str, Any]
        | Callable[..., Any]
        | Sequence[ToolProtocol | MutableMapping[str, Any] | Callable[..., Any]]
        | None = None,
    ) -> list[ToolProtocol | dict[str, Any] | Callable[..., Any]]:
        """tools入力を一貫したリスト形式に正規化します。

        MCP toolsを構成関数に展開し、必要に応じて接続します。

        Args:
            tools: 対応するさまざまな形式のtools。

        Returns:
            正規化されたtoolsのリスト。

        """
        from typing import cast

        final_tools: list[ToolProtocol | dict[str, Any] | Callable[..., Any]] = []
        if not tools:
            return final_tools
        # シーケンスが渡された場合にcastを使用します（おそらくすでにリストです）
        tools_list = (
            cast(list[ToolProtocol | MutableMapping[str, Any] | Callable[..., Any]], tools)
            if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes))
            else [tools]
        )
        for tool in tools_list:  # type: ignore[reportUnknownType]
            if isinstance(tool, MCPTool):
                if not tool.is_connected:
                    await tool.connect()
                final_tools.extend(tool.functions)  # type: ignore
                continue
            final_tools.append(tool)  # type: ignore
        return final_tools

    # region 派生クラスで実装される内部メソッド

    @abstractmethod
    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        """AIサービスにチャットリクエストを送信します。

        Keyword Args:
            messages: 送信するチャットメッセージ。
            chat_options: リクエストのオプション。
            kwargs: その他の追加キーワード引数。

        Returns:
            レスポンスを表すチャットレスポンスの内容。

        """

    @abstractmethod
    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """AIサービスにストリーミングチャットリクエストを送信します。

        Keyword Args:
            messages: 送信するチャットメッセージ。
            chat_options: リクエストのchat_options。
            kwargs: その他の追加キーワード引数。

        Yields:
            ChatResponseUpdate: ストリーミングチャットメッセージの内容。

        """
        # 以下はmypyに必要です:
        # https://mypy.readthedocs.io/en/stable/more_types.html#asynchronous-iterators
        if False:
            yield
        await asyncio.sleep(0)  # pragma: no cover
        # これはno-opですが、メソッドをasyncにしてAsyncIterableを返すことを可能にします。
        # 実際の実装では必要に応じてChatResponseUpdateインスタンスをyieldすべきです。

    # endregion region Public method

    async def get_response(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str | int, float] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        model_id: str | None = None,
        presence_penalty: float | None = None,
        response_format: type[BaseModel] | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        store: bool | None = None,
        temperature: float | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = "auto",
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """チャットクライアントからレスポンスを取得します。

        ``chat_options``（kwargs内）と個別パラメータの両方が提供された場合、
        個別パラメータが優先され、``chat_options``の対応する値を上書きします。
        両方のソースからのtoolsは単一のリストに結合されます。

        Args:
            messages: モデルに送信するメッセージまたはメッセージ群。

        Keyword Args:
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: Agentに使用するmodel_id。
            presence_penalty: 使用するpresence penalty。
            response_format: レスポンスのフォーマット。
            seed: 使用するランダムシード。
            stop: リクエストの停止シーケンス。
            store: レスポンスを保存するかどうか。
            temperature: 使用するサンプリング温度。
            tool_choice: リクエストのtool choice。
            tools: リクエストに使用するtools。
            top_p: 使用するnucleus sampling確率。
            user: リクエストに関連付けるユーザー。
            additional_properties: リクエストに含める追加のプロパティ。
                プロバイダー固有のパラメータに使用可能。
            kwargs: その他の追加キーワード引数。
                ``chat_options``を含む場合があり、これは直接パラメータで上書き可能な基本値を提供します。

        Returns:
            model_idからのチャットレスポンス。

        """
        # toolsを正規化しbase chat_optionsとマージします
        normalized_tools = await self._normalize_tools(tools)
        chat_options = merge_chat_options(
            base_chat_options=kwargs.pop("chat_options", None),
            model_id=model_id,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            metadata=metadata,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            store=store,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=normalized_tools,
            top_p=top_p,
            user=user,
            additional_properties=additional_properties,
        )

        # conversation_idが設定されている場合、storeがTrueであることを検証します
        if chat_options.conversation_id is not None and chat_options.store is not True:
            logger.warning(
                "When conversation_id is set, store must be True for service-managed threads. "
                "Automatically setting store=True."
            )
            chat_options.store = True

        prepped_messages = self.prepare_messages(messages, chat_options)
        self._prepare_tool_choice(chat_options=chat_options)

        filtered_kwargs = self._filter_internal_kwargs(kwargs)
        return await self._inner_get_response(messages=prepped_messages, chat_options=chat_options, **filtered_kwargs)

    async def get_streaming_response(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str | int, float] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        model_id: str | None = None,
        presence_penalty: float | None = None,
        response_format: type[BaseModel] | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        store: bool | None = None,
        temperature: float | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = "auto",
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """チャットクライアントからストリーミングレスポンスを取得します。

        ``chat_options``（kwargs内）と個別のパラメータの両方が提供された場合、
        個別のパラメータが優先され、``chat_options``内の対応する値を上書きします。
        両方のソースからのツールは単一のリストに結合されます。

        Args:
            messages: モデルに送信するメッセージまたはメッセージのリスト。

        Keyword Args:
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: エージェントに使用するmodel_id。
            presence_penalty: 使用するpresence penalty。
            response_format: レスポンスのフォーマット。
            seed: 使用するランダムシード。
            stop: リクエストの停止シーケンス。
            store: レスポンスを保存するかどうか。
            temperature: 使用するサンプリング温度。
            tool_choice: リクエストのツール選択。
            tools: リクエストに使用するツール。
            top_p: 使用するnucleus sampling確率。
            user: リクエストに関連付けるユーザー。
            additional_properties: リクエストに含める追加のプロパティ。
                プロバイダー固有のパラメータに使用可能。
            kwargs: その他のキーワード引数。
                ``chat_options``を含む場合があり、これは直接パラメータで上書き可能な基本値を提供します。

        Yields:
            ChatResponseUpdate: LLMからのレスポンスを表すストリーム。
        """
        # ツールを正規化し、基本のchat_optionsとマージします。
        normalized_tools = await self._normalize_tools(tools)
        chat_options = merge_chat_options(
            base_chat_options=kwargs.pop("chat_options", None),
            model_id=model_id,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            metadata=metadata,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            store=store,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=normalized_tools,
            top_p=top_p,
            user=user,
            additional_properties=additional_properties,
        )

        # conversation_idが設定されている場合、storeがTrueであることを検証します。
        if chat_options.conversation_id is not None and chat_options.store is not True:
            logger.warning(
                "When conversation_id is set, store must be True for service-managed threads. "
                "Automatically setting store=True."
            )
            chat_options.store = True

        prepped_messages = self.prepare_messages(messages, chat_options)
        self._prepare_tool_choice(chat_options=chat_options)

        filtered_kwargs = self._filter_internal_kwargs(kwargs)
        async for update in self._inner_get_streaming_response(
            messages=prepped_messages, chat_options=chat_options, **filtered_kwargs
        ):
            yield update

    def _prepare_tool_choice(self, chat_options: ChatOptions) -> None:
        """チャットオプションのためにツールとツール選択を準備します。

        この関数はサブクラスでオーバーライドしてツールの処理をカスタマイズすべきです。
        現状ではAIFunctionのみを解析します。

        Args:
            chat_options: 準備するチャットオプション。

        """
        chat_tool_mode = chat_options.tool_choice
        if chat_tool_mode is None or chat_tool_mode == ToolMode.NONE or chat_tool_mode == "none":
            chat_options.tools = None
            chat_options.tool_choice = ToolMode.NONE.mode
            return
        if not chat_options.tools:
            chat_options.tool_choice = ToolMode.NONE.mode
        else:
            chat_options.tool_choice = chat_tool_mode.mode if isinstance(chat_tool_mode, ToolMode) else chat_tool_mode

    def service_url(self) -> str:
        """サービスのURLを取得します。

        サブクラスでオーバーライドして適切なURLを返してください。
        サービスにURLがない場合はNoneを返します。

        Returns:
            サービスのURL、または未実装の場合は'Unknown'。

        """
        return "Unknown"

    def create_agent(
        self,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        chat_message_store_factory: Callable[[], ChatMessageStoreProtocol] | None = None,
        context_providers: ContextProvider | list[ContextProvider] | AggregateContextProvider | None = None,
        middleware: Middleware | list[Middleware] | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str | int, float] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        model_id: str | None = None,
        presence_penalty: float | None = None,
        response_format: type[BaseModel] | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        store: bool | None = None,
        temperature: float | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = "auto",
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_chat_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ChatAgent":
        """このクライアントでChatAgentを作成します。

        これは便利なメソッドで、このチャットクライアントが既に設定されたChatAgentインスタンスを作成します。

        Keyword Args:
            id: エージェントの一意の識別子。指定しない場合は自動生成されます。
            name: エージェントの名前。
            description: エージェントの目的の簡単な説明。
            instructions: エージェントへの任意の指示。
                これはシステムメッセージとしてチャットクライアントサービスに送信されるメッセージに含まれます。
            chat_message_store_factory: ChatMessageStoreProtocolのインスタンスを作成するファクトリ関数。
                指定しない場合はデフォルトのインメモリストアが使用されます。
            context_providers: エージェント呼び出し時に含めるコンテキストプロバイダー。
            middleware: エージェントおよび関数呼び出しをインターセプトするミドルウェアのリスト。
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: エージェントに使用するmodel_id。
            presence_penalty: 使用するpresence penalty。
            response_format: レスポンスのフォーマット。
            seed: 使用するランダムシード。
            stop: リクエストの停止シーケンス。
            store: レスポンスを保存するかどうか。
            temperature: 使用するサンプリング温度。
            tool_choice: リクエストのツール選択。
            tools: リクエストに使用するツール。
            top_p: 使用するnucleus sampling確率。
            user: リクエストに関連付けるユーザー。
            additional_chat_options: chat_clientの``get_response``および``get_streaming_response``メソッドに渡されるその他の値の辞書。
                プロバイダー固有のパラメータを渡すのに使用可能。
            kwargs: その他のキーワード引数。``additional_properties``として保存されます。

        Returns:
            このチャットクライアントで設定されたChatAgentインスタンス。

        Examples:
            .. code-block:: python

                from agent_framework.clients import OpenAIChatClient

                # クライアントを作成
                client = OpenAIChatClient(model_id="gpt-4")

                # 便利メソッドでエージェントを作成
                agent = client.create_agent(
                    name="assistant", instructions="You are a helpful assistant.", temperature=0.7
                )

                # エージェントを実行
                response = await agent.run("Hello!")

        """
        from ._agents import ChatAgent

        return ChatAgent(
            chat_client=self,
            id=id,
            name=name,
            description=description,
            instructions=instructions,
            chat_message_store_factory=chat_message_store_factory,
            context_providers=context_providers,
            middleware=middleware,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            metadata=metadata,
            model_id=model_id,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            store=store,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_p=top_p,
            user=user,
            additional_chat_options=additional_chat_options,
            **kwargs,
        )
