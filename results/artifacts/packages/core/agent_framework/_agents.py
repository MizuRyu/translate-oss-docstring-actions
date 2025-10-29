# Copyright (c) Microsoft. All rights reserved.

import inspect
import re
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from copy import copy
from itertools import chain
from typing import Any, ClassVar, Literal, Protocol, TypeVar, cast, runtime_checkable
from uuid import uuid4

from mcp import types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field, create_model

from ._clients import BaseChatClient, ChatClientProtocol
from ._logging import get_logger
from ._mcp import LOG_LEVEL_MAPPING, MCPTool
from ._memory import AggregateContextProvider, Context, ContextProvider
from ._middleware import Middleware, use_agent_middleware
from ._serialization import SerializationMixin
from ._threads import AgentThread, ChatMessageStoreProtocol
from ._tools import FUNCTION_INVOKING_CHAT_CLIENT_MARKER, AIFunction, ToolProtocol
from ._types import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Role,
    ToolMode,
)
from .exceptions import AgentExecutionException, AgentInitializationError
from .observability import use_agent_observability

if sys.version_info >= (3, 12):
    from typing import override  # type: ignore # pragma: no cover
else:
    from typing_extensions import override  # type: ignore[import] # pragma: no cover
if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover

logger = get_logger("agent_framework")

TThreadType = TypeVar("TThreadType", bound="AgentThread")


def _sanitize_agent_name(agent_name: str | None) -> str | None:
    """関数名として使用するためにagent名をサニタイズします。

    スペースや特殊文字をアンダースコアに置き換え、
    有効なPython識別子を作成します。

    Args:
        agent_name: サニタイズするagent名。

    Returns:
        無効な文字がアンダースコアに置き換えられたサニタイズ済みのagent名。
        入力がNoneの場合はNoneを返します。
        サニタイズ結果が空文字列（例: agent_name="@@@"）の場合はデフォルトで"agent"を返します。

    """
    if agent_name is None:
        return None

    # 英数字またはアンダースコアでない文字をすべてアンダースコアに置き換えます。
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", agent_name)

    # 連続する複数のアンダースコアを単一のアンダースコアに置き換えます。
    sanitized = re.sub(r"_+", "_", sanitized)

    # 先頭および末尾のアンダースコアを削除します。
    sanitized = sanitized.strip("_")

    # 空文字列の場合の処理を行います。
    if not sanitized:
        return "agent"

    # サニタイズされた名前が数字で始まる場合は先頭にアンダースコアを付加します。
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"

    return sanitized


__all__ = ["AgentProtocol", "BaseAgent", "ChatAgent"]


# region Agent Protocol


@runtime_checkable
class AgentProtocol(Protocol):
    """呼び出し可能なagentのためのプロトコル。

    このプロトコルはすべてのagentが実装すべきインターフェースを定義し、
    識別用のプロパティや実行用のメソッドを含みます。

    Note:
        プロトコルは構造的サブタイピング（ダックタイピング）を使用します。
        クラスはこのプロトコルを明示的に継承する必要はありません。
        これにより、Agent Frameworkのベースクラスを使用せずに
        完全にカスタムなagentを作成できます。

    Examples:
        .. code-block:: python

            from agent_framework import AgentProtocol


            # 必要なメソッドを実装する任意のクラスは互換性があります
            # AgentProtocolを継承したりフレームワーククラスを使う必要はありません
            class CustomAgent:
                def __init__(self):
                    self._id = "custom-agent-001"
                    self._name = "Custom Agent"

                @property
                def id(self) -> str:
                    return self._id

                @property
                def name(self) -> str | None:
                    return self._name

                @property
                def display_name(self) -> str:
                    return self.name or self.id

                @property
                def description(self) -> str | None:
                    return "A fully custom agent implementation"

                async def run(self, messages=None, *, thread=None, **kwargs):
                    # カスタム実装
                    from agent_framework import AgentRunResponse

                    return AgentRunResponse(messages=[], response_id="custom-response")

                def run_stream(self, messages=None, *, thread=None, **kwargs):
                    # カスタムストリーミング実装
                    async def _stream():
                        from agent_framework import AgentRunResponseUpdate

                        yield AgentRunResponseUpdate()

                    return _stream()

                def get_new_thread(self, **kwargs):
                    # 独自のスレッド実装を返す
                    return {"id": "custom-thread", "messages": []}


            # インスタンスがプロトコルを満たすことを検証
            instance = CustomAgent()
            assert isinstance(instance, AgentProtocol)

    """

    @property
    def id(self) -> str:
        """AgentのIDを返します。"""
        ...

    @property
    def name(self) -> str | None:
        """Agentの名前を返します。"""
        ...

    @property
    def display_name(self) -> str:
        """Agentの表示名を返します。"""
        ...

    @property
    def description(self) -> str | None:
        """Agentの説明を返します。"""
        ...

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Agentからのレスポンスを取得します。

        このメソッドはAgentの実行の最終結果を単一のAgentRunResponseオブジェクトとして返します。
        呼び出し元は最終結果が利用可能になるまでブロックされます。

        注意: ストリーミングレスポンスの場合は、run_streamメソッドを使用してください。
        これは中間ステップと最終結果をAgentRunResponseUpdateオブジェクトのストリームとして返します。
        最終結果のみをストリーミングすることは、最終結果の利用可能なタイミングが不明であり、
        その時点まで呼び出し元をブロックすることがストリーミングシナリオでは望ましくないため、実現不可能です。

        Args:
            messages: Agentに送信するメッセージ。

        Keyword Args:
            thread: メッセージに関連付けられた会話スレッド。
            kwargs: 追加のキーワード引数。

        Returns:
            Agentのレスポンスアイテム。

        """
        ...

    def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """エージェントをストリームとして実行します。

        このメソッドは、エージェントの実行の中間ステップと最終結果を
        AgentRunResponseUpdateオブジェクトのストリームとして呼び出し元に返します。

        注意: AgentRunResponseUpdateオブジェクトはメッセージのチャンクを含みます。

        Args:
            messages: エージェントに送信するメッセージ。

        Keyword Args:
            thread: メッセージに関連付けられた会話スレッド。
            kwargs: 追加のキーワード引数。

        Yields:
            エージェントのレスポンスアイテム。

        """
        ...

    def get_new_thread(self, **kwargs: Any) -> AgentThread:
        """エージェントの新しい会話スレッドを作成します。"""
        ...


# region BaseAgent


class BaseAgent(SerializationMixin):
    """すべてのAgent Frameworkエージェントの基底クラス。

    このクラスは、コンテキストプロバイダー、ミドルウェアサポート、スレッド管理など、
    エージェント実装のコア機能を提供します。

    注意:
        BaseAgentはAgentProtocolで要求される``run()``, ``run_stream()``などのメソッドを実装していないため、
        直接インスタンス化できません。
        ChatAgentのような具体的な実装を使用するか、サブクラスを作成してください。

    Examples:
        .. code-block:: python

            from agent_framework import BaseAgent, AgentThread, AgentRunResponse


            # プロトコルを実装する具体的なサブクラスを作成
            class SimpleAgent(BaseAgent):
                async def run(self, messages=None, *, thread=None, **kwargs):
                    # カスタム実装
                    return AgentRunResponse(messages=[], response_id="simple-response")

                def run_stream(self, messages=None, *, thread=None, **kwargs):
                    async def _stream():
                        # カスタムストリーミング実装
                        yield AgentRunResponseUpdate()

                    return _stream()


            # 具体的なサブクラスをインスタンス化
            agent = SimpleAgent(name="my-agent", description="A simple agent implementation")

            # 特定のIDと追加プロパティで作成
            agent = SimpleAgent(
                id="custom-id-123",
                name="configured-agent",
                description="An agent with custom configuration",
                additional_properties={"version": "1.0", "environment": "production"},
            )

            # エージェントのプロパティにアクセス
            print(agent.id)  # カスタムまたは自動生成されたUUID
            print(agent.display_name)  # nameまたはidを返す

    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"additional_properties"}

    def __init__(
        self,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: ContextProvider | Sequence[ContextProvider] | None = None,
        middleware: Middleware | Sequence[Middleware] | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """BaseAgentインスタンスを初期化します。

        Keyword Args:
            id: エージェントの一意の識別子。指定がない場合は新しいUUIDが生成されます。
            name: エージェントの名前。Noneも可。
            description: エージェントの説明。
            context_providers: エージェント呼び出し時に含める複数のコンテキストプロバイダーのコレクション。
            middleware: エージェントおよび関数呼び出しをインターセプトするミドルウェアのリスト。
            additional_properties: エージェントに設定する追加プロパティ。
            kwargs: 追加のキーワード引数（additional_propertiesにマージされます）。

        """
        if id is None:
            id = str(uuid4())
        self.id = id
        self.name = name
        self.description = description
        self.context_provider = self._prepare_context_providers(context_providers)
        if middleware is None or isinstance(middleware, Sequence):
            self.middleware: list[Middleware] | None = cast(list[Middleware], middleware) if middleware else None
        else:
            self.middleware = [middleware]

        # kwargsをadditional_propertiesにマージします
        self.additional_properties: dict[str, Any] = cast(dict[str, Any], additional_properties or {})
        self.additional_properties.update(kwargs)

    async def _notify_thread_of_new_messages(
        self,
        thread: AgentThread,
        input_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage],
    ) -> None:
        """スレッドに新しいメッセージを通知します。

        これにより、スレッド上の潜在的なコンテキストプロバイダーの呼び出しメソッドも呼ばれます。

        Args:
            thread: 新しいメッセージを通知するスレッド。
            input_messages: 通知する入力メッセージ。
            response_messages: 通知するレスポンスメッセージ。

        """
        if isinstance(input_messages, ChatMessage) or len(input_messages) > 0:
            await thread.on_new_messages(input_messages)
        if isinstance(response_messages, ChatMessage) or len(response_messages) > 0:
            await thread.on_new_messages(response_messages)
        if thread.context_provider:
            await thread.context_provider.invoked(input_messages, response_messages)

    @property
    def display_name(self) -> str:
        """エージェントの表示名を返します。

        nameが存在すればそれを返し、そうでなければidを返します。

        """
        return self.name or self.id

    def get_new_thread(self, **kwargs: Any) -> AgentThread:
        """エージェントに対応した新しいAgentThreadインスタンスを返します。

        Keyword Args:
            kwargs: AgentThreadに渡される追加のキーワード引数。

        Returns:
            エージェントのコンテキストプロバイダーで構成された新しいAgentThreadインスタンス。

        """
        return AgentThread(**kwargs, context_provider=self.context_provider)

    async def deserialize_thread(self, serialized_thread: Any, **kwargs: Any) -> AgentThread:
        """シリアライズされた状態からスレッドをデシリアライズします。

        Args:
            serialized_thread: シリアライズされたスレッドデータ。

        Keyword Args:
            kwargs: 追加のキーワード引数。

        Returns:
            シリアライズ状態から復元された新しいAgentThreadインスタンス。

        """
        thread: AgentThread = self.get_new_thread()
        await thread.update_from_thread_state(serialized_thread, **kwargs)
        return thread

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_name: str = "task",
        arg_description: str | None = None,
        stream_callback: Callable[[AgentRunResponseUpdate], None]
        | Callable[[AgentRunResponseUpdate], Awaitable[None]]
        | None = None,
    ) -> AIFunction[BaseModel, str]:
        """このエージェントをラップするAIFunctionツールを作成します。

        Keyword Args:
            name: ツールの名前。Noneの場合はエージェントの名前を使用します。
            description: ツールの説明。Noneの場合はエージェントの説明または空文字を使用します。
            arg_name: 関数引数の名前（デフォルトは"task"）。
            arg_description: 関数引数の説明。
                Noneの場合は"Task for {tool_name}"がデフォルト。
            stream_callback: ストリーミングレスポンス用のオプションのコールバック。指定があればrun_streamを使用します。

        Returns:
            他のエージェントがツールとして使用できるAIFunction。

        Raises:
            TypeError: エージェントがAgentProtocolを実装していない場合。
            ValueError: エージェントツール名が特定できない場合。

        Examples:
            .. code-block:: python

                from agent_framework import ChatAgent

                # エージェントを作成
                agent = ChatAgent(chat_client=client, name="research-agent", description="Performs research tasks")

                # エージェントをツールに変換
                research_tool = agent.as_tool()

                # 別のエージェントでツールを使用
                coordinator = ChatAgent(chat_client=client, name="coordinator", tools=research_tool)

        """
        # selfがAgentProtocolを実装していることを検証します
        if not isinstance(self, AgentProtocol):
            raise TypeError(f"Agent {self.__class__.__name__} must implement AgentProtocol to be used as a tool")

        tool_name = name or _sanitize_agent_name(self.name)
        if tool_name is None:
            raise ValueError("Agent tool name cannot be None. Either provide a name parameter or set the agent's name.")
        tool_description = description or self.description or ""
        argument_description = arg_description or f"Task for {tool_name}"

        # 指定されたarg_nameで動的入力モデルを作成します
        field_info = Field(..., description=argument_description)
        model_name = f"{name or _sanitize_agent_name(self.name) or 'agent'}_task"
        input_model = create_model(model_name, **{arg_name: (str, field_info)})  # type: ignore[call-overload]

        # ラッパー外で一度だけコールバックがasyncかどうかをチェックします
        is_async_callback = stream_callback is not None and inspect.iscoroutinefunction(stream_callback)

        async def agent_wrapper(**kwargs: Any) -> str:
            """エージェントを呼び出すラッパー関数です。"""
            # 指定されたarg_nameを使ってkwargsから入力を抽出します
            input_text = kwargs.get(arg_name, "")

            if stream_callback is None:
                # 非ストリーミングモードを使用します
                return (await self.run(input_text)).text

            # ストリーミングモードを使用します - 更新を蓄積し最終レスポンスを作成します
            response_updates: list[AgentRunResponseUpdate] = []
            async for update in self.run_stream(input_text):
                response_updates.append(update)
                if is_async_callback:
                    await stream_callback(update)  # type: ignore[misc]
                else:
                    stream_callback(update)

            # 蓄積された更新から最終テキストを作成します
            return AgentRunResponse.from_agent_run_response_updates(response_updates).text

        return AIFunction(
            name=tool_name,
            description=tool_description,
            func=agent_wrapper,
            input_model=input_model,  # type: ignore
        )

    def _normalize_messages(
        self,
        messages: str | ChatMessage | Sequence[str] | Sequence[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        if messages is None:
            return []

        if isinstance(messages, str):
            return [ChatMessage(role=Role.USER, text=messages)]

        if isinstance(messages, ChatMessage):
            return [messages]

        return [ChatMessage(role=Role.USER, text=msg) if isinstance(msg, str) else msg for msg in messages]

    def _prepare_context_providers(
        self,
        context_providers: ContextProvider | Sequence[ContextProvider] | None = None,
    ) -> AggregateContextProvider | None:
        if not context_providers:
            return None

        if isinstance(context_providers, AggregateContextProvider):
            return context_providers

        return AggregateContextProvider(context_providers)


# region ChatAgent


@use_agent_middleware
@use_agent_observability
class ChatAgent(BaseAgent):
    """Chat Client Agentです。

    これは言語モデルと対話するためにチャットクライアントを使用する主要なエージェント実装です。
    ツール、コンテキストプロバイダー、ミドルウェアをサポートし、
    ストリーミングおよび非ストリーミングのレスポンスの両方に対応しています。

    Examples:
        基本的な使用例:

        .. code-block:: python

            from agent_framework import ChatAgent
            from agent_framework.clients import OpenAIChatClient

            # 基本的なチャットエージェントを作成
            client = OpenAIChatClient(model_id="gpt-4")
            agent = ChatAgent(chat_client=client, name="assistant", description="A helpful assistant")

            # シンプルなメッセージでエージェントを実行
            response = await agent.run("Hello, how are you?")
            print(response.text)

        ツールとストリーミングを使う例:

        .. code-block:: python

            # ツールと指示を持つエージェントを作成
            def get_weather(location: str) -> str:
                return f"The weather in {location} is sunny."


            agent = ChatAgent(
                chat_client=client,
                name="weather-agent",
                instructions="You are a weather assistant.",
                tools=get_weather,
                temperature=0.7,
                max_tokens=500,
            )

            # ストリーミングレスポンスを使用
            async for update in agent.run_stream("What's the weather in Paris?"):
                print(update.text, end="")

        プロバイダー固有の追加オプションを使う例:

        .. code-block:: python

            agent = ChatAgent(
                chat_client=client,
                name="reasoning-agent",
                instructions="You are a reasoning assistant.",
                model_id="gpt-5",
                temperature=0.7,
                max_tokens=500,
                additional_chat_options={
                    "reasoning": {"effort": "high", "summary": "concise"}
                },  # OpenAI Responses固有のオプション
            )

            # ストリーミングレスポンスを使用
            async for update in agent.run_stream("How do you prove the pythagorean theorem?"):
                print(update.text, end="")

    """

    AGENT_SYSTEM_NAME: ClassVar[str] = "microsoft.agent_framework"

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        instructions: str | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        chat_message_store_factory: Callable[[], ChatMessageStoreProtocol] | None = None,
        conversation_id: str | None = None,
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
    ) -> None:
        """ChatAgentインスタンスを初期化します。

        注意:
            frequency_penaltyからrequest_kwargsまでのパラメータはチャットクライアント呼び出しに使用されます。
            これらは両方のrunメソッドにも渡せます。
            両方が設定された場合、runメソッドに渡されたものが優先されます。

        Args:
            chat_client: エージェントで使用するチャットクライアント。
            instructions: エージェントへのオプションの指示。
                これらはチャットクライアントサービスに送信されるメッセージのシステムメッセージとして含まれます。

        Keyword Args:
            id: エージェントの一意識別子。指定がなければ自動生成されます。
            name: エージェントの名前。
            description: エージェントの目的の簡単な説明。
            chat_message_store_factory: ChatMessageStoreProtocolのインスタンスを作成するファクトリ関数。
                指定しない場合はデフォルトのインメモリストアが使用されます。
            conversation_id: サービス管理スレッドの会話ID。
                chat_message_store_factoryと同時に使用できません。
            context_providers: エージェント呼び出し時に含める複数のコンテキストプロバイダーのコレクション。
            middleware: エージェントおよび関数呼び出しをインターセプトするミドルウェアのリスト。
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加メタデータ。
            model_id: エージェントで使用するmodel_id。
            presence_penalty: 使用するpresence penalty。
            response_format: レスポンスのフォーマット。
            seed: 使用するランダムシード。
            stop: リクエストの停止シーケンス。
            store: レスポンスを保存するかどうか。
            temperature: 使用するサンプリング温度。
            tool_choice: リクエストのツール選択。
            tools: リクエストで使用するツール。
            top_p: 使用するnucleus sampling確率。
            user: リクエストに関連付けるユーザー。
            additional_chat_options: chat_clientの``get_response``および``get_streaming_response``メソッドに渡されるその他の値の辞書。
                プロバイダー固有のパラメータを渡すために使用可能。
            kwargs: その他のキーワード引数。``additional_properties``として保存されます。

        Raises:
            AgentInitializationError: conversation_idとchat_message_store_factoryの両方が指定された場合。

        """
        if conversation_id is not None and chat_message_store_factory is not None:
            raise AgentInitializationError(
                "Cannot specify both conversation_id and chat_message_store_factory. "
                "Use conversation_id for service-managed threads or chat_message_store_factory for local storage."
            )

        if not hasattr(chat_client, FUNCTION_INVOKING_CHAT_CLIENT_MARKER) and isinstance(chat_client, BaseChatClient):
            logger.warning(
                "The provided chat client does not support function invoking, this might limit agent capabilities."
            )

        super().__init__(
            id=id,
            name=name,
            description=description,
            context_providers=context_providers,
            middleware=middleware,
            **kwargs,
        )
        self.chat_client = chat_client
        self.chat_message_store_factory = chat_message_store_factory

        # ここではMCPサーバーを無視し、別に保存します。 実行時にそれらの関数をtoolsリストに追加します。
        normalized_tools: list[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]] = (  # type:ignore[reportUnknownVariableType]
            [] if tools is None else tools if isinstance(tools, list) else [tools]  # type: ignore[list-item]
        )
        self._local_mcp_tools = [tool for tool in normalized_tools if isinstance(tool, MCPTool)]
        agent_tools = [tool for tool in normalized_tools if not isinstance(tool, MCPTool)]
        self.chat_options = ChatOptions(
            model_id=model_id,
            conversation_id=conversation_id,
            frequency_penalty=frequency_penalty,
            instructions=instructions,
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
            tools=agent_tools,
            top_p=top_p,
            user=user,
            additional_properties=additional_chat_options or {},  # type: ignore
        )
        self._async_exit_stack = AsyncExitStack()
        self._update_agent_name()

    async def __aenter__(self) -> "Self":
        """非同期コンテキストマネージャに入ります。

        chat_clientまたはlocal_mcp_toolsのいずれかがコンテキストマネージャであれば、
        適切なクリーンアップを保証するためにasync exit stackに入ります。

        注意:
            このリストは将来的に拡張される可能性があります。

        Returns:
            ChatAgentインスタンス。

        """
        for context_manager in chain([self.chat_client], self._local_mcp_tools):
            if isinstance(context_manager, AbstractAsyncContextManager):
                await self._async_exit_stack.enter_async_context(context_manager)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """非同期コンテキストマネージャーを終了します。

        すべてのコンテキストマネージャーが適切に終了されるように、非同期のexitスタックを閉じます。

        Args:
            exc_type: 例外が発生した場合の例外タイプ、そうでなければNone。
            exc_val: 例外が発生した場合の例外値、そうでなければNone。
            exc_tb: 例外が発生した場合の例外トレースバック、そうでなければNone。
        """
        await self._async_exit_stack.aclose()

    def _update_agent_name(self) -> None:
        """チャットクライアント内のagent名を更新します。

        チャットクライアントがagent名の更新をサポートしているかをチェックします。
        実装はすでにagent名が定義されているかを確認し、定義されていなければこの値に設定するべきです。
        """
        if hasattr(self.chat_client, "_update_agent_name") and callable(self.chat_client._update_agent_name):  # type: ignore[reportAttributeAccessIssue, attr-defined]
            self.chat_client._update_agent_name(self.name)  # type: ignore[reportAttributeAccessIssue, attr-defined]

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
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
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = None,
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | list[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_chat_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """指定されたメッセージとオプションでagentを実行します。

        Note:
            ``agent.run()``を常に直接呼び出すわけではないため（ワークフローを通じて呼ばれます）、
            agentコンストラクタでチャットクライアントのすべてのパラメータのデフォルト値を設定することを推奨します。
            両方のパラメータが使用された場合、runメソッドに渡されたものが優先されます。

        Args:
            messages: 処理するメッセージ。

        Keyword Args:
            thread: agentに使用するスレッド。
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: agentに使用するmodel_id。
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
            additional_chat_options: リクエストに含める追加のプロパティ。
                プロバイダー固有のパラメータにこのフィールドを使用してください。
            kwargs: agentへの追加のキーワード引数。
                呼び出される関数にのみ渡されます。

        Returns:
            agentのレスポンスを含むAgentRunResponse。
        """
        input_messages = self._normalize_messages(messages)
        thread, run_chat_options, thread_messages = await self._prepare_thread_and_messages(
            thread=thread, input_messages=input_messages
        )
        normalized_tools: list[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]] = (  # type:ignore[reportUnknownVariableType]
            [] if tools is None else tools if isinstance(tools, list) else [tools]
        )
        agent_name = self._get_agent_name()

        # 最終的なツールリストを解決する（ランタイム提供ツール + ローカルMCPサーバーツール）
        final_tools: list[ToolProtocol | Callable[..., Any] | dict[str, Any]] = []
        # 元のパラメータを変更せずにtools引数をリストに正規化する
        for tool in normalized_tools:
            if isinstance(tool, MCPTool):
                if not tool.is_connected:
                    await self._async_exit_stack.enter_async_context(tool)
                final_tools.extend(tool.functions)  # type: ignore
            else:
                final_tools.append(tool)  # type: ignore

        for mcp_server in self._local_mcp_tools:
            if not mcp_server.is_connected:
                await self._async_exit_stack.enter_async_context(mcp_server)
            final_tools.extend(mcp_server.functions)

        co = run_chat_options & ChatOptions(
            model_id=model_id,
            conversation_id=thread.service_thread_id,
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
            tools=final_tools,
            top_p=top_p,
            user=user,
            **(additional_chat_options or {}),
        )
        response = await self.chat_client.get_response(messages=thread_messages, chat_options=co, **kwargs)

        await self._update_thread_with_type_and_conversation_id(thread, response.conversation_id)

        # レスポンス内の各メッセージに対してauthor名が設定されていることを保証する。
        for message in response.messages:
            if message.author_name is None:
                message.author_name = agent_name

        # chatResponseが成功した場合にのみスレッドに新しいメッセージを通知し、スレッド内のメッセージ状態の不整合を避ける。
        await self._notify_thread_of_new_messages(thread, input_messages, response.messages)
        return AgentRunResponse(
            messages=response.messages,
            response_id=response.response_id,
            created_at=response.created_at,
            usage_details=response.usage_details,
            value=response.value,
            raw_representation=response,
            additional_properties=response.additional_properties,
        )

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
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
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict[str, Any] | None = None,
        tools: ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | list[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None = None,
        top_p: float | None = None,
        user: str | None = None,
        additional_chat_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """指定されたメッセージとオプションでagentをストリーム実行します。

        Note:
            ``agent.run_stream()``を常に直接呼び出すわけではないため（オーケストレーションを通じて呼ばれます）、
            agentコンストラクタでチャットクライアントのすべてのパラメータのデフォルト値を設定することを推奨します。
            両方のパラメータが使用された場合、runメソッドに渡されたものが優先されます。

        Args:
            messages: 処理するメッセージ。

        Keyword Args:
            thread: agentに使用するスレッド。
            frequency_penalty: 使用するfrequency penalty。
            logit_bias: 使用するlogit bias。
            max_tokens: 生成する最大トークン数。
            metadata: リクエストに含める追加のメタデータ。
            model_id: agentに使用するmodel_id。
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
            additional_chat_options: リクエストに含める追加のプロパティ。
                プロバイダー固有のパラメータにこのフィールドを使用してください。
            kwargs: 追加のキーワード引数。
                呼び出される関数にのみ渡されます。

        Yields:
            agentのレスポンスのチャンクを含むAgentRunResponseUpdateオブジェクト。
        """
        input_messages = self._normalize_messages(messages)
        thread, run_chat_options, thread_messages = await self._prepare_thread_and_messages(
            thread=thread, input_messages=input_messages
        )
        agent_name = self._get_agent_name()
        # 最終的なツールリストを解決する（ランタイム提供ツール + ローカルMCPサーバーツール）
        final_tools: list[ToolProtocol | MutableMapping[str, Any] | Callable[..., Any]] = []
        normalized_tools: list[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]] = (  # type: ignore[reportUnknownVariableType]
            [] if tools is None else tools if isinstance(tools, list) else [tools]
        )
        # 元のパラメータを変更せずにtools引数をリストに正規化する
        for tool in normalized_tools:
            if isinstance(tool, MCPTool):
                if not tool.is_connected:
                    await self._async_exit_stack.enter_async_context(tool)
                final_tools.extend(tool.functions)  # type: ignore
            else:
                final_tools.append(tool)

        for mcp_server in self._local_mcp_tools:
            if not mcp_server.is_connected:
                await self._async_exit_stack.enter_async_context(mcp_server)
            final_tools.extend(mcp_server.functions)

        co = run_chat_options & ChatOptions(
            conversation_id=thread.service_thread_id,
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
            tools=final_tools,
            top_p=top_p,
            user=user,
            **(additional_chat_options or {}),
        )

        response_updates: list[ChatResponseUpdate] = []
        async for update in self.chat_client.get_streaming_response(
            messages=thread_messages, chat_options=co, **kwargs
        ):
            response_updates.append(update)

            if update.author_name is None:
                update.author_name = agent_name

            yield AgentRunResponseUpdate(
                contents=update.contents,
                role=update.role,
                author_name=update.author_name,
                response_id=update.response_id,
                message_id=update.message_id,
                created_at=update.created_at,
                additional_properties=update.additional_properties,
                raw_representation=update,
            )

        response = ChatResponse.from_chat_response_updates(response_updates, output_format_type=co.response_format)
        await self._update_thread_with_type_and_conversation_id(thread, response.conversation_id)
        await self._notify_thread_of_new_messages(thread, input_messages, response.messages)

    @override
    def get_new_thread(
        self,
        *,
        service_thread_id: str | None = None,
        **kwargs: Any,
    ) -> AgentThread:
        """agent用の新しい会話スレッドを取得します。

        service_thread_idを指定すると、そのスレッドはサービス管理としてマークされます。

        service_thread_idを指定せず、agentにconversation_idが設定されている場合、そのconversation_idを使ってサービス管理スレッドが作成されます。

        service_thread_idを指定せず、agentにchat_message_store_factoryが設定されている場合、そのファクトリを使ってスレッドのメッセージストアが作成され、スレッドはローカルで管理されます。

        どちらも存在しない場合、スレッドはサービスIDやメッセージストアなしで作成されます。
        これは、このスレッドでagentを実行した際の使用状況に基づいて更新されます。
        ``store=True``で実行すると、レスポンスにthread_idが含まれ、それが設定されます。
        そうでなければ、デフォルトのファクトリからメッセージストアが作成されます。

        Keyword Args:
            service_thread_id: オプションのサービス管理スレッドID。
            kwargs: 現在は使用されていません。

        Returns:
            新しいAgentThreadインスタンス。
        """
        if service_thread_id is not None:
            return AgentThread(
                service_thread_id=service_thread_id,
                context_provider=self.context_provider,
            )
        if self.chat_options.conversation_id is not None:
            return AgentThread(
                service_thread_id=self.chat_options.conversation_id,
                context_provider=self.context_provider,
            )
        if self.chat_message_store_factory is not None:
            return AgentThread(
                message_store=self.chat_message_store_factory(),
                context_provider=self.context_provider,
            )
        return AgentThread(context_provider=self.context_provider)

    def as_mcp_server(
        self,
        *,
        server_name: str = "Agent",
        version: str | None = None,
        instructions: str | None = None,
        lifespan: Callable[["Server[Any]"], AbstractAsyncContextManager[Any]] | None = None,
        **kwargs: Any,
    ) -> "Server[Any]":
        """agentインスタンスからMCPサーバーを作成します。

        この関数はagentインスタンスから自動的にMCPサーバーを作成し、提供された引数を使ってサーバーを設定し、agentを単一のMCPツールとして公開します。

        Keyword Args:
            server_name: サーバーの名前。
            version: サーバーのバージョン。
            instructions: サーバーで使用する指示。
            lifespan: サーバーの寿命。
            **kwargs: サーバー作成に渡す追加の引数。

        Returns:
            MCPサーバーインスタンス。
        """
        server_args: dict[str, Any] = {
            "name": server_name,
            "version": version,
            "instructions": instructions,
        }
        if lifespan:
            server_args["lifespan"] = lifespan
        if kwargs:
            server_args.update(kwargs)

        server: "Server[Any]" = Server(**server_args)  # type: ignore[call-arg]

        agent_tool = self.as_tool(name=self._get_agent_name())

        async def _log(level: types.LoggingLevel, data: Any) -> None:
            """サーバーとロガーにメッセージをログします。"""
            # ローカルロガーにログします。
            logger.log(LOG_LEVEL_MAPPING[level], data)
            if server and server.request_context and server.request_context.session:
                try:
                    await server.request_context.session.send_log_message(level=level, data=data)
                except Exception as e:
                    logger.error("Failed to send log message to server: %s", e)

        @server.list_tools()  # type: ignore
        async def _list_tools() -> list[types.Tool]:  # type: ignore
            """agent内のすべてのツールを一覧表示します。"""
            # PydanticモデルからJSONスキーマを取得します。
            schema = agent_tool.input_model.model_json_schema()

            tool = types.Tool(
                name=agent_tool.name,
                description=agent_tool.description,
                inputSchema={
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            )

            await _log(level="debug", data=f"Agent tool: {agent_tool}")
            return [tool]

        @server.call_tool()  # type: ignore
        async def _call_tool(  # type: ignore
            name: str, arguments: dict[str, Any]
        ) -> Sequence[types.TextContent | types.ImageContent | types.AudioContent | types.EmbeddedResource]:
            """agent内のツールを呼び出します。"""
            await _log(level="debug", data=f"Calling tool with args: {arguments}")

            if name != agent_tool.name:
                raise McpError(
                    error=types.ErrorData(
                        code=types.INTERNAL_ERROR,
                        message=f"Tool {name} not found",
                    ),
                )

            # 引数を使って入力モデルのインスタンスを作成します。
            try:
                args_instance = agent_tool.input_model(**arguments)
                result = await agent_tool.invoke(arguments=args_instance)
            except Exception as e:
                raise McpError(
                    error=types.ErrorData(
                        code=types.INTERNAL_ERROR,
                        message=f"Error calling tool {name}: {e}",
                    ),
                ) from e

            # 結果をMCPコンテンツに変換します。
            if isinstance(result, str):
                return [types.TextContent(type="text", text=result)]

            return [types.TextContent(type="text", text=str(result))]

        @server.set_logging_level()  # type: ignore
        async def _set_logging_level(level: types.LoggingLevel) -> None:  # type: ignore
            """サーバーのログレベルを設定します。"""
            logger.setLevel(LOG_LEVEL_MAPPING[level])
            # このログを新しい最小レベルで出力します。
            await _log(level=level, data=f"Log level set to {level}")

        return server

    async def _update_thread_with_type_and_conversation_id(
        self, thread: AgentThread, response_conversation_id: str | None
    ) -> None:
        """スレッドをストレージタイプと会話IDで更新します。

        Args:
            thread: 更新するスレッド。
            response_conversation_id: レスポンスからの会話ID（存在する場合）。

        Raises:
            AgentExecutionException: サービス管理スレッドに会話IDがない場合。
        """
        if response_conversation_id is None and thread.service_thread_id is not None:
            # サービス管理されたスレッドが渡されましたが、チャットクライアントから会話IDが返されませんでした。
            # これはサービスがサービス管理スレッドをサポートしていないことを意味し、 そのためこのスレッドはこのサービスで使用できません。
            raise AgentExecutionException(
                "Service did not return a valid conversation id when using a service managed thread."
            )

        if response_conversation_id is not None:
            # チャットクライアントから会話IDが返された場合、それはサービスがサーバー側スレッドストレージをサポートしていることを意味し、
            # スレッドを新しいIDで更新すべきです。
            thread.service_thread_id = response_conversation_id
            if thread.context_provider:
                await thread.context_provider.thread_created(thread.service_thread_id)
        elif thread.message_store is None and self.chat_message_store_factory is not None:
            # サービスがサーバー側スレッドストレージを使用しない場合（呼び出しからIDが返されなかった場合）、
            # スレッドにまだメッセージストアがなく、カスタムメッセージストアがある場合、
            # チャット履歴を保存するためにカスタムメッセージストアでスレッドを更新すべきです。
            thread.message_store = self.chat_message_store_factory()

    async def _prepare_thread_and_messages(
        self,
        *,
        thread: AgentThread | None,
        input_messages: list[ChatMessage] | None = None,
    ) -> tuple[AgentThread, ChatOptions, list[ChatMessage]]:
        """agent実行のためにスレッドとメッセージを準備します。

        このメソッドは会話スレッドを準備し、コンテキストプロバイダーデータをマージし、
        チャットクライアント用の最終的なメッセージリストを組み立てます。

        Keyword Args:
            thread: 会話スレッド。
            input_messages: 処理するメッセージ。

        Returns:
            タプルを返します:
                - 検証または作成されたスレッド
                - マージされたチャットオプション
                - チャットクライアント用の完全なメッセージリスト

        Raises:
            AgentExecutionException: スレッドとagentの会話IDが一致しない場合。
        """
        chat_options = copy(self.chat_options) if self.chat_options else ChatOptions()
        thread = thread or self.get_new_thread()
        if thread.service_thread_id and thread.context_provider:
            await thread.context_provider.thread_created(thread.service_thread_id)
        thread_messages: list[ChatMessage] = []
        if thread.message_store:
            thread_messages.extend(await thread.message_store.list_messages() or [])
        context: Context | None = None
        if self.context_provider:
            async with self.context_provider:
                context = await self.context_provider.invoking(input_messages or [])
                if context:
                    if context.messages:
                        thread_messages.extend(context.messages)
                    if context.tools:
                        if chat_options.tools is not None:
                            chat_options.tools.extend(context.tools)
                        else:
                            chat_options.tools = list(context.tools)
                    if context.instructions:
                        chat_options.instructions = (
                            context.instructions
                            if not chat_options.instructions
                            else f"{chat_options.instructions}\n{context.instructions}"
                        )
        thread_messages.extend(input_messages or [])
        if (
            thread.service_thread_id
            and chat_options.conversation_id
            and thread.service_thread_id != chat_options.conversation_id
        ):
            raise AgentExecutionException(
                "The conversation_id set on the agent is different from the one set on the thread, "
                "only one ID can be used for a run."
            )
        return thread, chat_options, thread_messages

    def _get_agent_name(self) -> str:
        """メッセージの帰属のためにagent名を取得します。

        Returns:
            agentの名前、設定されていない場合は'UnnamedAgent'。
        """
        return self.name or "UnnamedAgent"
