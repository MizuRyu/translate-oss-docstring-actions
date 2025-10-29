# Copyright (c) Microsoft. All rights reserved.

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Awaitable, Callable, MutableSequence
from enum import Enum
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeAlias, TypeVar

from ._serialization import SerializationMixin
from ._types import AgentRunResponse, AgentRunResponseUpdate, ChatMessage
from .exceptions import MiddlewareException

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ._agents import AgentProtocol
    from ._clients import ChatClientProtocol
    from ._threads import AgentThread
    from ._tools import AIFunction
    from ._types import ChatOptions, ChatResponse, ChatResponseUpdate


__all__ = [
    "AgentMiddleware",
    "AgentMiddlewares",
    "AgentRunContext",
    "ChatContext",
    "ChatMiddleware",
    "FunctionInvocationContext",
    "FunctionMiddleware",
    "Middleware",
    "agent_middleware",
    "chat_middleware",
    "function_middleware",
    "use_agent_middleware",
    "use_chat_middleware",
]

TAgent = TypeVar("TAgent", bound="AgentProtocol")
TChatClient = TypeVar("TChatClient", bound="ChatClientProtocol")
TContext = TypeVar("TContext")


class MiddlewareType(str, Enum):
    """ミドルウェアの種類を表すEnumです。

    内部的にミドルウェアの種類を識別および分類するために使用されます。

    """

    AGENT = "agent"
    FUNCTION = "function"
    CHAT = "chat"


class AgentRunContext(SerializationMixin):
    """Agentミドルウェア呼び出しのためのContextオブジェクトです。

    このコンテキストはAgentミドルウェアパイプラインを通過し、Agent呼び出しに関するすべての情報を含みます。

    Attributes:
        agent: 呼び出されるAgent。
        messages: Agentに送信されるメッセージ。
        thread: この呼び出しのAgentスレッド（あれば）。
        is_streaming: ストリーミング呼び出しかどうか。
        metadata: Agentミドルウェア間でデータを共有するためのメタデータ辞書。
        result: Agent実行結果。``next()``呼び出し後に実際の実行結果を観察できるか、
                実行結果を上書きするために設定可能。
                非ストリーミングの場合はAgentRunResponseであるべき。
                ストリーミングの場合はAsyncIterable[AgentRunResponseUpdate]であるべき。
        terminate: 現在のミドルウェア後に実行を終了するかどうかのフラグ。
                Trueに設定されると、制御がフレームワークに戻るとすぐに実行が停止します。
        kwargs: Agent実行メソッドに渡される追加のキーワード引数。

    Examples:
        .. code-block:: python

            from agent_framework import AgentMiddleware, AgentRunContext


            class LoggingMiddleware(AgentMiddleware):
                async def process(self, context: AgentRunContext, next):
                    print(f"Agent: {context.agent.name}")
                    print(f"Messages: {len(context.messages)}")
                    print(f"Thread: {context.thread}")
                    print(f"Streaming: {context.is_streaming}")

                    # メタデータを保存
                    context.metadata["start_time"] = time.time()

                    # 実行を継続
                    await next(context)

                    # 実行後に結果にアクセス
                    print(f"Result: {context.result}")

    """

    INJECTABLE: ClassVar[set[str]] = {"agent", "thread", "result"}

    def __init__(
        self,
        agent: "AgentProtocol",
        messages: list[ChatMessage],
        thread: "AgentThread | None" = None,
        is_streaming: bool = False,
        metadata: dict[str, Any] | None = None,
        result: AgentRunResponse | AsyncIterable[AgentRunResponseUpdate] | None = None,
        terminate: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """AgentRunContextを初期化します。

        Args:
            agent: 呼び出されるAgent。
            messages: Agentに送信されるメッセージ。
            thread: この呼び出しのAgentスレッド（あれば）。
            is_streaming: ストリーミング呼び出しかどうか。
            metadata: Agentミドルウェア間でデータを共有するためのメタデータ辞書。
            result: Agent実行結果。
            terminate: 現在のミドルウェア後に実行を終了するかどうかのフラグ。
            kwargs: Agent実行メソッドに渡される追加のキーワード引数。

        """
        self.agent = agent
        self.messages = messages
        self.thread = thread
        self.is_streaming = is_streaming
        self.metadata = metadata if metadata is not None else {}
        self.result = result
        self.terminate = terminate
        self.kwargs = kwargs if kwargs is not None else {}


class FunctionInvocationContext(SerializationMixin):
    """関数ミドルウェア呼び出しのためのContextオブジェクトです。

    このコンテキストは関数ミドルウェアパイプラインを通過し、関数呼び出しに関するすべての情報を含みます。

    Attributes:
        function: 呼び出される関数。
        arguments: 関数の検証済み引数。
        metadata: 関数ミドルウェア間でデータを共有するためのメタデータ辞書。
        result: 関数実行結果。``next()``呼び出し後に実際の実行結果を観察できるか、
                実行結果を上書きするために設定可能。
        terminate: 現在のミドルウェア後に実行を終了するかどうかのフラグ。
                Trueに設定されると、制御がフレームワークに戻るとすぐに実行が停止します。
        kwargs: この関数を呼び出したチャットメソッドに渡される追加のキーワード引数。

    Examples:
        .. code-block:: python

            from agent_framework import FunctionMiddleware, FunctionInvocationContext


            class ValidationMiddleware(FunctionMiddleware):
                async def process(self, context: FunctionInvocationContext, next):
                    print(f"Function: {context.function.name}")
                    print(f"Arguments: {context.arguments}")

                    # 引数を検証
                    if not self.validate(context.arguments):
                        context.result = {"error": "Validation failed"}
                        context.terminate = True
                        return

                    # 実行を継続
                    await next(context)

    """

    INJECTABLE: ClassVar[set[str]] = {"function", "arguments", "result"}

    def __init__(
        self,
        function: "AIFunction[Any, Any]",
        arguments: "BaseModel",
        metadata: dict[str, Any] | None = None,
        result: Any = None,
        terminate: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """FunctionInvocationContextを初期化します。

        Args:
            function: 呼び出される関数。
            arguments: 関数の検証済み引数。
            metadata: 関数ミドルウェア間でデータを共有するためのメタデータ辞書。
            result: 関数実行結果。
            terminate: 現在のミドルウェア後に実行を終了するかどうかのフラグ。
            kwargs: この関数を呼び出したチャットメソッドに渡される追加のキーワード引数。

        """
        self.function = function
        self.arguments = arguments
        self.metadata = metadata if metadata is not None else {}
        self.result = result
        self.terminate = terminate
        self.kwargs = kwargs if kwargs is not None else {}


class ChatContext(SerializationMixin):
    """Context object for chat middleware invocations.

    This context is passed through the chat middleware pipeline and contains all information
    about the chat request.

    Attributes:
        chat_client: The chat client being invoked.
        messages: The messages being sent to the chat client.
        chat_options: The options for the chat request.
        is_streaming: Whether this is a streaming invocation.
        metadata: Metadata dictionary for sharing data between chat middleware.
        result: Chat execution result. Can be observed after calling ``next()``
                to see the actual execution result or can be set to override the execution result.
                For non-streaming: should be ChatResponse.
                For streaming: should be AsyncIterable[ChatResponseUpdate].
        terminate: A flag indicating whether to terminate execution after current middleware.
                When set to True, execution will stop as soon as control returns to framework.
        kwargs: Additional keyword arguments passed to the chat client.

    Examples:
        .. code-block:: python

            from agent_framework import ChatMiddleware, ChatContext


            class TokenCounterMiddleware(ChatMiddleware):
                async def process(self, context: ChatContext, next):
                    print(f"Chat client: {context.chat_client.__class__.__name__}")
                    print(f"Messages: {len(context.messages)}")
                    print(f"Model: {context.chat_options.model_id}")

                    # Store metadata
                    context.metadata["input_tokens"] = self.count_tokens(context.messages)

                    # Continue execution
                    await next(context)

                    # Access result and count output tokens
                    if context.result:
                        context.metadata["output_tokens"] = self.count_tokens(context.result)
    """

    INJECTABLE: ClassVar[set[str]] = {"chat_client", "result"}

    def __init__(
        self,
        chat_client: "ChatClientProtocol",
        messages: "MutableSequence[ChatMessage]",
        chat_options: "ChatOptions",
        is_streaming: bool = False,
        metadata: dict[str, Any] | None = None,
        result: "ChatResponse | AsyncIterable[ChatResponseUpdate] | None" = None,
        terminate: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the ChatContext.

        Args:
            chat_client: The chat client being invoked.
            messages: The messages being sent to the chat client.
            chat_options: The options for the chat request.
            is_streaming: Whether this is a streaming invocation.
            metadata: Metadata dictionary for sharing data between chat middleware.
            result: Chat execution result.
            terminate: A flag indicating whether to terminate execution after current middleware.
            kwargs: Additional keyword arguments passed to the chat client.
        """
        self.chat_client = chat_client
        self.messages = messages
        self.chat_options = chat_options
        self.is_streaming = is_streaming
        self.metadata = metadata if metadata is not None else {}
        self.result = result
        self.terminate = terminate
        self.kwargs = kwargs if kwargs is not None else {}


class AgentMiddleware(ABC):
    """Abstract base class for agent middleware that can intercept agent invocations.

    Agent middleware allows you to intercept and modify agent invocations before and after
    execution. You can inspect messages, modify context, override results, or terminate
    execution early.

    Note:
        AgentMiddleware is an abstract base class. You must subclass it and implement
        the ``process()`` method to create custom agent middleware.

    Examples:
        .. code-block:: python

            from agent_framework import AgentMiddleware, AgentRunContext, ChatAgent


            class RetryMiddleware(AgentMiddleware):
                def __init__(self, max_retries: int = 3):
                    self.max_retries = max_retries

                async def process(self, context: AgentRunContext, next):
                    for attempt in range(self.max_retries):
                        await next(context)
                        if context.result and not context.result.is_error:
                            break
                        print(f"Retry {attempt + 1}/{self.max_retries}")


            # Use with an agent
            agent = ChatAgent(chat_client=client, name="assistant", middleware=RetryMiddleware())
    """

    @abstractmethod
    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Awaitable[None]],
    ) -> None:
        """Process an agent invocation.

        Args:
            context: Agent invocation context containing agent, messages, and metadata.
                    Use context.is_streaming to determine if this is a streaming call.
                    Middleware can set context.result to override execution, or observe
                    the actual execution result after calling next().
                    For non-streaming: AgentRunResponse
                    For streaming: AsyncIterable[AgentRunResponseUpdate]
            next: Function to call the next middleware or final agent execution.
                  Does not return anything - all data flows through the context.

        Note:
            Middleware should not return anything. All data manipulation should happen
            within the context object. Set context.result to override execution,
            or observe context.result after calling next() for actual results.
        """
        ...


class FunctionMiddleware(ABC):
    """Abstract base class for function middleware that can intercept function invocations.

    Function middleware allows you to intercept and modify function/tool invocations before
    and after execution. You can validate arguments, cache results, log invocations, or
    override function execution.

    Note:
        FunctionMiddleware is an abstract base class. You must subclass it and implement
        the ``process()`` method to create custom function middleware.

    Examples:
        .. code-block:: python

            from agent_framework import FunctionMiddleware, FunctionInvocationContext, ChatAgent


            class CachingMiddleware(FunctionMiddleware):
                def __init__(self):
                    self.cache = {}

                async def process(self, context: FunctionInvocationContext, next):
                    cache_key = f"{context.function.name}:{context.arguments}"

                    # Check cache
                    if cache_key in self.cache:
                        context.result = self.cache[cache_key]
                        context.terminate = True
                        return

                    # Execute function
                    await next(context)

                    # Cache result
                    if context.result:
                        self.cache[cache_key] = context.result


            # Use with an agent
            agent = ChatAgent(chat_client=client, name="assistant", middleware=CachingMiddleware())
    """

    @abstractmethod
    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        """Process a function invocation.

        Args:
            context: Function invocation context containing function, arguments, and metadata.
                    Middleware can set context.result to override execution, or observe
                    the actual execution result after calling next().
            next: Function to call the next middleware or final function execution.
                  Does not return anything - all data flows through the context.

        Note:
            Middleware should not return anything. All data manipulation should happen
            within the context object. Set context.result to override execution,
            or observe context.result after calling next() for actual results.
        """
        ...


class ChatMiddleware(ABC):
    """Abstract base class for chat middleware that can intercept chat client requests.

    Chat middleware allows you to intercept and modify chat client requests before and after
    execution. You can modify messages, add system prompts, log requests, or override
    chat responses.

    Note:
        ChatMiddleware is an abstract base class. You must subclass it and implement
        the ``process()`` method to create custom chat middleware.

    Examples:
        .. code-block:: python

            from agent_framework import ChatMiddleware, ChatContext, ChatAgent


            class SystemPromptMiddleware(ChatMiddleware):
                def __init__(self, system_prompt: str):
                    self.system_prompt = system_prompt

                async def process(self, context: ChatContext, next):
                    # Add system prompt to messages
                    from agent_framework import ChatMessage

                    context.messages.insert(0, ChatMessage(role="system", content=self.system_prompt))

                    # Continue execution
                    await next(context)


            # Use with an agent
            agent = ChatAgent(
                chat_client=client, name="assistant", middleware=SystemPromptMiddleware("You are a helpful assistant.")
            )
    """

    @abstractmethod
    async def process(
        self,
        context: ChatContext,
        next: Callable[[ChatContext], Awaitable[None]],
    ) -> None:
        """Process a chat client request.

        Args:
            context: Chat invocation context containing chat client, messages, options, and metadata.
                    Use context.is_streaming to determine if this is a streaming call.
                    Middleware can set context.result to override execution, or observe
                    the actual execution result after calling next().
                    For non-streaming: ChatResponse
                    For streaming: AsyncIterable[ChatResponseUpdate]
            next: Function to call the next middleware or final chat execution.
                  Does not return anything - all data flows through the context.

        Note:
            Middleware should not return anything. All data manipulation should happen
            within the context object. Set context.result to override execution,
            or observe context.result after calling next() for actual results.
        """
        ...


# Pure function type definitions for convenience
AgentMiddlewareCallable = Callable[[AgentRunContext, Callable[[AgentRunContext], Awaitable[None]]], Awaitable[None]]

FunctionMiddlewareCallable = Callable[
    [FunctionInvocationContext, Callable[[FunctionInvocationContext], Awaitable[None]]], Awaitable[None]
]

ChatMiddlewareCallable = Callable[[ChatContext, Callable[[ChatContext], Awaitable[None]]], Awaitable[None]]

# Type alias for all middleware types
Middleware: TypeAlias = (
    AgentMiddleware
    | AgentMiddlewareCallable
    | FunctionMiddleware
    | FunctionMiddlewareCallable
    | ChatMiddleware
    | ChatMiddlewareCallable
)
AgentMiddlewares: TypeAlias = AgentMiddleware | AgentMiddlewareCallable

# region Middleware type markers for decorators


def agent_middleware(func: AgentMiddlewareCallable) -> AgentMiddlewareCallable:
    """Decorator to mark a function as agent middleware.

    This decorator explicitly identifies a function as agent middleware,
    which processes AgentRunContext objects.

    Args:
        func: The middleware function to mark as agent middleware.

    Returns:
        The same function with agent middleware marker.

    Examples:
        .. code-block:: python

            from agent_framework import agent_middleware, AgentRunContext, ChatAgent


            @agent_middleware
            async def logging_middleware(context: AgentRunContext, next):
                print(f"Before: {context.agent.name}")
                await next(context)
                print(f"After: {context.result}")


            # Use with an agent
            agent = ChatAgent(chat_client=client, name="assistant", middleware=logging_middleware)
    """
    # Add marker attribute to identify this as agent middleware
    func._middleware_type: MiddlewareType = MiddlewareType.AGENT  # type: ignore
    return func


def function_middleware(func: FunctionMiddlewareCallable) -> FunctionMiddlewareCallable:
    """Decorator to mark a function as function middleware.

    This decorator explicitly identifies a function as function middleware,
    which processes FunctionInvocationContext objects.

    Args:
        func: The middleware function to mark as function middleware.

    Returns:
        The same function with function middleware marker.

    Examples:
        .. code-block:: python

            from agent_framework import function_middleware, FunctionInvocationContext, ChatAgent


            @function_middleware
            async def logging_middleware(context: FunctionInvocationContext, next):
                print(f"Calling: {context.function.name}")
                await next(context)
                print(f"Result: {context.result}")


            # Use with an agent
            agent = ChatAgent(chat_client=client, name="assistant", middleware=logging_middleware)
    """
    # Add marker attribute to identify this as function middleware
    func._middleware_type: MiddlewareType = MiddlewareType.FUNCTION  # type: ignore
    return func


def chat_middleware(func: ChatMiddlewareCallable) -> ChatMiddlewareCallable:
    """Decorator to mark a function as chat middleware.

    This decorator explicitly identifies a function as chat middleware,
    which processes ChatContext objects.

    Args:
        func: The middleware function to mark as chat middleware.

    Returns:
        The same function with chat middleware marker.

    Examples:
        .. code-block:: python

            from agent_framework import chat_middleware, ChatContext, ChatAgent


            @chat_middleware
            async def logging_middleware(context: ChatContext, next):
                print(f"Messages: {len(context.messages)}")
                await next(context)
                print(f"Response: {context.result}")


            # Use with an agent
            agent = ChatAgent(chat_client=client, name="assistant", middleware=logging_middleware)
    """
    # Add marker attribute to identify this as chat middleware
    func._middleware_type: MiddlewareType = MiddlewareType.CHAT  # type: ignore
    return func


class MiddlewareWrapper(Generic[TContext]):
    """純粋関数をmiddlewareプロトコルオブジェクトに変換するための汎用ラッパー。

    このラッパーにより、関数ベースのmiddlewareをクラスベースのmiddlewareと共に使用できるようにし、統一されたインターフェースを提供します。

    Type Parameters:
        TContext: このmiddlewareが操作するコンテキストオブジェクトの型。
    """

    def __init__(self, func: Callable[[TContext, Callable[[TContext], Awaitable[None]]], Awaitable[None]]) -> None:
        self.func = func

    async def process(self, context: TContext, next: Callable[[TContext], Awaitable[None]]) -> None:
        await self.func(context, next)


class BaseMiddlewarePipeline(ABC):
    """middlewareパイプライン実行のための基底クラス。

    middlewareチェーンの構築と実行のための共通機能を提供します。
    """

    def __init__(self) -> None:
        """基底middlewareパイプラインを初期化します。"""
        self._middlewares: list[Any] = []

    @abstractmethod
    def _register_middleware(self, middleware: Any) -> None:
        """middlewareアイテムを登録します。

        サブクラスで実装する必要があります。

        Args:
            middleware: 登録するmiddleware。
        """
        ...

    @property
    def has_middlewares(self) -> bool:
        """登録されているmiddlewareがあるかどうかをチェックします。

        Returns:
            middlewareが登録されていればTrue、そうでなければFalse。
        """
        return bool(self._middlewares)

    def _register_middleware_with_wrapper(
        self,
        middleware: Any,
        expected_type: type,
    ) -> None:
        """自動ラッピングによる汎用middleware登録。

        必要に応じて呼び出し可能なmiddlewareをMiddlewareWrapperでラップします。

        Args:
            middleware: 登録するmiddlewareインスタンスまたは呼び出し可能オブジェクト。
            expected_type: 期待されるmiddlewareの基底クラスの型。
        """
        if isinstance(middleware, expected_type):
            self._middlewares.append(middleware)
        elif callable(middleware):
            self._middlewares.append(MiddlewareWrapper(middleware))  # type: ignore[arg-type]

    def _create_handler_chain(
        self,
        final_handler: Callable[[Any], Awaitable[Any]],
        result_container: dict[str, Any],
        result_key: str = "result",
    ) -> Callable[[Any], Awaitable[None]]:
        """middlewareハンドラのチェーンを作成します。

        Args:
            final_handler: 実行する最終ハンドラ。
            result_container: 結果を格納するコンテナ。
            result_key: 結果コンテナで使用するキー。

        Returns:
            チェーンの最初のハンドラ。
        """

        def create_next_handler(index: int) -> Callable[[Any], Awaitable[None]]:
            if index >= len(self._middlewares):

                async def final_wrapper(c: Any) -> None:
                    # 実際のハンドラを実行し、観測性のためにコンテキストを設定します。
                    result = await final_handler(c)
                    result_container[result_key] = result
                    c.result = result

                return final_wrapper

            middleware = self._middlewares[index]
            next_handler = create_next_handler(index + 1)

            async def current_handler(c: Any) -> None:
                await middleware.process(c, next_handler)

            return current_handler

        return create_next_handler(0)

    def _create_streaming_handler_chain(
        self,
        final_handler: Callable[[Any], Any],
        result_container: dict[str, Any],
        result_key: str = "result_stream",
    ) -> Callable[[Any], Awaitable[None]]:
        """ストリーミング操作のためのmiddlewareハンドラチェーンを作成します。

        Args:
            final_handler: 実行する最終ハンドラ。
            result_container: 結果を格納するコンテナ。
            result_key: 結果コンテナで使用するキー。

        Returns:
            チェーンの最初のハンドラ。
        """

        def create_next_handler(index: int) -> Callable[[Any], Awaitable[None]]:
            if index >= len(self._middlewares):

                async def final_wrapper(c: Any) -> None:
                    # terminateが設定されていれば、実行をスキップします。
                    if c.terminate:
                        return

                    # 実際のハンドラを実行し、観測性のためにコンテキストを設定します。 注意:
                    # final_handlerはストリーミングの場合、awaitableでない可能性があります。
                    try:
                        result = await final_handler(c)
                    except TypeError:
                        # 非awaitableケース（例：ジェネレータ関数）を処理します。
                        result = final_handler(c)
                    result_container[result_key] = result
                    c.result = result

                return final_wrapper

            middleware = self._middlewares[index]
            next_handler = create_next_handler(index + 1)

            async def current_handler(c: Any) -> None:
                await middleware.process(c, next_handler)
                # terminateが設定されていれば、パイプラインを継続しません。
                if c.terminate:
                    return

            return current_handler

        return create_next_handler(0)


class AgentMiddlewarePipeline(BaseMiddlewarePipeline):
    """Agent middlewareをチェーンで実行します。

    複数のagent middlewareを順に実行し、それぞれのmiddlewareがagentの呼び出しを処理し、
    次のmiddlewareに制御を渡すことを可能にします。
    """

    def __init__(self, middlewares: list[AgentMiddleware | AgentMiddlewareCallable] | None = None):
        """agent middlewareパイプラインを初期化します。

        Args:
            middlewares: パイプラインに含めるagent middlewareのリスト。
        """
        super().__init__()
        self._middlewares: list[AgentMiddleware] = []

        if middlewares:
            for middleware in middlewares:
                self._register_middleware(middleware)

    def _register_middleware(self, middleware: AgentMiddleware | AgentMiddlewareCallable) -> None:
        """agent middlewareアイテムを登録します。

        Args:
            middleware: 登録するagent middleware。
        """
        self._register_middleware_with_wrapper(middleware, AgentMiddleware)

    async def execute(
        self,
        agent: "AgentProtocol",
        messages: list[ChatMessage],
        context: AgentRunContext,
        final_handler: Callable[[AgentRunContext], Awaitable[AgentRunResponse]],
    ) -> AgentRunResponse | None:
        """非ストリーミング用のagent middlewareパイプラインを実行します。

        Args:
            agent: 呼び出されるagent。
            messages: agentに送信するメッセージ。
            context: agent呼び出しのコンテキスト。
            final_handler: 実際のagent実行を行う最終ハンドラ。

        Returns:
            全middlewareを通過した後のagentのレスポンス。
        """
        # contextをagentとmessagesで更新します。
        context.agent = agent
        context.messages = messages
        context.is_streaming = False

        if not self._middlewares:
            return await final_handler(context)

        # 最終結果を格納します。
        result_container: dict[str, AgentRunResponse | None] = {"result": None}

        # 終了と結果の上書きを処理するカスタム最終ハンドラ。
        async def agent_final_handler(c: AgentRunContext) -> AgentRunResponse:
            # terminateが設定されていれば、結果（Noneの場合もあり）を返します。
            if c.terminate:
                if c.result is not None and isinstance(c.result, AgentRunResponse):
                    return c.result
                return AgentRunResponse()
            # 実際のハンドラを実行し、観測性のためにコンテキストを設定します。
            return await final_handler(c)

        first_handler = self._create_handler_chain(agent_final_handler, result_container, "result")
        await first_handler(context)

        # 結果コンテナまたは上書きされた結果から結果を返します。
        if context.result is not None and isinstance(context.result, AgentRunResponse):
            return context.result

        # 結果が設定されていない場合（next()が呼ばれていない場合）、空のAgentRunResponseを返します。
        response = result_container.get("result")
        if response is None:
            return AgentRunResponse()
        return response

    async def execute_stream(
        self,
        agent: "AgentProtocol",
        messages: list[ChatMessage],
        context: AgentRunContext,
        final_handler: Callable[[AgentRunContext], AsyncIterable[AgentRunResponseUpdate]],
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """ストリーミング用のagent middlewareパイプラインを実行します。

        Args:
            agent: 呼び出されるagent。
            messages: agentに送信するメッセージ。
            context: agent呼び出しのコンテキスト。
            final_handler: 実際のagentストリーミング実行を行う最終ハンドラ。

        Yields:
            全middlewareを通過した後のagentレスポンスの更新。
        """
        # contextをagentとmessagesで更新します。
        context.agent = agent
        context.messages = messages
        context.is_streaming = True

        if not self._middlewares:
            async for update in final_handler(context):
                yield update
            return

        # 最終結果を格納します。
        result_container: dict[str, AsyncIterable[AgentRunResponseUpdate] | None] = {"result_stream": None}

        first_handler = self._create_streaming_handler_chain(final_handler, result_container, "result_stream")
        await first_handler(context)

        # 結果コンテナまたは上書きされた結果のストリームからyieldします。
        if context.result is not None and hasattr(context.result, "__aiter__"):
            async for update in context.result:  # type: ignore
                yield update
            return

        result_stream = result_container["result_stream"]
        if result_stream is None:
            # 結果ストリームが設定されていない場合（next()が呼ばれていない場合）、何もyieldしません。
            return

        async for update in result_stream:
            yield update


class FunctionMiddlewarePipeline(BaseMiddlewarePipeline):
    """function middlewareをチェーンで実行します。

    複数のfunction middlewareを順に実行し、それぞれのmiddlewareがfunctionの呼び出しを処理し、
    次のmiddlewareに制御を渡すことを可能にします。
    """

    def __init__(self, middlewares: list[FunctionMiddleware | FunctionMiddlewareCallable] | None = None):
        """function middlewareパイプラインを初期化します。

        Args:
            middlewares: パイプラインに含めるfunction middlewareのリスト。
        """
        super().__init__()
        self._middlewares: list[FunctionMiddleware] = []

        if middlewares:
            for middleware in middlewares:
                self._register_middleware(middleware)

    def _register_middleware(self, middleware: FunctionMiddleware | FunctionMiddlewareCallable) -> None:
        """function middlewareアイテムを登録します。

        Args:
            middleware: 登録するfunction middleware。
        """
        self._register_middleware_with_wrapper(middleware, FunctionMiddleware)

    async def execute(
        self,
        function: Any,
        arguments: "BaseModel",
        context: FunctionInvocationContext,
        final_handler: Callable[[FunctionInvocationContext], Awaitable[Any]],
    ) -> Any:
        """function middlewareパイプラインを実行します。

        Args:
            function: 呼び出されるfunction。
            arguments: functionの検証済み引数。
            context: function呼び出しのコンテキスト。
            final_handler: 実際のfunction実行を行う最終ハンドラ。

        Returns:
            全middlewareを通過した後のfunctionの結果。
        """
        # contextをfunctionとargumentsで更新します。
        context.function = function
        context.arguments = arguments

        if not self._middlewares:
            return await final_handler(context)

        # 最終結果を格納します。
        result_container: dict[str, Any] = {"result": None}

        # 既存の結果を処理するカスタム最終ハンドラ。
        async def function_final_handler(c: FunctionInvocationContext) -> Any:
            # terminateが設定されていれば、実行をスキップし結果（Noneの場合もあり）を返します。
            if c.terminate:
                return c.result
            # 実際のハンドラを実行し、観測性のためにコンテキストを設定します。
            return await final_handler(c)

        first_handler = self._create_handler_chain(function_final_handler, result_container, "result")
        await first_handler(context)

        # 結果コンテナまたは上書きされた結果から結果を返します。
        if context.result is not None:
            return context.result
        return result_container["result"]


class ChatMiddlewarePipeline(BaseMiddlewarePipeline):
    """chat middlewareをチェーンで実行します。

    複数のchat middlewareを順に実行し、それぞれのmiddlewareがchatリクエストを処理し、
    次のmiddlewareに制御を渡すことを可能にします。
    """

    def __init__(self, middlewares: list[ChatMiddleware | ChatMiddlewareCallable] | None = None):
        """chat middlewareパイプラインを初期化します。

        Args:
            middlewares: パイプラインに含めるchat middlewareのリスト。
        """
        super().__init__()
        self._middlewares: list[ChatMiddleware] = []

        if middlewares:
            for middleware in middlewares:
                self._register_middleware(middleware)

    def _register_middleware(self, middleware: ChatMiddleware | ChatMiddlewareCallable) -> None:
        """chat middlewareアイテムを登録します。

        Args:
            middleware: 登録するchat middleware。
        """
        self._register_middleware_with_wrapper(middleware, ChatMiddleware)

    async def execute(
        self,
        chat_client: "ChatClientProtocol",
        messages: "MutableSequence[ChatMessage]",
        chat_options: "ChatOptions",
        context: ChatContext,
        final_handler: Callable[[ChatContext], Awaitable["ChatResponse"]],
        **kwargs: Any,
    ) -> "ChatResponse":
        """chat middlewareパイプラインを実行します。

        Args:
            chat_client: 呼び出されるchat client。
            messages: chat clientに送信するメッセージ。
            chat_options: chatリクエストのオプション。
            context: chat呼び出しのコンテキスト。
            final_handler: 実際のchat実行を行う最終ハンドラ。
            **kwargs: 追加のキーワード引数。

        Returns:
            全middlewareを通過した後のchatレスポンス。
        """
        # contextをchat client、messages、およびoptionsで更新します。
        context.chat_client = chat_client
        context.messages = messages
        context.chat_options = chat_options

        if not self._middlewares:
            return await final_handler(context)

        # 最終結果を格納します。
        result_container: dict[str, Any] = {"result": None}

        # 既存の結果を処理するカスタム最終ハンドラ。
        async def chat_final_handler(c: ChatContext) -> "ChatResponse":
            # terminateが設定されていれば、実行をスキップし結果（Noneの場合もあり）を返します。
            if c.terminate:
                return c.result  # type: ignore
            # 実際のハンドラを実行し、観測性のためにコンテキストを設定します。
            return await final_handler(c)

        first_handler = self._create_handler_chain(chat_final_handler, result_container, "result")
        await first_handler(context)

        # 結果コンテナまたは上書きされた結果から結果を返します。
        if context.result is not None:
            return context.result  # type: ignore
        return result_container["result"]  # type: ignore

    async def execute_stream(
        self,
        chat_client: "ChatClientProtocol",
        messages: "MutableSequence[ChatMessage]",
        chat_options: "ChatOptions",
        context: ChatContext,
        final_handler: Callable[[ChatContext], AsyncIterable["ChatResponseUpdate"]],
        **kwargs: Any,
    ) -> AsyncIterable["ChatResponseUpdate"]:
        """ストリーミング用のchat middlewareパイプラインを実行します。

        Args:
            chat_client: 呼び出されるchat client。
            messages: chat clientに送信するメッセージ。
            chat_options: chatリクエストのオプション。
            context: chat呼び出しのコンテキスト。
            final_handler: 実際のストリーミングchat実行を行う最終ハンドラ。
            **kwargs: 追加のキーワード引数。

        Yields:
            全middlewareを通過した後のchatレスポンスの更新。
        """
        # contextをchat client、messages、およびoptionsで更新します。
        context.chat_client = chat_client
        context.messages = messages
        context.chat_options = chat_options
        context.is_streaming = True

        if not self._middlewares:
            async for update in final_handler(context):
                yield update
            return

        # 最終結果のストリームを格納します。
        result_container: dict[str, Any] = {"result_stream": None}

        first_handler = self._create_streaming_handler_chain(final_handler, result_container, "result_stream")
        await first_handler(context)

        # 結果コンテナまたは上書きされた結果のストリームからyieldします。
        if context.result is not None and hasattr(context.result, "__aiter__"):
            async for update in context.result:  # type: ignore
                yield update
            return

        result_stream = result_container["result_stream"]
        if result_stream is None:
            # 結果ストリームが設定されていない場合（next()が呼ばれていない場合）、何もyieldしません。
            return

        async for update in result_stream:
            yield update


def _determine_middleware_type(middleware: Any) -> MiddlewareType:
    """デコレータおよび/またはパラメータ型注釈を使用してmiddlewareの種類を判定します。

    Args:
        middleware: 分析するmiddleware関数。

    Returns:
        MiddlewareType.AGENT、MiddlewareType.FUNCTION、またはMiddlewareType.CHATのいずれかでmiddlewareの種類を示します。

    Raises:
        MiddlewareException: middlewareの種類が判定できない場合や不一致がある場合。
    """
    # デコレータマーカーをチェックします。
    decorator_type: MiddlewareType | None = getattr(middleware, "_middleware_type", None)

    # パラメータ型注釈をチェックします。
    param_type: MiddlewareType | None = None
    try:
        sig = inspect.signature(middleware)
        params = list(sig.parameters.values())

        # 少なくとも2つのパラメータ（contextとnext）を持つ必要があります。
        if len(params) >= 2:
            first_param = params[0]
            if hasattr(first_param.annotation, "__name__"):
                annotation_name = first_param.annotation.__name__
                if annotation_name == "AgentRunContext":
                    param_type = MiddlewareType.AGENT
                elif annotation_name == "FunctionInvocationContext":
                    param_type = MiddlewareType.FUNCTION
                elif annotation_name == "ChatContext":
                    param_type = MiddlewareType.CHAT
        else:
            # パラメータが不足しているため、有効なmiddlewareではありません。
            raise MiddlewareException(
                f"Middleware function must have at least 2 parameters (context, next), "
                f"but {middleware.__name__} has {len(params)}"
            )
    except Exception as e:
        if isinstance(e, MiddlewareException):
            raise
        # シグネチャの検査に失敗しました - 他のチェックを続行します。
        pass

    if decorator_type and param_type:
        # デコレータとパラメータ型の両方が指定されている場合、それらは一致している必要があります。
        if decorator_type != param_type:
            raise MiddlewareException(
                f"Middleware type mismatch: decorator indicates '{decorator_type.value}' "
                f"but parameter type indicates '{param_type.value}' for function {middleware.__name__}"
            )
        return decorator_type

    if decorator_type:
        # デコレータのみが指定されている場合は、デコレータに依存します。
        return decorator_type

    if param_type:
        # パラメータ型のみが指定されている場合は、型に依存します。
        return param_type

    # デコレータもパラメータ型も指定されていない場合は例外をスローします。
    raise MiddlewareException(
        f"Cannot determine middleware type for function {middleware.__name__}. "
        f"Please either use @agent_middleware/@function_middleware/@chat_middleware decorators "
        f"or specify parameter types (AgentRunContext, FunctionInvocationContext, or ChatContext)."
    )


# agentクラスにmiddlewareサポートを追加するためのデコレータ。
def use_agent_middleware(agent_class: type[TAgent]) -> type[TAgent]:
    """agentクラスにmiddlewareサポートを追加するクラスデコレータ。

    このデコレータは任意のagentクラスにmiddleware機能を追加します。
    ``run()``および``run_stream()``メソッドをラップしてmiddleware実行を提供します。

    middleware実行は、``context.terminate``プロパティをTrueに設定することで任意の時点で終了できます。
    一度設定されると、パイプラインは制御がパイプラインに戻った時点でそれ以上のmiddlewareの実行を停止します。

    Note:
        このデコレータは組み込みのagentクラスには既に適用されています。
        カスタムagent実装を作成する場合にのみ使用してください。

    Args:
        agent_class: middlewareサポートを追加するagentクラス。

    Returns:
        middlewareサポートが追加された修正済みのagentクラス。

    Examples:
        .. code-block:: python

            from agent_framework import use_agent_middleware


            @use_agent_middleware
            class CustomAgent:
                async def run(self, messages, **kwargs):
                    # Agentの実装
                    pass

                async def run_stream(self, messages, **kwargs):
                    # ストリーミング実装
                    pass

    """
    # 元のメソッドを保存します。
    original_run = agent_class.run  # type: ignore[attr-defined]
    original_run_stream = agent_class.run_stream  # type: ignore[attr-defined]

    def _build_middleware_pipelines(
        agent_level_middlewares: Middleware | list[Middleware] | None,
        run_level_middlewares: Middleware | list[Middleware] | None = None,
    ) -> tuple[AgentMiddlewarePipeline, FunctionMiddlewarePipeline, list[ChatMiddleware | ChatMiddlewareCallable]]:
        """提供されたmiddlewareリストから新しいagentおよびfunction middlewareパイプラインを構築します。

        Args:
            agent_level_middlewares: Agentレベルのmiddleware（最初に実行されます）
            run_level_middlewares: Runレベルのmiddleware（agent middlewareの後に実行されます）

        """
        middleware = categorize_middleware(agent_level_middlewares, run_level_middlewares)

        return (
            AgentMiddlewarePipeline(middleware["agent"]),  # type: ignore[arg-type]
            FunctionMiddlewarePipeline(middleware["function"]),  # type: ignore[arg-type]
            middleware["chat"],  # type: ignore[return-value]
        )

    async def middleware_enabled_run(
        self: Any,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: Any = None,
        middleware: Middleware | list[Middleware] | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Middleware対応のrunメソッド。"""
        # 現在のmiddlewareコレクションとrunレベルmiddlewareから新しいmiddlewareパイプラインを構築します。
        agent_middleware = getattr(self, "middleware", None)

        agent_pipeline, function_pipeline, chat_middlewares = _build_middleware_pipelines(agent_middleware, middleware)

        # 利用可能な場合、function middlewareパイプラインをkwargsに追加します。
        if function_pipeline.has_middlewares:
            kwargs["_function_middleware_pipeline"] = function_pipeline

        # runレベル適用のためにchat middlewareをkwargs経由で渡します。
        if chat_middlewares:
            kwargs["middleware"] = chat_middlewares

        normalized_messages = self._normalize_messages(messages)

        # middlewareが利用可能な場合はmiddleware付きで実行します。
        if agent_pipeline.has_middlewares:
            context = AgentRunContext(
                agent=self,  # type: ignore[arg-type]
                messages=normalized_messages,
                thread=thread,
                is_streaming=False,
                kwargs=kwargs,
            )

            async def _execute_handler(ctx: AgentRunContext) -> AgentRunResponse:
                return await original_run(self, ctx.messages, thread=thread, **ctx.kwargs)  # type: ignore

            result = await agent_pipeline.execute(
                self,  # type: ignore[arg-type]
                normalized_messages,
                context,
                _execute_handler,
            )

            return result if result else AgentRunResponse()

        # middlewareがない場合は直接実行します。
        return await original_run(self, normalized_messages, thread=thread, **kwargs)  # type: ignore[return-value]

    def middleware_enabled_run_stream(
        self: Any,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: Any = None,
        middleware: Middleware | list[Middleware] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """Middleware対応のrun_streamメソッド。"""
        # 現在のmiddlewareコレクションとrunレベルmiddlewareから新しいmiddlewareパイプラインを構築します。
        agent_middleware = getattr(self, "middleware", None)
        agent_pipeline, function_pipeline, chat_middlewares = _build_middleware_pipelines(agent_middleware, middleware)

        # 利用可能な場合、function middlewareパイプラインをkwargsに追加します。
        if function_pipeline.has_middlewares:
            kwargs["_function_middleware_pipeline"] = function_pipeline

        # runレベル適用のためにchat middlewareをkwargs経由で渡します。
        if chat_middlewares:
            kwargs["middleware"] = chat_middlewares

        normalized_messages = self._normalize_messages(messages)

        # middlewareが利用可能な場合はmiddleware付きで実行します。
        if agent_pipeline.has_middlewares:
            context = AgentRunContext(
                agent=self,  # type: ignore[arg-type]
                messages=normalized_messages,
                thread=thread,
                is_streaming=True,
                kwargs=kwargs,
            )

            async def _execute_stream_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
                async for update in original_run_stream(self, ctx.messages, thread=thread, **ctx.kwargs):  # type: ignore[misc]
                    yield update

            async def _stream_generator() -> AsyncIterable[AgentRunResponseUpdate]:
                async for update in agent_pipeline.execute_stream(
                    self,  # type: ignore[arg-type]
                    normalized_messages,
                    context,
                    _execute_stream_handler,
                ):
                    yield update

            return _stream_generator()

        # middlewareがない場合は直接実行します。
        return original_run_stream(self, normalized_messages, thread=thread, **kwargs)  # type: ignore

    agent_class.run = update_wrapper(middleware_enabled_run, original_run)  # type: ignore
    agent_class.run_stream = update_wrapper(middleware_enabled_run_stream, original_run_stream)  # type: ignore

    return agent_class


def use_chat_middleware(chat_client_class: type[TChatClient]) -> type[TChatClient]:
    """chat clientクラスにmiddlewareサポートを追加するクラスデコレータ。

    このデコレータは任意のchat clientクラスにmiddleware機能を追加します。
    ``get_response()``および``get_streaming_response()``メソッドをラップしてmiddlewareの実行を提供します。

    注意:
        このデコレータは組み込みのchat clientクラスにはすでに適用されています。
        カスタムchat client実装を作成する場合にのみ使用してください。

    Args:
        chat_client_class: middlewareサポートを追加するchat clientクラス。

    Returns:
        middlewareサポートが追加された修正済みのchat clientクラス。

    Examples:
        .. code-block:: python

            from agent_framework import use_chat_middleware


            @use_chat_middleware
            class CustomChatClient:
                async def get_response(self, messages, **kwargs):
                    # Chat clientの実装
                    pass

                async def get_streaming_response(self, messages, **kwargs):
                    # Streamingの実装
                    pass

    """
    # 元のメソッドを保存します。
    original_get_response = chat_client_class.get_response
    original_get_streaming_response = chat_client_class.get_streaming_response

    async def middleware_enabled_get_response(
        self: Any,
        messages: Any,
        **kwargs: Any,
    ) -> Any:
        """Middleware対応のget_responseメソッド。"""
        # 呼び出しレベルまたはインスタンスレベルでmiddlewareが提供されているか確認します。
        call_middleware = kwargs.pop("middleware", None)
        instance_middleware = getattr(self, "middleware", None)

        # すべてのmiddlewareをマージし、タイプ別に分類します。
        middleware = categorize_middleware(instance_middleware, call_middleware)
        chat_middleware_list = middleware["chat"]  # type: ignore[assignment]

        # function呼び出しパイプライン用にfunction middlewareを抽出します。
        function_middleware_list = middleware["function"]

        # 存在する場合、function呼び出しシステムにfunction middlewareを渡します。
        if function_middleware_list:
            kwargs["_function_middleware_pipeline"] = FunctionMiddlewarePipeline(function_middleware_list)  # type: ignore[arg-type]

        # chat middlewareがない場合は元のメソッドを使用します。
        if not chat_middleware_list:
            return await original_get_response(self, messages, **kwargs)

        # パイプラインを作成し、middleware付きで実行します。
        from ._types import ChatOptions

        # chat_optionsを抽出するか、デフォルトを作成します。
        chat_options = kwargs.pop("chat_options", ChatOptions())

        pipeline = ChatMiddlewarePipeline(chat_middleware_list)  # type: ignore[arg-type]
        context = ChatContext(
            chat_client=self,
            messages=self.prepare_messages(messages, chat_options),
            chat_options=chat_options,
            is_streaming=False,
            kwargs=kwargs,
        )

        async def final_handler(ctx: ChatContext) -> Any:
            return await original_get_response(self, list(ctx.messages), chat_options=ctx.chat_options, **ctx.kwargs)

        return await pipeline.execute(
            chat_client=self,
            messages=context.messages,
            chat_options=context.chat_options,
            context=context,
            final_handler=final_handler,
            **kwargs,
        )

    def middleware_enabled_get_streaming_response(
        self: Any,
        messages: Any,
        **kwargs: Any,
    ) -> Any:
        """Middleware対応のget_streaming_responseメソッド。"""

        async def _stream_generator() -> Any:
            # 呼び出しレベルまたはインスタンスレベルでmiddlewareが提供されているか確認します。
            call_middleware = kwargs.pop("middleware", None)
            instance_middleware = getattr(self, "middleware", None)

            # 両方のソースからmiddlewareをマージし、chat middlewareのみをフィルタリングします。
            all_middleware: list[ChatMiddleware | ChatMiddlewareCallable] = _merge_and_filter_chat_middleware(
                instance_middleware, call_middleware
            )

            # middlewareがない場合は元のメソッドを使用します。
            if not all_middleware:
                async for update in original_get_streaming_response(self, messages, **kwargs):
                    yield update
                return

            # パイプラインを作成し、middleware付きで実行します。
            from ._types import ChatOptions

            # chat_optionsを抽出するか、デフォルトを作成します。
            chat_options = kwargs.pop("chat_options", ChatOptions())

            pipeline = ChatMiddlewarePipeline(all_middleware)  # type: ignore[arg-type]
            context = ChatContext(
                chat_client=self,
                messages=self.prepare_messages(messages, chat_options),
                chat_options=chat_options,
                is_streaming=True,
                kwargs=kwargs,
            )

            def final_handler(ctx: ChatContext) -> Any:
                return original_get_streaming_response(
                    self, list(ctx.messages), chat_options=ctx.chat_options, **ctx.kwargs
                )

            async for update in pipeline.execute_stream(
                chat_client=self,
                messages=context.messages,
                chat_options=context.chat_options,
                context=context,
                final_handler=final_handler,
                **kwargs,
            ):
                yield update

        return _stream_generator()

    # メソッドを置き換えます。
    chat_client_class.get_response = update_wrapper(middleware_enabled_get_response, original_get_response)  # type: ignore
    chat_client_class.get_streaming_response = update_wrapper(  # type: ignore
        middleware_enabled_get_streaming_response, original_get_streaming_response
    )

    return chat_client_class


def categorize_middleware(
    *middleware_sources: Any | list[Any] | None,
) -> dict[str, list[Any]]:
    """複数のソースからのmiddlewareをagent、function、chatタイプに分類します。

    Args:
        *middleware_sources: 分類する可変数のmiddlewareソース。

    Returns:
        "agent"、"function"、"chat"キーを持つ辞書で、分類されたmiddlewareのリストを含みます。

    """
    result: dict[str, list[Any]] = {"agent": [], "function": [], "chat": []}

    # すべてのmiddlewareソースを単一のリストにマージします。
    all_middleware: list[Any] = []
    for source in middleware_sources:
        if source:
            if isinstance(source, list):
                all_middleware.extend(source)  # type: ignore
            else:
                all_middleware.append(source)

    # 各middlewareアイテムを分類します。
    for middleware in all_middleware:
        if isinstance(middleware, AgentMiddleware):
            result["agent"].append(middleware)
        elif isinstance(middleware, FunctionMiddleware):
            result["function"].append(middleware)
        elif isinstance(middleware, ChatMiddleware):
            result["chat"].append(middleware)
        elif callable(middleware):
            # 常に_determine_middleware_typeを呼び出して適切な検証を保証します。
            middleware_type = _determine_middleware_type(middleware)
            if middleware_type == MiddlewareType.AGENT:
                result["agent"].append(middleware)
            elif middleware_type == MiddlewareType.FUNCTION:
                result["function"].append(middleware)
            elif middleware_type == MiddlewareType.CHAT:
                result["chat"].append(middleware)
        else:
            # 不明なタイプの場合はagent middlewareにフォールバックします。
            result["agent"].append(middleware)

    return result


def create_function_middleware_pipeline(
    *middleware_sources: list[Middleware] | None,
) -> FunctionMiddlewarePipeline | None:
    """複数のmiddlewareソースからfunction middlewareパイプラインを作成します。

    Args:
        *middleware_sources: 可変数のmiddlewareソース。

    Returns:
        function middlewareが見つかった場合はFunctionMiddlewarePipeline、そうでなければNone。

    """
    middleware = categorize_middleware(*middleware_sources)
    function_middlewares = middleware["function"]
    return FunctionMiddlewarePipeline(function_middlewares) if function_middlewares else None  # type: ignore[arg-type]


def _merge_and_filter_chat_middleware(
    instance_middleware: Any | list[Any] | None,
    call_middleware: Any | list[Any] | None,
) -> list[ChatMiddleware | ChatMiddlewareCallable]:
    """インスタンスレベルと呼び出しレベルのmiddlewareをマージし、chat middlewareのみをフィルタリングします。

    Args:
        instance_middleware: インスタンスレベルで定義されたmiddleware。
        call_middleware: 呼び出しレベルで提供されたmiddleware。

    Returns:
        chat middlewareのみを含むマージ済みリスト。

    """
    middleware = categorize_middleware(instance_middleware, call_middleware)
    return middleware["chat"]  # type: ignore[return-value]


def extract_and_merge_function_middleware(chat_client: Any, **kwargs: Any) -> None:
    """chat clientからfunction middlewareを抽出し、kwargs内の既存パイプラインとマージします。

    Args:
        chat_client: middlewareを抽出するchat clientインスタンス。

    Keyword Args:
        **kwargs: middlewareおよびパイプライン情報を含む辞書。

    """
    # middlewareソースを取得します。
    client_middleware = getattr(chat_client, "middleware", None) if hasattr(chat_client, "middleware") else None
    run_level_middleware = kwargs.get("middleware")
    existing_pipeline = kwargs.get("_function_middleware_pipeline")

    # 存在する場合は既存のパイプラインmiddlewareを抽出します。
    existing_middlewares = existing_pipeline._middlewares if existing_pipeline else None

    # 既存のヘルパーを使ってすべてのソースから結合パイプラインを作成します。
    combined_pipeline = create_function_middleware_pipeline(
        client_middleware, run_level_middleware, existing_middlewares
    )

    if combined_pipeline:
        kwargs["_function_middleware_pipeline"] = combined_pipeline
