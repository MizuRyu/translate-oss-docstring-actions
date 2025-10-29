# Copyright (c) Microsoft. All rights reserved.

import asyncio
import sys
from abc import ABC, abstractmethod
from collections.abc import MutableSequence, Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Final, cast

from ._tools import ToolProtocol
from ._types import ChatMessage

if sys.version_info >= (3, 12):
    from typing import override  # type: ignore # pragma: no cover
else:
    from typing_extensions import override  # type: ignore[import] # pragma: no cover
if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover

# region Context

__all__ = ["AggregateContextProvider", "Context", "ContextProvider"]


class Context:
    """ContextProviderによって提供される任意のコンテキストをAIモデルに提供するためのクラスです。

    各ContextProviderは、呼び出しごとに独自のコンテキストを提供する能力を持っています。
    ContextクラスはContextProviderによって提供される追加のコンテキストを含みます。
    このコンテキストは、他のプロバイダーによって提供されるコンテキストと組み合わされてからAIモデルに渡されます。
    このコンテキストは呼び出しごとのものであり、チャット履歴の一部として保存されることはありません。

    Examples:
        .. code-block:: python

            from agent_framework import Context, ChatMessage

            # 指示を含むコンテキストを作成
            context = Context(
                instructions="Use a professional tone when responding.",
                messages=[ChatMessage(content="Previous context", role="user")],
                tools=[my_tool],
            )

            # コンテキストのプロパティにアクセス
            print(context.instructions)
            print(len(context.messages))

    """

    def __init__(
        self,
        instructions: str | None = None,
        messages: Sequence[ChatMessage] | None = None,
        tools: Sequence[ToolProtocol] | None = None,
    ):
        """新しいContextオブジェクトを作成します。

        Args:
            instructions: AIモデルに提供する指示。
            messages: コンテキストに含めるメッセージのリスト。
            tools: この実行に提供するツールのリスト。

        """
        self.instructions = instructions
        self.messages: Sequence[ChatMessage] = messages or []
        self.tools: Sequence[ToolProtocol] = tools or []


# region ContextProvider


class ContextProvider(ABC):
    """すべてのContextProviderの基底クラスです。

    ContextProviderは、AIのコンテキスト管理を強化するために使用できるコンポーネントです。
    会話の変化を監視し、呼び出し直前にAIモデルに追加のコンテキストを提供できます。

    Note:
        ContextProviderは抽象基底クラスです。カスタムContextProviderを作成するには、
        サブクラス化して``invoking()``メソッドを実装する必要があります。
        理想的には、会話状態を追跡するために``invoked()``および``thread_created()``メソッドも実装すべきですが、これらは任意です。

    Examples:
        .. code-block:: python

            from agent_framework import ContextProvider, Context, ChatMessage


            class CustomContextProvider(ContextProvider):
                async def invoking(self, messages, **kwargs):
                    # 各呼び出し前にカスタム指示を追加
                    return Context(instructions="Always be concise and helpful.", messages=[], tools=[])


            # チャットAgentで使用
            async with CustomContextProvider() as provider:
                agent = ChatAgent(chat_client=client, name="assistant", context_providers=provider)

    """

    # メモリや指示を組み立てる際にすべてのContextProviderが使用するデフォルトのprompt
    DEFAULT_CONTEXT_PROMPT: Final[str] = "## Memories\nConsider the following memories when answering user questions:"

    async def thread_created(self, thread_id: str | None) -> None:
        """新しいスレッドが作成された直後に呼び出されます。

        実装者はこのメソッドを使用して、新しいスレッド作成時に必要な操作を行うことができます。
        例えば、現在のセッションに関連するデータを長期ストレージで確認するなどです。

        Args:
            thread_id: 新しいスレッドのID。

        """
        pass

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Agentが基盤となる推論サービスからレスポンスを受け取った後に呼び出されます。

        リクエストおよびレスポンスのメッセージを検査し、ContextProviderの状態を更新できます。

        Args:
            request_messages: モデル/Agentに送信されたメッセージ。
            response_messages: モデル/Agentから返されたメッセージ。
            invoke_exception: 発生した例外（あれば）。

        Keyword Args:
            kwargs: 追加のキーワード引数（現在は未使用）。

        """
        pass

    @abstractmethod
    async def invoking(self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any) -> Context:
        """モデル/Agentが呼び出される直前に呼び出されます。

        実装者はこの時点で必要な追加コンテキストを読み込み、
        Agentに渡すべきコンテキストを返すべきです。

        Args:
            messages: Agentが呼び出される最新のメッセージ。

        Keyword Args:
            kwargs: 追加のキーワード引数（現在は未使用）。

        Returns:
            指示、メッセージ、ツールを含むContextオブジェクト。

        """
        pass

    async def __aenter__(self) -> "Self":
        """非同期コンテキストマネージャに入ります。

        ContextProviderが開始される際のセットアップ操作を行うためにこのメソッドをオーバーライドしてください。

        Returns:
            チェーン用のContextProviderインスタンス。

        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """非同期コンテキストマネージャを終了します。

        ContextProviderが終了される際のクリーンアップ操作を行うためにこのメソッドをオーバーライドしてください。

        Args:
            exc_type: 例外が発生した場合の例外タイプ、そうでなければNone。
            exc_val: 例外が発生した場合の例外値、そうでなければNone。
            exc_tb: 例外が発生した場合のトレースバック、そうでなければNone。

        """
        pass


# region AggregateContextProvider


class AggregateContextProvider(ContextProvider):
    """複数のContextProviderを含むContextProviderです。

    複数のContextProviderにイベントを委譲し、それらのイベントからの応答を集約して返します。
    これにより、複数のContextProviderを単一のプロバイダーにまとめることができます。

    Note:
        AggregateContextProviderは、単一のContextProviderまたはContextProviderのシーケンスを
        Agentのコンストラクタに渡すと自動的に作成されます。

    Examples:
        .. code-block:: python

            from agent_framework import AggregateContextProvider, ChatAgent

            # 複数のContextProviderを作成
            provider1 = CustomContextProvider1()
            provider2 = CustomContextProvider2()
            provider3 = CustomContextProvider3()

            # Agentに渡す - AggregateContextProviderが自動的に作成される
            agent = ChatAgent(chat_client=client, name="assistant", context_providers=[provider1, provider2, provider3])

            # AggregateContextProviderが作成されたことを検証
            assert isinstance(agent.context_providers, AggregateContextProvider)

            # Agentに追加のプロバイダーを追加
            provider4 = CustomContextProvider4()
            agent.context_providers.add(provider4)

    """

    def __init__(self, context_providers: ContextProvider | Sequence[ContextProvider] | None = None) -> None:
        """AggregateContextProviderをContextProviderで初期化します。

        Args:
            context_providers: 追加するContextProviderまたは複数のContextProvider。

        """
        if isinstance(context_providers, ContextProvider):
            self.providers = [context_providers]
        else:
            self.providers = cast(list[ContextProvider], context_providers) or []
        self._exit_stack: AsyncExitStack | None = None

    def add(self, context_provider: ContextProvider) -> None:
        """新しいContextProviderを追加します。

        Args:
            context_provider: 追加するContextProvider。

        """
        self.providers.append(context_provider)

    @override
    async def thread_created(self, thread_id: str | None = None) -> None:
        await asyncio.gather(*[x.thread_created(thread_id) for x in self.providers])

    @override
    async def invoking(self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any) -> Context:
        contexts = await asyncio.gather(*[provider.invoking(messages, **kwargs) for provider in self.providers])
        instructions: str = ""
        return_messages: list[ChatMessage] = []
        tools: list[ToolProtocol] = []
        for ctx in contexts:
            if ctx.instructions:
                instructions += ctx.instructions
            if ctx.messages:
                return_messages.extend(ctx.messages)
            if ctx.tools:
                tools.extend(ctx.tools)
        return Context(instructions=instructions, messages=return_messages, tools=tools)

    @override
    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        await asyncio.gather(*[
            x.invoked(
                request_messages=request_messages,
                response_messages=response_messages,
                invoke_exception=invoke_exception,
                **kwargs,
            )
            for x in self.providers
        ])

    @override
    async def __aenter__(self) -> "Self":
        """非同期コンテキストマネージャに入り、すべてのプロバイダーをセットアップします。

        Returns:
            チェーン用のAggregateContextProviderインスタンス。

        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # すべてのContextProviderに入ります
        for provider in self.providers:
            await self._exit_stack.enter_async_context(provider)

        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """非同期コンテキストマネージャを終了し、すべてのプロバイダーをクリーンアップします。

        Args:
            exc_type: 例外が発生した場合の例外タイプ、そうでなければNone。
            exc_val: 例外が発生した場合の例外値、そうでなければNone。
            exc_tb: 例外が発生した場合のトレースバック、そうでなければNone。

        """
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
