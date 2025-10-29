# Copyright (c) Microsoft. All rights reserved.

from collections.abc import Awaitable, Callable
from typing import Any

from agent_framework import (
    ChatAgent,
    ChatContext,
    ChatMessage,
    ChatMiddleware,
    ChatResponse,
    FunctionCallContent,
    FunctionInvocationContext,
    Role,
    chat_middleware,
    function_middleware,
    use_chat_middleware,
    use_function_invocation,
)

from .conftest import MockBaseChatClient


class TestChatMiddleware:
    """チャットミドルウェア機能のテストケース。"""

    async def test_class_based_chat_middleware(self, chat_client_base: "MockBaseChatClient") -> None:
        """ChatClientを使ったクラスベースのチャットミドルウェアのテスト。"""
        execution_order: list[str] = []

        class LoggingChatMiddleware(ChatMiddleware):
            async def process(
                self,
                context: ChatContext,
                next: Callable[[ChatContext], Awaitable[None]],
            ) -> None:
                execution_order.append("chat_middleware_before")
                await next(context)
                execution_order.append("chat_middleware_after")

        # チャットクライアントにミドルウェアを追加する
        chat_client_base.middleware = [LoggingChatMiddleware()]

        # チャットクライアントを直接実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await chat_client_base.get_response(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT

        # ミドルウェアの実行順序を検証する
        assert execution_order == ["chat_middleware_before", "chat_middleware_after"]

    async def test_function_based_chat_middleware(self, chat_client_base: "MockBaseChatClient") -> None:
        """ChatClientを使った関数ベースのチャットミドルウェアのテスト。"""
        execution_order: list[str] = []

        @chat_middleware
        async def logging_chat_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("function_middleware_before")
            await next(context)
            execution_order.append("function_middleware_after")

        # チャットクライアントにミドルウェアを追加する
        chat_client_base.middleware = [logging_chat_middleware]

        # チャットクライアントを直接実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await chat_client_base.get_response(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT

        # ミドルウェアの実行順序を検証する
        assert execution_order == ["function_middleware_before", "function_middleware_after"]

    async def test_chat_middleware_can_modify_messages(self, chat_client_base: "MockBaseChatClient") -> None:
        """チャットミドルウェアがモデルに送信する前にメッセージを変更できることをテストする。"""

        @chat_middleware
        async def message_modifier_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            # 最初のメッセージにプレフィックスを追加して変更する
            if context.messages and len(context.messages) > 0:
                original_text = context.messages[0].text or ""
                context.messages[0] = ChatMessage(role=context.messages[0].role, text=f"MODIFIED: {original_text}")
            await next(context)

        # チャットクライアントにミドルウェアを追加する
        chat_client_base.middleware = [message_modifier_middleware]

        # チャットクライアントを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await chat_client_base.get_response(messages)

        # メッセージが変更されたことを検証する（MockChatClientは入力をそのまま返す）
        assert response is not None
        assert len(response.messages) > 0
        # モッククライアントは変更されたメッセージを受け取るはずである
        assert "MODIFIED: test message" in response.messages[0].text

    async def test_chat_middleware_can_override_response(self, chat_client_base: "MockBaseChatClient") -> None:
        """チャットミドルウェアがレスポンスを上書きできることをテストする。"""

        @chat_middleware
        async def response_override_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            # next()を呼ばずにレスポンスを上書きする
            context.result = ChatResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text="Middleware overridden response")],
                response_id="middleware-response-123",
            )
            context.terminate = True

        # チャットクライアントにミドルウェアを追加する
        chat_client_base.middleware = [response_override_middleware]

        # チャットクライアントを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await chat_client_base.get_response(messages)

        # レスポンスが上書きされたことを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].text == "Middleware overridden response"
        assert response.response_id == "middleware-response-123"

    async def test_multiple_chat_middleware_execution_order(self, chat_client_base: "MockBaseChatClient") -> None:
        """複数のチャットミドルウェアが正しい順序で実行されることをテストする。"""
        execution_order: list[str] = []

        @chat_middleware
        async def first_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("first_before")
            await next(context)
            execution_order.append("first_after")

        @chat_middleware
        async def second_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("second_before")
            await next(context)
            execution_order.append("second_after")

        # チャットクライアントにミドルウェアを追加する（順序は保持されるべき）
        chat_client_base.middleware = [first_middleware, second_middleware]

        # チャットクライアントを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await chat_client_base.get_response(messages)

        # レスポンスを検証する
        assert response is not None

        # ミドルウェアの実行順序を検証する（ネストされた実行）
        expected_order = ["first_before", "second_before", "second_after", "first_after"]
        assert execution_order == expected_order

    async def test_chat_agent_with_chat_middleware(self) -> None:
        """エージェントレベルでチャットミドルウェアを指定したChatAgentのテスト。"""
        execution_order: list[str] = []

        @chat_middleware
        async def agent_level_chat_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            execution_order.append("agent_chat_middleware_before")
            await next(context)
            execution_order.append("agent_chat_middleware_after")

        chat_client = MockBaseChatClient()

        # チャットミドルウェア付きのChatAgentを作成する
        agent = ChatAgent(chat_client=chat_client, middleware=[agent_level_chat_middleware])

        # エージェントを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT

        # ミドルウェアの実行順序を検証する
        assert execution_order == ["agent_chat_middleware_before", "agent_chat_middleware_after"]

    async def test_chat_agent_with_multiple_chat_middleware(self, chat_client_base: "MockBaseChatClient") -> None:
        """ChatAgentが複数のチャットミドルウェアを持てることをテストする。"""
        execution_order: list[str] = []

        @chat_middleware
        async def first_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("first_before")
            await next(context)
            execution_order.append("first_after")

        @chat_middleware
        async def second_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("second_before")
            await next(context)
            execution_order.append("second_after")

        # 複数のチャットミドルウェア付きのChatAgentを作成する
        agent = ChatAgent(chat_client=chat_client_base, middleware=[first_middleware, second_middleware])

        # エージェントを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None

        # 両方のミドルウェアが実行されたことを検証する（ネストされた実行順序）
        expected_order = ["first_before", "second_before", "second_after", "first_after"]
        assert execution_order == expected_order

    async def test_chat_middleware_with_streaming(self, chat_client_base: "MockBaseChatClient") -> None:
        """ストリーミングレスポンスを伴うチャットミドルウェアのテスト。"""
        execution_order: list[str] = []

        @chat_middleware
        async def streaming_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("streaming_before")
            # ストリーミングコンテキストであることを検証する
            assert context.is_streaming is True
            await next(context)
            execution_order.append("streaming_after")

        # チャットクライアントにミドルウェアを追加する
        chat_client_base.middleware = [streaming_middleware]

        # ストリーミングレスポンスを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        updates: list[object] = []
        async for update in chat_client_base.get_streaming_response(messages):
            updates.append(update)

        # 更新を受け取ったことを検証する
        assert len(updates) > 0

        # ミドルウェアが実行されたことを検証する
        assert execution_order == ["streaming_before", "streaming_after"]

    async def test_run_level_middleware_isolation(self, chat_client_base: "MockBaseChatClient") -> None:
        """ランレベルミドルウェアが呼び出し間で持続せず分離されていることをテストする。"""
        execution_count = {"count": 0}

        @chat_middleware
        async def counting_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_count["count"] += 1
            await next(context)

        # ランレベルミドルウェアを使った最初の呼び出し
        messages = [ChatMessage(role=Role.USER, text="first message")]
        response1 = await chat_client_base.get_response(messages, middleware=[counting_middleware])
        assert response1 is not None
        assert execution_count["count"] == 1

        # ランレベルミドルウェアなしの2回目の呼び出し - ミドルウェアは実行されないはず
        messages = [ChatMessage(role=Role.USER, text="second message")]
        response2 = await chat_client_base.get_response(messages)
        assert response2 is not None
        assert execution_count["count"] == 1  # まだ1のままで、2にはならないはず

        # 再度ランレベルミドルウェアを使った3回目の呼び出し - 実行されるはず
        messages = [ChatMessage(role=Role.USER, text="third message")]
        response3 = await chat_client_base.get_response(messages, middleware=[counting_middleware])
        assert response3 is not None
        assert execution_count["count"] == 2  # 今度は2であるはず

    async def test_chat_client_middleware_can_access_and_override_custom_kwargs(
        self, chat_client_base: "MockBaseChatClient"
    ) -> None:
        """チャットクライアントのミドルウェアがtemperatureなどのカスタムパラメータにアクセスし上書きできることをテストする。"""
        captured_kwargs: dict[str, Any] = {}
        modified_kwargs: dict[str, Any] = {}

        @chat_middleware
        async def kwargs_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            # 元のkwargsをキャプチャする
            captured_kwargs.update(context.kwargs)

            # いくつかのkwargsを変更する
            context.kwargs["temperature"] = 0.9
            context.kwargs["max_tokens"] = 500
            context.kwargs["new_param"] = "added_by_middleware"

            # 検証のために変更されたkwargsを保存する
            modified_kwargs.update(context.kwargs)

            await next(context)

        # チャットクライアントにミドルウェアを追加する
        chat_client_base.middleware = [kwargs_middleware]

        # カスタムパラメータ付きでチャットクライアントを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await chat_client_base.get_response(
            messages, temperature=0.7, max_tokens=100, custom_param="test_value"
        )

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0

        assert captured_kwargs["temperature"] == 0.7
        assert captured_kwargs["max_tokens"] == 100
        assert captured_kwargs["custom_param"] == "test_value"

        # ミドルウェアがkwargsを変更できることを検証する
        assert modified_kwargs["temperature"] == 0.9
        assert modified_kwargs["max_tokens"] == 500
        assert modified_kwargs["new_param"] == "added_by_middleware"
        assert modified_kwargs["custom_param"] == "test_value"  # まだ存在しているはず

    async def test_function_middleware_registration_on_chat_client(self) -> None:
        """ChatClientに登録された関数ミドルウェアが関数呼び出し時に実行されることをテストする。"""
        execution_order: list[str] = []

        @function_middleware
        async def test_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            nonlocal execution_order
            execution_order.append(f"function_middleware_before_{context.function.name}")
            await next(context)
            execution_order.append(f"function_middleware_after_{context.function.name}")

        # シンプルなツール関数を定義する
        def sample_tool(location: str) -> str:
            """ある場所の天気を取得する。"""
            return f"Weather in {location}: sunny"

        # 関数呼び出し対応のチャットクライアントを作成する
        chat_client = use_chat_middleware(use_function_invocation(MockBaseChatClient))()

        # チャットクライアントに直接関数ミドルウェアを設定する
        chat_client.middleware = [test_function_middleware]

        # 関数呼び出しをトリガーするレスポンスを準備する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="call_1",
                            name="sample_tool",
                            arguments={"location": "San Francisco"},
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text="Based on the weather data, it's sunny!")]
        )

        chat_client.run_responses = [function_call_response, final_response]

        # ツールを使ってチャットクライアントを直接実行する - これにより関数呼び出しとミドルウェアがトリガーされるはず
        messages = [ChatMessage(role=Role.USER, text="What's the weather in San Francisco?")]
        response = await chat_client.get_response(messages, tools=[sample_tool])

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # 2回の呼び出し：関数呼び出し＋最終レスポンス

        # 関数ミドルウェアが実行されたことを検証する
        assert execution_order == [
            "function_middleware_before_sample_tool",
            "function_middleware_after_sample_tool",
        ]

    async def test_run_level_function_middleware(self) -> None:
        """get_responseメソッドに渡された関数ミドルウェアも呼び出されることをテストする。"""
        execution_order: list[str] = []

        @function_middleware
        async def run_level_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("run_level_function_middleware_before")
            await next(context)
            execution_order.append("run_level_function_middleware_after")

        # シンプルなツール関数を定義する
        def sample_tool(location: str) -> str:
            """ある場所の天気を取得する。"""
            return f"Weather in {location}: sunny"

        # 関数呼び出し対応のチャットクライアントを作成する
        chat_client = use_function_invocation(MockBaseChatClient)()

        # 関数呼び出しをトリガーするレスポンスを準備する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="call_2",
                            name="sample_tool",
                            arguments={"location": "New York"},
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, text="The weather information has been retrieved!")]
        )

        chat_client.run_responses = [function_call_response, final_response]

        # ランレベルミドルウェアとツールを使ってチャットクライアントを直接実行する
        messages = [ChatMessage(role=Role.USER, text="What's the weather in New York?")]
        response = await chat_client.get_response(
            messages, tools=[sample_tool], middleware=[run_level_function_middleware]
        )

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # 2回の呼び出し：関数呼び出し＋最終レスポンス

        # ランレベル関数ミドルウェアが1回だけ実行されたことを検証する（関数呼び出し時）
        assert execution_order == [
            "run_level_function_middleware_before",
            "run_level_function_middleware_after",
        ]
