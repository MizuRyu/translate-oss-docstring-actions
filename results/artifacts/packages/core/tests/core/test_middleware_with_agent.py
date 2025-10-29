# Copyright (c) Microsoft. All rights reserved.

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from agent_framework import (
    AgentRunResponseUpdate,
    ChatAgent,
    ChatContext,
    ChatMessage,
    ChatMiddleware,
    ChatResponse,
    ChatResponseUpdate,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    agent_middleware,
    chat_middleware,
    function_middleware,
    use_function_invocation,
)
from agent_framework._middleware import (
    AgentMiddleware,
    AgentRunContext,
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareType,
)
from agent_framework.exceptions import MiddlewareException

from .conftest import MockBaseChatClient, MockChatClient

# region ChatAgent Tests


class TestChatAgentClassBasedMiddleware:
    """ChatAgentとのクラスベースmiddleware統合のテストケース。"""

    async def test_class_based_agent_middleware_with_chat_agent(self, chat_client: "MockChatClient") -> None:
        """ChatAgentでのクラスベースagent middlewareをテストします。"""
        execution_order: list[str] = []

        class TrackingAgentMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        # middlewareを使ってChatAgentを作成します
        middleware = TrackingAgentMiddleware("agent_middleware")
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])

        # agentを実行します
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証します
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT
        # 注意: conftestの"MockChatClient"は異なるテキスト形式を返します
        assert "test response" in response.messages[0].text

        # middlewareの実行順序を検証します
        assert execution_order == ["agent_middleware_before", "agent_middleware_after"]

    async def test_class_based_function_middleware_with_chat_agent(self, chat_client: "MockChatClient") -> None:
        """ChatAgentでのクラスベースfunction middlewareをテストします。"""
        execution_order: list[str] = []

        class TrackingFunctionMiddleware(FunctionMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        # function middlewareを使ってChatAgentを作成します（ツールがないためfunction middlewareはトリガーされません）
        middleware = TrackingFunctionMiddleware("function_middleware")
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])

        # agentを実行します
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証します
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 1

        # 注意: function middlewareはfunction呼び出しがないため実行されません
        assert execution_order == []


class TestChatAgentFunctionBasedMiddleware:
    """ChatAgentでのfunctionベースmiddleware統合のテストケース。"""

    async def test_agent_middleware_with_pre_termination(self, chat_client: "MockChatClient") -> None:
        """agent middlewareがnext()を呼び出す前に実行を終了できることをテストします。"""
        execution_order: list[str] = []

        class PreTerminationMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("middleware_before")
                context.terminate = True
                # next()を呼び出しますが、terminate=Trueのため後続のmiddlewareとハンドラーは実行されません
                await next(context)
                execution_order.append("middleware_after")

        # 終了するmiddlewareを使ってChatAgentを作成します
        middleware = PreTerminationMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])

        # 複数メッセージでagentを実行します
        messages = [
            ChatMessage(role=Role.USER, text="message1"),
            ChatMessage(role=Role.USER, text="message2"),  # This should not be processed due to termination
        ]
        response = await agent.run(messages)

        # レスポンスを検証します
        assert response is not None
        assert not response.messages  # 事前終了のためレスポンスにメッセージは含まれないはずです
        assert execution_order == ["middleware_before", "middleware_after"]  # middlewareはそれでも完了します
        assert chat_client.call_count == 0  # 終了のため呼び出しは行われません

    async def test_agent_middleware_with_post_termination(self, chat_client: "MockChatClient") -> None:
        """agent middlewareがnext()を呼び出した後に実行を終了できることをテストします。"""
        execution_order: list[str] = []

        class PostTerminationMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("middleware_before")
                await next(context)
                execution_order.append("middleware_after")
                context.terminate = True

        # 終了するmiddlewareを使ってChatAgentを作成します
        middleware = PostTerminationMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])

        # 複数メッセージでagentを実行します
        messages = [
            ChatMessage(role=Role.USER, text="message1"),
            ChatMessage(role=Role.USER, text="message2"),
        ]
        response = await agent.run(messages)

        # レスポンスを検証します
        assert response is not None
        assert len(response.messages) == 1
        assert response.messages[0].role == Role.ASSISTANT
        assert "test response" in response.messages[0].text

        # middlewareの実行順序を検証します
        assert execution_order == ["middleware_before", "middleware_after"]
        assert chat_client.call_count == 1

    async def test_function_middleware_with_pre_termination(self, chat_client: "MockChatClient") -> None:
        """function middlewareがnext()を呼び出す前に実行を終了できることをテストします。"""
        execution_order: list[str] = []

        class PreTerminationFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("middleware_before")
                context.terminate = True
                # next()を呼び出しますが、terminate=Trueのため後続のmiddlewareとハンドラーは実行されません
                await next(context)
                execution_order.append("middleware_after")

        # 会話を開始するためのメッセージを作成します
        messages = [ChatMessage(role=Role.USER, text="test message")]

        # function callを返すようにchat clientを設定します
        chat_client.responses = [
            ChatResponse(
                messages=[
                    ChatMessage(
                        role=Role.ASSISTANT,
                        contents=[
                            FunctionCallContent(call_id="test_call", name="test_function", arguments={"text": "test"})
                        ],
                    )
                ]
            )
        ]

        # 期待されるシグネチャでテスト用functionを作成します
        def test_function(text: str) -> str:
            execution_order.append("function_called")
            return "test_result"

        # function middlewareとテスト用functionを使ってChatAgentを作成します
        middleware = PreTerminationFunctionMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware], tools=[test_function])

        # agentを実行します
        await agent.run(messages)

        # functionが呼び出されずmiddlewareのみが実行されたことを検証します
        assert execution_order == ["middleware_before", "middleware_after"]
        assert "function_called" not in execution_order
        assert execution_order == ["middleware_before", "middleware_after"]

    async def test_function_middleware_with_post_termination(self, chat_client: "MockChatClient") -> None:
        """function middlewareがnext()を呼び出した後に実行を終了できることをテストします。"""
        execution_order: list[str] = []

        class PostTerminationFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("middleware_before")
                await next(context)
                execution_order.append("middleware_after")
                context.terminate = True

        # 会話を開始するためのメッセージを作成する
        messages = [ChatMessage(role=Role.USER, text="test message")]

        # 関数呼び出しを返すようにchat clientを設定する
        chat_client.responses = [
            ChatResponse(
                messages=[
                    ChatMessage(
                        role=Role.ASSISTANT,
                        contents=[
                            FunctionCallContent(call_id="test_call", name="test_function", arguments={"text": "test"})
                        ],
                    )
                ]
            )
        ]

        # 期待されるシグネチャでテスト関数を作成する
        def test_function(text: str) -> str:
            execution_order.append("function_called")
            return "test_result"

        # function middlewareとテスト関数を使ってChatAgentを作成する
        middleware = PostTerminationFunctionMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware], tools=[test_function])

        # agentを実行する
        response = await agent.run(messages)

        # 関数が呼び出されmiddlewareが実行されたことを検証する
        assert response is not None
        assert "function_called" in execution_order
        assert execution_order == ["middleware_before", "function_called", "middleware_after"]

    async def test_function_based_agent_middleware_with_chat_agent(self, chat_client: "MockChatClient") -> None:
        """ChatAgentで関数ベースのagent middlewareをテストする。"""
        execution_order: list[str] = []

        async def tracking_agent_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            execution_order.append("agent_function_before")
            await next(context)
            execution_order.append("agent_function_after")

        # function middlewareを使ってChatAgentを作成する
        agent = ChatAgent(chat_client=chat_client, middleware=[tracking_agent_middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT
        assert response.messages[0].text == "test response"
        assert chat_client.call_count == 1

        # middlewareの実行順序を検証する
        assert execution_order == ["agent_function_before", "agent_function_after"]

    async def test_function_based_function_middleware_with_chat_agent(self, chat_client: "MockChatClient") -> None:
        """ChatAgentで関数ベースのfunction middlewareをテストする。"""
        execution_order: list[str] = []

        async def tracking_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_function_before")
            await next(context)
            execution_order.append("function_function_after")

        # function middlewareを使ってChatAgentを作成する（ツールなしのためfunction middlewareはトリガーされない）
        agent = ChatAgent(chat_client=chat_client, middleware=[tracking_function_middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 1

        # 注意: 関数呼び出しがないためfunction middlewareは実行されない
        assert execution_order == []


class TestChatAgentStreamingMiddleware:
    """ChatAgentでのストリーミングmiddleware統合のテストケース。"""

    async def test_agent_middleware_with_streaming(self, chat_client: "MockChatClient") -> None:
        """ストリーミングChatAgentレスポンスでagent middlewareをテストする。"""
        execution_order: list[str] = []
        streaming_flags: list[bool] = []

        class StreamingTrackingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("middleware_before")
                streaming_flags.append(context.is_streaming)
                await next(context)
                execution_order.append("middleware_after")

        # middlewareを使ってChatAgentを作成する
        middleware = StreamingTrackingMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])

        # モックのストリーミングレスポンスを設定する
        chat_client.streaming_responses = [
            [
                ChatResponseUpdate(contents=[TextContent(text="Streaming")], role=Role.ASSISTANT),
                ChatResponseUpdate(contents=[TextContent(text=" response")], role=Role.ASSISTANT),
            ]
        ]

        # ストリーミングを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream(messages):
            updates.append(update)

        # ストリーミングレスポンスを検証する
        assert len(updates) == 2
        assert updates[0].text == "Streaming"
        assert updates[1].text == " response"
        assert chat_client.call_count == 1

        # middlewareが呼び出されストリーミングフラグが正しく設定されたことを検証する
        assert execution_order == ["middleware_before", "middleware_after"]
        assert streaming_flags == [True]  # Contextはストリーミングを示すべきである

    async def test_non_streaming_vs_streaming_flag_validation(self, chat_client: "MockChatClient") -> None:
        """異なる実行モードでis_streamingフラグが正しく設定されていることをテストする。"""
        streaming_flags: list[bool] = []

        class FlagTrackingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                streaming_flags.append(context.is_streaming)
                await next(context)

        # middlewareを使ってChatAgentを作成する
        middleware = FlagTrackingMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])
        messages = [ChatMessage(role=Role.USER, text="test message")]

        # 非ストリーミング実行をテストする
        response = await agent.run(messages)
        assert response is not None

        # ストリーミング実行をテストする
        async for _ in agent.run_stream(messages):
            pass

        # フラグを検証する: [非ストリーミング, ストリーミング]
        assert streaming_flags == [False, True]


class TestChatAgentMultipleMiddlewareOrdering:
    """ChatAgentでの複数middleware実行順序のテストケース。"""

    async def test_multiple_agent_middleware_execution_order(self, chat_client: "MockChatClient") -> None:
        """複数のagent middlewareがChatAgentで正しい順序で実行されることをテストする。"""
        execution_order: list[str] = []

        class OrderedMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        # 複数のmiddlewareを作成する
        middleware1 = OrderedMiddleware("first")
        middleware2 = OrderedMiddleware("second")
        middleware3 = OrderedMiddleware("third")

        # 複数のmiddlewareを使ってChatAgentを作成する
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware1, middleware2, middleware3])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert chat_client.call_count == 1

        # 実行順序を検証する（ネストされるべき：最初が二番目をラップし、二番目が三番目をラップする）
        expected_order = ["first_before", "second_before", "third_before", "third_after", "second_after", "first_after"]
        assert execution_order == expected_order

    async def test_mixed_middleware_types_with_chat_agent(self, chat_client: "MockChatClient") -> None:
        """ChatAgentでクラスベースと関数ベースのmiddleware混合をテストする。"""
        execution_order: list[str] = []

        class ClassAgentMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("class_agent_before")
                await next(context)
                execution_order.append("class_agent_after")

        async def function_agent_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_agent_before")
            await next(context)
            execution_order.append("function_agent_after")

        class ClassFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("class_function_before")
                await next(context)
                execution_order.append("class_function_after")

        async def function_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_function_before")
            await next(context)
            execution_order.append("function_function_after")

        # 混合middlewareタイプでChatAgentを作成する（ツールなし、agent middlewareに注目）
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[
                ClassAgentMiddleware(),
                function_agent_middleware,
                ClassFunctionMiddleware(),  # Won't execute without function calls
                function_function_middleware,  # Won't execute without function calls
            ],
        )

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert chat_client.call_count == 1

        # agent middlewareが正しい順序で実行されたことを検証する （関数middlewareは関数呼び出しがないため実行されない）
        expected_order = ["class_agent_before", "function_agent_before", "function_agent_after", "class_agent_after"]
        assert execution_order == expected_order


# region テスト用ツール関数


def sample_tool_function(location: str) -> str:
    """middlewareテスト用のシンプルなツール関数。"""
    return f"Weather in {location}: sunny"


# region ツール使用時のChatAgent関数middlewareテスト


class TestChatAgentFunctionMiddlewareWithTools:
    """ツール使用時のChatAgentでのfunction middleware統合のテストケース。"""

    async def test_class_based_function_middleware_with_tool_calls(self, chat_client: "MockChatClient") -> None:
        """関数呼び出しがある場合のクラスベースfunction middlewareをChatAgentでテストする。"""
        execution_order: list[str] = []

        class TrackingFunctionMiddleware(FunctionMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        # 最初に関数呼び出しを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="call_123",
                            name="sample_tool_function",
                            arguments='{"location": "Seattle"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])

        chat_client.responses = [function_call_response, final_response]

        # function middlewareとツールを使ってChatAgentを作成する
        middleware = TrackingFunctionMiddleware("function_middleware")
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[middleware],
            tools=[sample_tool_function],
        )

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="Get weather for Seattle")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # 2回の呼び出し：1回は関数呼び出し用、もう1回は最終レスポンス用

        # function middlewareが実行されたことを検証する
        assert execution_order == ["function_middleware_before", "function_middleware_after"]

        # レスポンスに関数呼び出しと結果が含まれていることを検証する
        all_contents = [content for message in response.messages for content in message.contents]
        function_calls = [c for c in all_contents if isinstance(c, FunctionCallContent)]
        function_results = [c for c in all_contents if isinstance(c, FunctionResultContent)]

        assert len(function_calls) == 1
        assert len(function_results) == 1
        assert function_calls[0].name == "sample_tool_function"
        assert function_results[0].call_id == function_calls[0].call_id

    async def test_function_based_function_middleware_with_tool_calls(self, chat_client: "MockChatClient") -> None:
        """関数呼び出しがある場合の関数ベースfunction middlewareをChatAgentでテストする。"""
        execution_order: list[str] = []

        async def tracking_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_middleware_before")
            await next(context)
            execution_order.append("function_middleware_after")

        # 最初に関数呼び出しを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="call_456",
                            name="sample_tool_function",
                            arguments='{"location": "San Francisco"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])

        chat_client.responses = [function_call_response, final_response]

        # function middlewareとツールを使ってChatAgentを作成する
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[tracking_function_middleware],
            tools=[sample_tool_function],
        )

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="Get weather for San Francisco")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # 2回の呼び出し：1回は関数呼び出し用、もう1回は最終レスポンス用

        # function middlewareが実行されたことを検証する
        assert execution_order == ["function_middleware_before", "function_middleware_after"]

        # レスポンスに関数呼び出しと結果が含まれていることを検証する
        all_contents = [content for message in response.messages for content in message.contents]
        function_calls = [c for c in all_contents if isinstance(c, FunctionCallContent)]
        function_results = [c for c in all_contents if isinstance(c, FunctionResultContent)]

        assert len(function_calls) == 1
        assert len(function_results) == 1
        assert function_calls[0].name == "sample_tool_function"
        assert function_results[0].call_id == function_calls[0].call_id

    async def test_mixed_agent_and_function_middleware_with_tool_calls(self, chat_client: "MockChatClient") -> None:
        """関数呼び出しがある場合のagentとfunction middleware両方をChatAgentでテストする。"""
        execution_order: list[str] = []

        class TrackingAgentMiddleware(AgentMiddleware):
            async def process(
                self,
                context: AgentRunContext,
                next: Callable[[AgentRunContext], Awaitable[None]],
            ) -> None:
                execution_order.append("agent_middleware_before")
                await next(context)
                execution_order.append("agent_middleware_after")

        class TrackingFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("function_middleware_before")
                await next(context)
                execution_order.append("function_middleware_after")

        # 最初に関数呼び出しを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="call_789",
                            name="sample_tool_function",
                            arguments='{"location": "New York"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])

        chat_client.responses = [function_call_response, final_response]

        # agentとfunction middleware両方とツールを使ってChatAgentを作成する
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[TrackingAgentMiddleware(), TrackingFunctionMiddleware()],
            tools=[sample_tool_function],
        )

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="Get weather for New York")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # 2回の呼び出し：1回は関数呼び出し用、もう1回は最終レスポンス用

        # middlewareの実行順序を検証する：agent middlewareが全体をラップし、 function
        # middlewareは関数呼び出し時のみ実行される
        expected_order = [
            "agent_middleware_before",
            "function_middleware_before",
            "function_middleware_after",
            "agent_middleware_after",
        ]
        assert execution_order == expected_order

        # レスポンスに関数呼び出しと結果が含まれていることを検証する
        all_contents = [content for message in response.messages for content in message.contents]
        function_calls = [c for c in all_contents if isinstance(c, FunctionCallContent)]
        function_results = [c for c in all_contents if isinstance(c, FunctionResultContent)]

        assert len(function_calls) == 1
        assert len(function_results) == 1
        assert function_calls[0].name == "sample_tool_function"
        assert function_results[0].call_id == function_calls[0].call_id

    async def test_function_middleware_can_access_and_override_custom_kwargs(
        self, chat_client: "MockChatClient"
    ) -> None:
        """function middlewareがtemperatureなどのカスタムパラメータにアクセスし上書きできることをテストする。"""
        captured_kwargs: dict[str, Any] = {}
        modified_kwargs: dict[str, Any] = {}
        middleware_called = False

        @function_middleware
        async def kwargs_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            nonlocal middleware_called
            middleware_called = True

            # 元のkwargsをキャプチャする
            captured_kwargs["has_chat_options"] = "chat_options" in context.kwargs
            captured_kwargs["has_custom_param"] = "custom_param" in context.kwargs
            captured_kwargs["custom_param"] = context.kwargs.get("custom_param")

            # chat_optionsの元の値をキャプチャする（存在する場合）
            if "chat_options" in context.kwargs:
                chat_options = context.kwargs["chat_options"]
                captured_kwargs["original_temperature"] = getattr(chat_options, "temperature", None)
                captured_kwargs["original_max_tokens"] = getattr(chat_options, "max_tokens", None)

            # いくつかのkwargsを修正する
            context.kwargs["temperature"] = 0.9
            context.kwargs["max_tokens"] = 500
            context.kwargs["new_param"] = "added_by_middleware"

            # chat_optionsも修正する（存在する場合）
            if "chat_options" in context.kwargs:
                context.kwargs["chat_options"].temperature = 0.9
                context.kwargs["chat_options"].max_tokens = 500

            # 検証用に修正したkwargsを保存する
            modified_kwargs["temperature"] = context.kwargs.get("temperature")
            modified_kwargs["max_tokens"] = context.kwargs.get("max_tokens")
            modified_kwargs["new_param"] = context.kwargs.get("new_param")
            modified_kwargs["custom_param"] = context.kwargs.get("custom_param")

            # 修正したchat_optionsの値をキャプチャする（存在する場合）
            if "chat_options" in context.kwargs:
                chat_options = context.kwargs["chat_options"]
                modified_kwargs["chat_options_temperature"] = getattr(chat_options, "temperature", None)
                modified_kwargs["chat_options_max_tokens"] = getattr(chat_options, "max_tokens", None)

            await next(context)

        chat_client.responses = [
            ChatResponse(
                messages=[
                    ChatMessage(
                        role=Role.ASSISTANT,
                        contents=[
                            FunctionCallContent(
                                call_id="test_call", name="sample_tool_function", arguments={"location": "Seattle"}
                            )
                        ],
                    )
                ]
            ),
            ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent("Function completed")])]),
        ]

        # function middlewareを使ってChatAgentを作成する
        agent = ChatAgent(chat_client=chat_client, middleware=[kwargs_middleware], tools=[sample_tool_function])

        # カスタムパラメータでagentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages, temperature=0.7, max_tokens=100, custom_param="test_value")

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0

        # まずmiddlewareが呼び出されたかどうかを確認する
        assert middleware_called, "Function middleware was not called"

        # middlewareが元のkwargsをキャプチャしたことを検証する
        assert captured_kwargs["has_chat_options"] is True
        assert captured_kwargs["has_custom_param"] is True
        assert captured_kwargs["custom_param"] == "test_value"
        assert captured_kwargs["original_temperature"] == 0.7
        assert captured_kwargs["original_max_tokens"] == 100

        # middlewareがkwargsを修正できたことを検証する
        assert modified_kwargs["temperature"] == 0.9
        assert modified_kwargs["max_tokens"] == 500
        assert modified_kwargs["new_param"] == "added_by_middleware"
        assert modified_kwargs["custom_param"] == "test_value"
        assert modified_kwargs["chat_options_temperature"] == 0.9
        assert modified_kwargs["chat_options_max_tokens"] == 500


class TestMiddlewareDynamicRebuild:
    """ChatAgentでの動的middlewareパイプライン再構築のテストケース。"""

    class TrackingAgentMiddleware(AgentMiddleware):
        """実行を追跡するmiddlewareをテストする。"""

        def __init__(self, name: str, execution_log: list[str]):
            self.name = name
            self.execution_log = execution_log

        async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
            self.execution_log.append(f"{self.name}_start")
            await next(context)
            self.execution_log.append(f"{self.name}_end")

    async def test_middleware_dynamic_rebuild_non_streaming(self, chat_client: "MockChatClient") -> None:
        """agent.middlewareコレクションが変更されたときに非ストリーミングでmiddlewareパイプラインが再構築されることをテストする。"""
        execution_log: list[str] = []

        # 初期middlewareでagentを作成する
        middleware1 = self.TrackingAgentMiddleware("middleware1", execution_log)
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware1])

        # 最初の実行 - middleware1を使用するはず
        await agent.run("Test message 1")
        assert "middleware1_start" in execution_log
        assert "middleware1_end" in execution_log

        # 実行ログをクリアする
        execution_log.clear()

        # middlewareコレクションを変更して別のmiddlewareを追加する
        middleware2 = self.TrackingAgentMiddleware("middleware2", execution_log)
        agent.middleware = [middleware1, middleware2]

        # 2回目の実行 - middleware1とmiddleware2の両方を使用するはず
        await agent.run("Test message 2")
        assert "middleware1_start" in execution_log
        assert "middleware1_end" in execution_log
        assert "middleware2_start" in execution_log
        assert "middleware2_end" in execution_log

        # 実行ログをクリアする
        execution_log.clear()

        # middlewareコレクションを変更してmiddleware2だけに置き換える
        agent.middleware = [middleware2]

        # 3回目の実行 - middleware2のみを使用するはず
        await agent.run("Test message 3")
        assert "middleware1_start" not in execution_log
        assert "middleware1_end" not in execution_log
        assert "middleware2_start" in execution_log
        assert "middleware2_end" in execution_log

        # 実行ログをクリアする
        execution_log.clear()

        # すべてのmiddlewareを削除する
        agent.middleware = []

        # 4回目の実行 - middlewareなしで実行するはず
        await agent.run("Test message 4")
        assert len(execution_log) == 0

    async def test_middleware_dynamic_rebuild_streaming(self, chat_client: "MockChatClient") -> None:
        """agent.middlewareコレクションが変更されたときにストリーミングでmiddlewareパイプラインが再構築されることをテストする。"""
        execution_log: list[str] = []

        # 初期middlewareでagentを作成する
        middleware1 = self.TrackingAgentMiddleware("stream_middleware1", execution_log)
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware1])

        # 最初のストリーミング実行
        updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream("Test stream message 1"):
            updates.append(update)

        assert "stream_middleware1_start" in execution_log
        assert "stream_middleware1_end" in execution_log

        # 実行ログをクリアする
        execution_log.clear()

        # middlewareコレクションを変更する
        middleware2 = self.TrackingAgentMiddleware("stream_middleware2", execution_log)
        agent.middleware = [middleware2]

        # 2回目のストリーミング実行 - middleware2のみを使用するはず
        updates = []
        async for update in agent.run_stream("Test stream message 2"):
            updates.append(update)

        assert "stream_middleware1_start" not in execution_log
        assert "stream_middleware1_end" not in execution_log
        assert "stream_middleware2_start" in execution_log
        assert "stream_middleware2_end" in execution_log

    async def test_middleware_order_change_detection(self, chat_client: "MockChatClient") -> None:
        """middlewareの順序変更が検出され適用されることをテストする。"""
        execution_log: list[str] = []

        middleware1 = self.TrackingAgentMiddleware("first", execution_log)
        middleware2 = self.TrackingAgentMiddleware("second", execution_log)

        # [first, second]の順序でmiddlewareを持つagentを作成する
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware1, middleware2])

        # 最初の実行
        await agent.run("Test message 1")
        assert execution_log == ["first_start", "second_start", "second_end", "first_end"]

        # 実行ログをクリアする
        execution_log.clear()

        # 順序を[second, first]に変更する
        agent.middleware = [middleware2, middleware1]

        # 2回目の実行 - 新しい順序が反映されるはず
        await agent.run("Test message 2")
        assert execution_log == ["second_start", "first_start", "first_end", "second_end"]


class TestRunLevelMiddleware:
    """run-level middleware機能のテストケース。"""

    class TrackingAgentMiddleware(AgentMiddleware):
        """実行を追跡するmiddlewareをテストする。"""

        def __init__(self, name: str, execution_log: list[str]):
            self.name = name
            self.execution_log = execution_log

        async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
            self.execution_log.append(f"{self.name}_start")
            await next(context)
            self.execution_log.append(f"{self.name}_end")

    async def test_run_level_middleware_isolation(self, chat_client: "MockChatClient") -> None:
        """複数のrun間でrun-level middlewareが分離されていることをテストする。"""
        execution_log: list[str] = []

        # agent-level middlewareなしでagentを作成する
        agent = ChatAgent(chat_client=chat_client)

        # run-level middlewareを作成する
        run_middleware1 = self.TrackingAgentMiddleware("run1", execution_log)
        run_middleware2 = self.TrackingAgentMiddleware("run2", execution_log)

        # run_middleware1での最初の実行
        await agent.run("Test message 1", middleware=[run_middleware1])
        assert execution_log == ["run1_start", "run1_end"]

        # 実行ログをクリアする
        execution_log.clear()

        # run_middleware2での2回目の実行 - run_middleware1は見えないはず
        await agent.run("Test message 2", middleware=[run_middleware2])
        assert execution_log == ["run2_start", "run2_end"]
        assert "run1_start" not in execution_log
        assert "run1_end" not in execution_log

        # 実行ログをクリアする
        execution_log.clear()

        # middlewareなしでの3回目の実行 - middleware実行は見えないはず
        await agent.run("Test message 3")
        assert execution_log == []

        # 実行ログをクリアする
        execution_log.clear()

        # 両方のrun middlewareでの4回目の実行 - 両方が見えるはず
        await agent.run("Test message 4", middleware=[run_middleware1, run_middleware2])
        assert execution_log == ["run1_start", "run2_start", "run2_end", "run1_end"]

    async def test_agent_plus_run_middleware_execution_order(self, chat_client: "MockChatClient") -> None:
        """agent middlewareが最初に実行され、その後にrun middlewareが実行されることをテストする。"""
        execution_log: list[str] = []
        metadata_log: list[str] = []

        class MetadataAgentMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_log.append(f"{self.name}_start")
                # run middlewareに情報を渡すためにmetadataを設定する
                context.metadata[f"{self.name}_key"] = f"{self.name}_value"
                await next(context)
                execution_log.append(f"{self.name}_end")

        class MetadataRunMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_log.append(f"{self.name}_start")
                # agent middlewareによって設定されたmetadataを読み取る
                for key, value in context.metadata.items():
                    metadata_log.append(f"{self.name}_reads_{key}:{value}")
                # run-level metadataを設定する
                context.metadata[f"{self.name}_key"] = f"{self.name}_value"
                await next(context)
                execution_log.append(f"{self.name}_end")

        # agent-level middlewareを持つagentを作成する
        agent_middleware = MetadataAgentMiddleware("agent")
        agent = ChatAgent(chat_client=chat_client, middleware=[agent_middleware])

        # run-level middlewareを作成する
        run_middleware = MetadataRunMiddleware("run")

        # agentとrun middlewareの両方で実行する
        await agent.run("Test message", middleware=[run_middleware])

        # 実行順序を検証する：agent middlewareがrun middlewareをラップする
        expected_order = ["agent_start", "run_start", "run_end", "agent_end"]
        assert execution_log == expected_order

        # run middlewareがagent middlewareのメタデータを読み取れることを検証する
        assert "run_reads_agent_key:agent_value" in metadata_log

    async def test_run_level_middleware_non_streaming(self, chat_client: "MockChatClient") -> None:
        """非ストリーミング実行でのrun-level middlewareのテスト。"""
        execution_log: list[str] = []

        # agent-level middlewareなしでagentを作成する
        agent = ChatAgent(chat_client=chat_client)

        # run-level middlewareを作成する
        run_middleware = self.TrackingAgentMiddleware("run_nonstream", execution_log)

        # run middlewareで非ストリーミング実行を行う
        response = await agent.run("Test non-streaming", middleware=[run_middleware])

        # レスポンスが正しいことを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT
        assert "test response" in response.messages[0].text

        # middlewareが実行されたことを検証する
        assert execution_log == ["run_nonstream_start", "run_nonstream_end"]

    async def test_run_level_middleware_streaming(self, chat_client: "MockChatClient") -> None:
        """ストリーミング実行でのrun-level middlewareのテスト。"""
        execution_log: list[str] = []
        streaming_flags: list[bool] = []

        class StreamingTrackingMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_log.append(f"{self.name}_start")
                streaming_flags.append(context.is_streaming)
                await next(context)
                execution_log.append(f"{self.name}_end")

        # agent-level middlewareなしでagentを作成する
        agent = ChatAgent(chat_client=chat_client)

        # モックのストリーミングレスポンスを設定する
        chat_client.streaming_responses = [
            [
                ChatResponseUpdate(contents=[TextContent(text="Stream")], role=Role.ASSISTANT),
                ChatResponseUpdate(contents=[TextContent(text=" response")], role=Role.ASSISTANT),
            ]
        ]

        # run-level middlewareを作成する
        run_middleware = StreamingTrackingMiddleware("run_stream")

        # run middlewareでストリーミング実行を行う
        updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream("Test streaming", middleware=[run_middleware]):
            updates.append(update)

        # ストリーミングレスポンスを検証する
        assert len(updates) == 2
        assert updates[0].text == "Stream"
        assert updates[1].text == " response"

        # 正しいストリーミングフラグでmiddlewareが実行されたことを検証する
        assert execution_log == ["run_stream_start", "run_stream_end"]
        assert streaming_flags == [True]  # Contextはストリーミングを示すべきである

    async def test_agent_and_run_level_both_agent_and_function_middleware(self, chat_client: "MockChatClient") -> None:
        """agent-levelとrun-levelの両方でagentとfunction middlewareを使った完全なシナリオをテストする。"""
        execution_log: list[str] = []

        # agent-level middleware
        class AgentLevelAgentMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_log.append("agent_level_agent_start")
                context.metadata["agent_level_agent"] = "processed"
                await next(context)
                execution_log.append("agent_level_agent_end")

        class AgentLevelFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_log.append("agent_level_function_start")
                context.metadata["agent_level_function"] = "processed"
                await next(context)
                execution_log.append("agent_level_function_end")

        # run-level middleware
        class RunLevelAgentMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_log.append("run_level_agent_start")
                # agent-level middlewareのメタデータが利用可能であることを検証する
                assert "agent_level_agent" in context.metadata
                context.metadata["run_level_agent"] = "processed"
                await next(context)
                execution_log.append("run_level_agent_end")

        class RunLevelFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_log.append("run_level_function_start")
                # agent-level function middlewareのメタデータが利用可能であることを検証する
                assert "agent_level_function" in context.metadata
                context.metadata["run_level_function"] = "processed"
                await next(context)
                execution_log.append("run_level_function_end")

        # function middlewareのテスト用のツール関数を作成する
        def custom_tool(message: str) -> str:
            execution_log.append("tool_executed")
            return f"Tool response: {message}"

        # 最初にfunction callを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="test_call",
                            name="custom_tool",
                            arguments='{"message": "test"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])
        chat_client.responses = [function_call_response, final_response]

        # agent-level middleware付きでagentを作成する
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[AgentLevelAgentMiddleware(), AgentLevelFunctionMiddleware()],
            tools=[custom_tool],
        )

        # run-level middlewareで実行する
        response = await agent.run(
            "Test message",
            middleware=[RunLevelAgentMiddleware(), RunLevelFunctionMiddleware()],
        )

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # Function call + 最終レスポンス

        expected_order = [
            "agent_level_agent_start",
            "run_level_agent_start",
            "agent_level_function_start",
            "run_level_function_start",
            "tool_executed",
            "run_level_function_end",
            "agent_level_function_end",
            "run_level_agent_end",
            "agent_level_agent_end",
        ]
        assert execution_log == expected_order

        # function callと結果がレスポンスに含まれていることを検証する
        all_contents = [content for message in response.messages for content in message.contents]
        function_calls = [c for c in all_contents if isinstance(c, FunctionCallContent)]
        function_results = [c for c in all_contents if isinstance(c, FunctionResultContent)]

        assert len(function_calls) == 1
        assert len(function_results) == 1
        assert function_calls[0].name == "custom_tool"
        assert function_results[0].call_id == function_calls[0].call_id
        assert function_results[0].result is not None
        assert "Tool response: test" in str(function_results[0].result)


class TestMiddlewareDecoratorLogic:
    """middlewareのデコレータと型注釈のロジックをテストする。"""

    async def test_decorator_and_type_match(self, chat_client: MockChatClient) -> None:
        """デコレータとパラメータ型の両方が指定され、一致している。"""

        execution_order: list[str] = []

        @agent_middleware
        async def matching_agent_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            execution_order.append("decorator_type_match_agent")
            await next(context)

        @function_middleware
        async def matching_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("decorator_type_match_function")
            await next(context)

        # function middlewareのテスト用のツール関数を作成する
        def custom_tool(message: str) -> str:
            execution_order.append("tool_executed")
            return f"Tool response: {message}"

        # 最初にfunction callを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="test_call",
                            name="custom_tool",
                            arguments='{"message": "test"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])
        chat_client.responses = [function_call_response, final_response]

        # エラーなく動作するはずである
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[matching_agent_middleware, matching_function_middleware],
            tools=[custom_tool],
        )

        response = await agent.run([ChatMessage(role=Role.USER, text="test")])

        assert response is not None
        assert "decorator_type_match_agent" in execution_order
        assert "decorator_type_match_function" in execution_order

    async def test_decorator_and_type_mismatch(self, chat_client: MockChatClient) -> None:
        """デコレータとパラメータ型の両方が指定されているが一致しない。"""

        # これはデコレーション時に型エラーを引き起こすため、異なる方法でテストする必要がある
        # agent作成時に不一致のためMiddlewareExceptionを発生させるべきである
        with pytest.raises(MiddlewareException, match="Middleware type mismatch"):

            @agent_middleware  # type: ignore[arg-type]
            async def mismatched_middleware(
                context: FunctionInvocationContext,  # Wrong type for @agent_middleware
                next: Any,
            ) -> None:
                await next(context)

            agent = ChatAgent(chat_client=chat_client, middleware=[mismatched_middleware])
            await agent.run([ChatMessage(role=Role.USER, text="test")])

    async def test_only_decorator_specified(self, chat_client: Any) -> None:
        """デコレータのみ指定 - デコレータに依存する。"""
        execution_order: list[str] = []

        @agent_middleware
        async def decorator_only_agent(context: Any, next: Any) -> None:  # No type annotation
            execution_order.append("decorator_only_agent")
            await next(context)

        @function_middleware
        async def decorator_only_function(context: Any, next: Any) -> None:  # No type annotation
            execution_order.append("decorator_only_function")
            await next(context)

        # function middlewareのテスト用のツール関数を作成する
        def custom_tool(message: str) -> str:
            execution_order.append("tool_executed")
            return f"Tool response: {message}"

        # 最初にfunction callを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="test_call",
                            name="custom_tool",
                            arguments='{"message": "test"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])
        chat_client.responses = [function_call_response, final_response]

        # デコレータに依存して動作するはずである
        agent = ChatAgent(
            chat_client=chat_client, middleware=[decorator_only_agent, decorator_only_function], tools=[custom_tool]
        )

        response = await agent.run([ChatMessage(role=Role.USER, text="test")])

        assert response is not None
        assert "decorator_only_agent" in execution_order
        assert "decorator_only_function" in execution_order

    async def test_only_type_specified(self, chat_client: Any) -> None:
        """パラメータ型のみ指定 - 型に依存する。"""
        execution_order: list[str] = []

        # デコレータなし
        async def type_only_agent(context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
            execution_order.append("type_only_agent")
            await next(context)

        # デコレータなし
        async def type_only_function(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("type_only_function")
            await next(context)

        # function middlewareのテスト用のツール関数を作成する
        def custom_tool(message: str) -> str:
            execution_order.append("tool_executed")
            return f"Tool response: {message}"

        # 最初にfunction callを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="test_call",
                            name="custom_tool",
                            arguments='{"message": "test"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])
        chat_client.responses = [function_call_response, final_response]

        # 型注釈に依存して動作するはずである
        agent = ChatAgent(
            chat_client=chat_client, middleware=[type_only_agent, type_only_function], tools=[custom_tool]
        )

        response = await agent.run([ChatMessage(role=Role.USER, text="test")])

        assert response is not None
        assert "type_only_agent" in execution_order
        assert "type_only_function" in execution_order

    async def test_neither_decorator_nor_type(self, chat_client: Any) -> None:
        """デコレータもパラメータ型も指定されていない - 例外を投げるべきである。"""

        async def no_info_middleware(context: Any, next: Any) -> None:  # No decorator, no type
            await next(context)

        # MiddlewareExceptionを発生させるべきである
        with pytest.raises(MiddlewareException, match="Cannot determine middleware type"):
            agent = ChatAgent(chat_client=chat_client, middleware=[no_info_middleware])
            await agent.run([ChatMessage(role=Role.USER, text="test")])

    async def test_insufficient_parameters_error(self, chat_client: Any) -> None:
        """パラメータが不十分なmiddlewareがエラーを発生させることをテストする。"""
        from agent_framework import ChatAgent, agent_middleware

        # パラメータ不足に関するMiddlewareExceptionを発生させるべきである
        with pytest.raises(MiddlewareException, match="must have at least 2 parameters"):

            @agent_middleware  # type: ignore[arg-type]
            async def insufficient_params_middleware(context: Any) -> None:  # Missing 'next' parameter
                pass

            agent = ChatAgent(chat_client=chat_client, middleware=[insufficient_params_middleware])
            await agent.run([ChatMessage(role=Role.USER, text="test")])

    async def test_decorator_markers_preserved(self) -> None:
        """デコレータマーカーが関数に正しく設定されていることをテストする。"""

        @agent_middleware
        async def test_agent_middleware(context: Any, next: Any) -> None:
            pass

        @function_middleware
        async def test_function_middleware(context: Any, next: Any) -> None:
            pass

        # デコレータマーカーが設定されていることを確認する
        assert hasattr(test_agent_middleware, "_middleware_type")
        assert test_agent_middleware._middleware_type == MiddlewareType.AGENT  # type: ignore[attr-defined]

        assert hasattr(test_function_middleware, "_middleware_type")
        assert test_function_middleware._middleware_type == MiddlewareType.FUNCTION  # type: ignore[attr-defined]


class TestChatAgentThreadBehavior:
    """複数の実行にわたるAgentRunContextのThread動作のテストケース。"""

    async def test_agent_run_context_thread_behavior_across_multiple_runs(self, chat_client: "MockChatClient") -> None:
        """複数のagent実行にわたるAgentRunContext.threadプロパティの動作をテストする。"""
        thread_states: list[dict[str, Any]] = []

        class ThreadTrackingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # next()呼び出し前の状態をキャプチャする
                thread_messages = []
                if context.thread and context.thread.message_store:
                    thread_messages = await context.thread.message_store.list_messages()

                before_state = {
                    "before_next": True,
                    "messages_count": len(context.messages),
                    "thread_count": len(thread_messages),
                    "messages_text": [msg.text for msg in context.messages if msg.text],
                    "thread_messages_text": [msg.text for msg in thread_messages if msg.text],
                }
                thread_states.append(before_state)

                await next(context)

                # next()呼び出し後の状態をキャプチャする
                thread_messages_after = []
                if context.thread and context.thread.message_store:
                    thread_messages_after = await context.thread.message_store.list_messages()

                after_state = {
                    "before_next": False,
                    "messages_count": len(context.messages),
                    "thread_count": len(thread_messages_after),
                    "messages_text": [msg.text for msg in context.messages if msg.text],
                    "thread_messages_text": [msg.text for msg in thread_messages_after if msg.text],
                }
                thread_states.append(after_state)

        # ChatMessageStoreをインポートして、メッセージストアファクトリでagentを設定する
        from agent_framework import ChatMessageStore

        # thread追跡middlewareとメッセージストアファクトリ付きでChatAgentを作成する
        middleware = ThreadTrackingMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware], chat_message_store_factory=ChatMessageStore)

        # 実行間でメッセージを保持するthreadを作成する
        thread = agent.get_new_thread()

        # 最初の実行
        first_messages = [ChatMessage(role=Role.USER, text="first message")]
        first_response = await agent.run(first_messages, thread=thread)

        # 最初のレスポンスを検証する
        assert first_response is not None
        assert len(first_response.messages) > 0

        # 2回目の実行 - 同じthreadを使用する
        second_messages = [ChatMessage(role=Role.USER, text="second message")]
        second_response = await agent.run(second_messages, thread=thread)

        # 2回目のレスポンスを検証する
        assert second_response is not None
        assert len(second_response.messages) > 0

        # 両方の実行の状態（各next()の前後）をキャプチャしたことを検証する
        assert len(thread_states) == 4

        # 最初の実行 - next()前
        first_before = thread_states[0]
        assert first_before["before_next"] is True
        assert first_before["messages_count"] == 1
        assert first_before["thread_count"] == 0  # 最初の実行前はThreadは空である
        assert first_before["messages_text"] == ["first message"]
        assert first_before["thread_messages_text"] == []

        # 最初の実行 - next()後
        first_after = thread_states[1]
        assert first_after["before_next"] is False
        assert first_after["messages_count"] == 1  # 入力メッセージは変更されていない
        assert first_after["thread_count"] == 2  # 入力 + レスポンス
        assert first_after["messages_text"] == ["first message"]
        # Threadは入力 + レスポンスを含むべきである
        assert "first message" in first_after["thread_messages_text"]
        assert "test response" in " ".join(first_after["thread_messages_text"])

        # 2回目の実行 - next()前
        second_before = thread_states[2]
        assert second_before["before_next"] is True
        assert second_before["messages_count"] == 1  # 現在の実行の入力のみ
        assert second_before["thread_count"] == 2  # 前回の実行履歴（入力 + レスポンス）
        assert second_before["messages_text"] == ["second message"]
        # Threadは前回の実行履歴を含むが、まだ現在の入力は含まないべきである
        assert "first message" in second_before["thread_messages_text"]
        assert "test response" in " ".join(second_before["thread_messages_text"])
        assert "second message" not in second_before["thread_messages_text"]

        # 2回目の実行 - next()後
        second_after = thread_states[3]
        assert second_after["before_next"] is False
        assert second_after["messages_count"] == 1  # 入力メッセージは変更されていない
        assert second_after["thread_count"] == 4  # 前回の履歴 + 現在の入力 + 現在のレスポンス
        assert second_after["messages_text"] == ["second message"]
        # Threadは最初の入力 + 最初のレスポンス + 2回目の入力 + 2回目のレスポンスを含むべきである
        assert "first message" in second_after["thread_messages_text"]
        assert "second message" in second_after["thread_messages_text"]
        # "test response"エントリが2つあるはず（各実行ごとに1つ）
        response_count = sum(1 for text in second_after["thread_messages_text"] if "test response" in text)
        assert response_count == 2


class TestChatAgentChatMiddleware:
    """ChatAgentとのchat middleware統合のテストケース。"""

    async def test_class_based_chat_middleware_with_chat_agent(self) -> None:
        """クラスベースのchat middlewareをChatAgentでテストする。"""
        execution_order: list[str] = []

        class TrackingChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("chat_middleware_before")
                await next(context)
                execution_order.append("chat_middleware_after")

        # chat middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        middleware = TrackingChatMiddleware()
        agent = ChatAgent(chat_client=chat_client, middleware=[middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT
        assert "test response" in response.messages[0].text
        assert execution_order == ["chat_middleware_before", "chat_middleware_after"]

    async def test_function_based_chat_middleware_with_chat_agent(self) -> None:
        """関数ベースのchat middlewareをChatAgentでテストする。"""
        execution_order: list[str] = []

        async def tracking_chat_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            execution_order.append("chat_middleware_before")
            await next(context)
            execution_order.append("chat_middleware_after")

        # 関数ベースのchat middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[tracking_chat_middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].role == Role.ASSISTANT
        assert "test response" in response.messages[0].text
        assert execution_order == ["chat_middleware_before", "chat_middleware_after"]

    async def test_chat_middleware_can_modify_messages(self) -> None:
        """chat middlewareがモデルに送信する前にメッセージを変更できることをテストする。"""

        @chat_middleware
        async def message_modifier_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            # 最初のメッセージにプレフィックスを追加して変更する
            if context.messages:
                for idx, msg in enumerate(context.messages):
                    if msg.role.value == "system":
                        continue
                    original_text = msg.text or ""
                    context.messages[idx] = ChatMessage(role=msg.role, text=f"MODIFIED: {original_text}")
                    break
            await next(context)

        # メッセージ変更middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[message_modifier_middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # メッセージが変更されたことを検証する（MockBaseChatClientは入力をそのまま返す）
        assert response and response.messages
        assert "MODIFIED: test message" in response.messages[0].text

    async def test_chat_middleware_can_override_response(self) -> None:
        """chat middlewareがレスポンスをオーバーライドできることをテストする。"""

        @chat_middleware
        async def response_override_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            # next()を呼ばずにレスポンスをオーバーライドする
            context.result = ChatResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, text="Middleware overridden response")],
                response_id="middleware-response-123",
            )
            context.terminate = True

        # レスポンスオーバーライドmiddleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[response_override_middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスがオーバーライドされたことを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].text == "Middleware overridden response"
        assert response.response_id == "middleware-response-123"

    async def test_multiple_chat_middleware_execution_order(self) -> None:
        """複数のchat middlewareが正しい順序で実行されることをテストする。"""
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

        # 複数のchat middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[first_middleware, second_middleware])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert execution_order == ["first_before", "second_before", "second_after", "first_after"]

    async def test_chat_middleware_with_streaming(self) -> None:
        """ストリーミングレスポンスを伴うchat middlewareのテスト。"""
        execution_order: list[str] = []
        streaming_flags: list[bool] = []

        class StreamingTrackingChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("streaming_chat_before")
                streaming_flags.append(context.is_streaming)
                await next(context)
                execution_order.append("streaming_chat_after")

        # chat middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[StreamingTrackingChatMiddleware()])

        # モックのストリーミングレスポンスを設定する
        chat_client.streaming_responses = [
            [
                ChatResponseUpdate(contents=[TextContent(text="Stream")], role=Role.ASSISTANT),
                ChatResponseUpdate(contents=[TextContent(text=" response")], role=Role.ASSISTANT),
            ]
        ]

        # ストリーミング実行を行う
        messages = [ChatMessage(role=Role.USER, text="test message")]
        updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream(messages):
            updates.append(update)

        # ストリーミングレスポンスを検証する
        assert len(updates) >= 1  # 少なくともいくつかの更新がある
        assert execution_order == ["streaming_chat_before", "streaming_chat_after"]

        # ストリーミングフラグが設定されていることを検証する（少なくとも1つTrue）
        assert True in streaming_flags

    async def test_chat_middleware_termination_before_execution(self) -> None:
        """chat middlewareがnext()を呼ぶ前に実行を終了できることをテストする。"""
        execution_order: list[str] = []

        class PreTerminationChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("middleware_before")
                context.terminate = True
                # 終了するためカスタムレスポンスを設定する
                context.result = ChatResponse(
                    messages=[ChatMessage(role=Role.ASSISTANT, text="Terminated by middleware")]
                )
                # next()を呼ぶがterminate=Trueなので実行は停止するはずである
                await next(context)
                execution_order.append("middleware_after")

        # 終了middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[PreTerminationChatMiddleware()])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスがmiddleware由来であることを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert response.messages[0].text == "Terminated by middleware"
        assert execution_order == ["middleware_before", "middleware_after"]

    async def test_chat_middleware_termination_after_execution(self) -> None:
        """chat middlewareがnext()呼び出し後に実行を終了できることをテストする。"""
        execution_order: list[str] = []

        class PostTerminationChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("middleware_before")
                await next(context)
                execution_order.append("middleware_after")
                context.terminate = True

        # 終了middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[PostTerminationChatMiddleware()])

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages)

        # レスポンスが実際の実行由来であることを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert "test response" in response.messages[0].text
        assert execution_order == ["middleware_before", "middleware_after"]

    async def test_combined_middleware(self) -> None:
        """複合middlewareタイプを持つChatAgentをテストする。"""
        execution_order: list[str] = []

        async def agent_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            execution_order.append("agent_middleware_before")
            await next(context)
            execution_order.append("agent_middleware_after")

        async def chat_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            execution_order.append("chat_middleware_before")
            await next(context)
            execution_order.append("chat_middleware_after")

        async def function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_middleware_before")
            await next(context)
            execution_order.append("function_middleware_after")

        # 最初にfunction callを返し、その後通常のレスポンスを返すモックを設定する
        function_call_response = ChatResponse(
            messages=[
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[
                        FunctionCallContent(
                            call_id="call_456",
                            name="sample_tool_function",
                            arguments='{"location": "San Francisco"}',
                        )
                    ],
                )
            ]
        )
        final_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Final response")])

        chat_client = use_function_invocation(MockBaseChatClient)()
        chat_client.run_responses = [function_call_response, final_response]

        # function middlewareとtools付きでChatAgentを作成する
        agent = ChatAgent(
            chat_client=chat_client,
            middleware=[chat_middleware, function_middleware, agent_middleware],
            tools=[sample_tool_function],
        )

        # agentを実行する
        messages = [ChatMessage(role=Role.USER, text="Get weather for San Francisco")]
        response = await agent.run(messages)

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0
        assert chat_client.call_count == 2  # 2回の呼び出し：function call用と最終レスポンス用

        # function middlewareが実行されたことを検証する
        assert execution_order == [
            "agent_middleware_before",
            "chat_middleware_before",
            "chat_middleware_after",
            "function_middleware_before",
            "function_middleware_after",
            "chat_middleware_before",
            "chat_middleware_after",
            "agent_middleware_after",
        ]

        # function callと結果がレスポンスに含まれていることを検証する
        all_contents = [content for message in response.messages for content in message.contents]
        function_calls = [c for c in all_contents if isinstance(c, FunctionCallContent)]
        function_results = [c for c in all_contents if isinstance(c, FunctionResultContent)]

        assert len(function_calls) == 1
        assert len(function_results) == 1
        assert function_calls[0].name == "sample_tool_function"
        assert function_results[0].call_id == function_calls[0].call_id

    async def test_agent_middleware_can_access_and_override_custom_kwargs(self) -> None:
        """agent middlewareがtemperatureなどのカスタムパラメータにアクセスし、オーバーライドできることをテストする。"""
        captured_kwargs: dict[str, Any] = {}
        modified_kwargs: dict[str, Any] = {}

        @agent_middleware
        async def kwargs_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            # 元のkwargsをキャプチャする
            captured_kwargs.update(context.kwargs)

            # いくつかのkwargsを変更する
            context.kwargs["temperature"] = 0.9
            context.kwargs["max_tokens"] = 500
            context.kwargs["new_param"] = "added_by_middleware"

            # 検証用に変更したkwargsを保存する
            modified_kwargs.update(context.kwargs)

            await next(context)

        # agent middleware付きでChatAgentを作成する
        chat_client = MockBaseChatClient()
        agent = ChatAgent(chat_client=chat_client, middleware=[kwargs_middleware])

        # カスタムパラメータでagentを実行する
        messages = [ChatMessage(role=Role.USER, text="test message")]
        response = await agent.run(messages, temperature=0.7, max_tokens=100, custom_param="test_value")

        # レスポンスを検証する
        assert response is not None
        assert len(response.messages) > 0

        # Verify middlewareが元のkwargsをキャプチャしたことを検証する
        assert captured_kwargs["temperature"] == 0.7
        assert captured_kwargs["max_tokens"] == 100
        assert captured_kwargs["custom_param"] == "test_value"

        # Verify middlewareがkwargsを変更できることを検証する
        assert modified_kwargs["temperature"] == 0.9
        assert modified_kwargs["max_tokens"] == 500
        assert modified_kwargs["new_param"] == "added_by_middleware"
        assert modified_kwargs["custom_param"] == "test_value"  # まだ存在しているはず
