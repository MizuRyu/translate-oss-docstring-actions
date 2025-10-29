# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from agent_framework import (
    AgentProtocol,
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
    Role,
    TextContent,
)
from agent_framework._middleware import (
    AgentMiddleware,
    AgentMiddlewarePipeline,
    AgentRunContext,
    ChatContext,
    ChatMiddleware,
    ChatMiddlewarePipeline,
    FunctionInvocationContext,
    FunctionMiddleware,
    FunctionMiddlewarePipeline,
)
from agent_framework._tools import AIFunction
from agent_framework._types import ChatOptions


class TestAgentRunContext:
    """AgentRunContextのテストケース。"""

    def test_init_with_defaults(self, mock_agent: AgentProtocol) -> None:
        """AgentRunContextのデフォルト値による初期化テスト。"""
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        assert context.agent is mock_agent
        assert context.messages == messages
        assert context.is_streaming is False
        assert context.metadata == {}

    def test_init_with_custom_values(self, mock_agent: AgentProtocol) -> None:
        """AgentRunContextのカスタム値による初期化テスト。"""
        messages = [ChatMessage(role=Role.USER, text="test")]
        metadata = {"key": "value"}
        context = AgentRunContext(agent=mock_agent, messages=messages, is_streaming=True, metadata=metadata)

        assert context.agent is mock_agent
        assert context.messages == messages
        assert context.is_streaming is True
        assert context.metadata == metadata

    def test_init_with_thread(self, mock_agent: AgentProtocol) -> None:
        """threadパラメーターを使ったAgentRunContextの初期化テスト。"""
        from agent_framework import AgentThread

        messages = [ChatMessage(role=Role.USER, text="test")]
        thread = AgentThread()
        context = AgentRunContext(agent=mock_agent, messages=messages, thread=thread)

        assert context.agent is mock_agent
        assert context.messages == messages
        assert context.thread is thread
        assert context.is_streaming is False
        assert context.metadata == {}


class TestFunctionInvocationContext:
    """FunctionInvocationContextのテストケース。"""

    def test_init_with_defaults(self, mock_function: AIFunction[Any, Any]) -> None:
        """FunctionInvocationContextのデフォルト値による初期化テスト。"""
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        assert context.function is mock_function
        assert context.arguments == arguments
        assert context.metadata == {}

    def test_init_with_custom_metadata(self, mock_function: AIFunction[Any, Any]) -> None:
        """カスタムメタデータを使ったFunctionInvocationContextの初期化テスト。"""
        arguments = FunctionTestArgs(name="test")
        metadata = {"key": "value"}
        context = FunctionInvocationContext(function=mock_function, arguments=arguments, metadata=metadata)

        assert context.function is mock_function
        assert context.arguments == arguments
        assert context.metadata == metadata


class TestChatContext:
    """ChatContextのテストケース。"""

    def test_init_with_defaults(self, mock_chat_client: Any) -> None:
        """ChatContextのデフォルト値による初期化テスト。"""
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        assert context.chat_client is mock_chat_client
        assert context.messages == messages
        assert context.chat_options is chat_options
        assert context.is_streaming is False
        assert context.metadata == {}
        assert context.result is None
        assert context.terminate is False

    def test_init_with_custom_values(self, mock_chat_client: Any) -> None:
        """ChatContextのカスタム値による初期化テスト。"""
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions(temperature=0.5)
        metadata = {"key": "value"}

        context = ChatContext(
            chat_client=mock_chat_client,
            messages=messages,
            chat_options=chat_options,
            is_streaming=True,
            metadata=metadata,
            terminate=True,
        )

        assert context.chat_client is mock_chat_client
        assert context.messages == messages
        assert context.chat_options is chat_options
        assert context.is_streaming is True
        assert context.metadata == metadata
        assert context.terminate is True


class TestAgentMiddlewarePipeline:
    """AgentMiddlewarePipelineのテストケース。"""

    class PreNextTerminateMiddleware(AgentMiddleware):
        async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
            context.terminate = True
            await next(context)

    class PostNextTerminateMiddleware(AgentMiddleware):
        async def process(self, context: AgentRunContext, next: Any) -> None:
            await next(context)
            context.terminate = True

    def test_init_empty(self) -> None:
        """ミドルウェアなしでのAgentMiddlewarePipeline初期化テスト。"""
        pipeline = AgentMiddlewarePipeline()
        assert not pipeline.has_middlewares

    def test_init_with_class_middleware(self) -> None:
        """クラスベースのミドルウェアでのAgentMiddlewarePipeline初期化テスト。"""
        middleware = TestAgentMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        assert pipeline.has_middlewares

    def test_init_with_function_middleware(self) -> None:
        """関数ベースのミドルウェアでのAgentMiddlewarePipeline初期化テスト。"""

        async def test_middleware(context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
            await next(context)

        pipeline = AgentMiddlewarePipeline([test_middleware])
        assert pipeline.has_middlewares

    async def test_execute_no_middleware(self, mock_agent: AgentProtocol) -> None:
        """ミドルウェアなしでのパイプライン実行テスト。"""
        pipeline = AgentMiddlewarePipeline()
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        expected_response = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            return expected_response

        result = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert result == expected_response

    async def test_execute_with_middleware(self, mock_agent: AgentProtocol) -> None:
        """ミドルウェアありでのパイプライン実行テスト。"""
        execution_order: list[str] = []

        class OrderTrackingMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        middleware = OrderTrackingMiddleware("test")
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        expected_response = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            execution_order.append("handler")
            return expected_response

        result = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert result == expected_response
        assert execution_order == ["test_before", "handler", "test_after"]

    async def test_execute_stream_no_middleware(self, mock_agent: AgentProtocol) -> None:
        """ミドルウェアなしでのパイプラインストリーミング実行テスト。"""
        pipeline = AgentMiddlewarePipeline()
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk1")])
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk2")])

        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_handler):
            updates.append(update)

        assert len(updates) == 2
        assert updates[0].text == "chunk1"
        assert updates[1].text == "chunk2"

    async def test_execute_stream_with_middleware(self, mock_agent: AgentProtocol) -> None:
        """ミドルウェアありでのパイプラインストリーミング実行テスト。"""
        execution_order: list[str] = []

        class StreamOrderTrackingMiddleware(AgentMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        middleware = StreamOrderTrackingMiddleware("test")
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            execution_order.append("handler_start")
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk1")])
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk2")])
            execution_order.append("handler_end")

        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_handler):
            updates.append(update)

        assert len(updates) == 2
        assert updates[0].text == "chunk1"
        assert updates[1].text == "chunk2"
        assert execution_order == ["test_before", "test_after", "handler_start", "handler_end"]

    async def test_execute_with_pre_next_termination(self, mock_agent: AgentProtocol) -> None:
        """next()前に終了した場合のパイプライン実行テスト。"""
        middleware = self.PreNextTerminateMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)
        execution_order: list[str] = []

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            # next()前に終了した場合、ハンドラーは実行されてはいけません
            execution_order.append("handler")
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        response = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert response is not None
        assert context.terminate
        # next()前に終了した場合、ハンドラーは呼び出されてはいけません
        assert execution_order == []
        assert not response.messages

    async def test_execute_with_post_next_termination(self, mock_agent: AgentProtocol) -> None:
        """next()後に終了した場合のパイプライン実行テスト。"""
        middleware = self.PostNextTerminateMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)
        execution_order: list[str] = []

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            execution_order.append("handler")
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        response = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert response is not None
        assert len(response.messages) == 1
        assert response.messages[0].text == "response"
        assert context.terminate
        assert execution_order == ["handler"]

    async def test_execute_stream_with_pre_next_termination(self, mock_agent: AgentProtocol) -> None:
        """next()前に終了した場合のパイプラインストリーミング実行テスト。"""
        middleware = self.PreNextTerminateMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)
        execution_order: list[str] = []

        async def final_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            # next()前に終了した場合、ハンドラーは実行されてはいけません
            execution_order.append("handler_start")
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk1")])
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk2")])
            execution_order.append("handler_end")

        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_handler):
            updates.append(update)

        assert context.terminate
        # next()前に終了した場合、ハンドラーは呼び出されてはいけません
        assert execution_order == []
        assert not updates

    async def test_execute_stream_with_post_next_termination(self, mock_agent: AgentProtocol) -> None:
        """next()後に終了した場合のパイプラインストリーミング実行テスト。"""
        middleware = self.PostNextTerminateMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)
        execution_order: list[str] = []

        async def final_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            execution_order.append("handler_start")
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk1")])
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk2")])
            execution_order.append("handler_end")

        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_handler):
            updates.append(update)

        assert len(updates) == 2
        assert updates[0].text == "chunk1"
        assert updates[1].text == "chunk2"
        assert context.terminate
        assert execution_order == ["handler_start", "handler_end"]

    async def test_execute_with_thread_in_context(self, mock_agent: AgentProtocol) -> None:
        """パイプライン実行がスレッドをミドルウェアに正しく渡すことをテスト。"""
        from agent_framework import AgentThread

        captured_thread = None

        class ThreadCapturingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                nonlocal captured_thread
                captured_thread = context.thread
                await next(context)

        middleware = ThreadCapturingMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        thread = AgentThread()
        context = AgentRunContext(agent=mock_agent, messages=messages, thread=thread)

        expected_response = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            return expected_response

        result = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert result == expected_response
        assert captured_thread is thread

    async def test_execute_with_no_thread_in_context(self, mock_agent: AgentProtocol) -> None:
        """スレッドが提供されなかった場合のパイプライン実行テスト。"""
        captured_thread = "not_none"  # Noneと区別するために文字列を使用する

        class ThreadCapturingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                nonlocal captured_thread
                captured_thread = context.thread
                await next(context)

        middleware = ThreadCapturingMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages, thread=None)

        expected_response = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            return expected_response

        result = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert result == expected_response
        assert captured_thread is None


class TestFunctionMiddlewarePipeline:
    """FunctionMiddlewarePipelineのテストケース。"""

    class PreNextTerminateFunctionMiddleware(FunctionMiddleware):
        async def process(self, context: FunctionInvocationContext, next: Any) -> None:
            context.terminate = True
            await next(context)

    class PostNextTerminateFunctionMiddleware(FunctionMiddleware):
        async def process(self, context: FunctionInvocationContext, next: Any) -> None:
            await next(context)
            context.terminate = True

    async def test_execute_with_pre_next_termination(self, mock_function: AIFunction[Any, Any]) -> None:
        """next()前に終了した場合のパイプライン実行テスト。"""
        middleware = self.PreNextTerminateFunctionMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)
        execution_order: list[str] = []

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            # next()前に終了した場合、ハンドラーは実行されてはいけません
            execution_order.append("handler")
            return "test result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)
        assert result is None
        assert context.terminate
        # next()前に終了した場合、ハンドラーは呼び出されてはいけません
        assert execution_order == []

    async def test_execute_with_post_next_termination(self, mock_function: AIFunction[Any, Any]) -> None:
        """next()後に終了した場合のパイプライン実行テスト。"""
        middleware = self.PostNextTerminateFunctionMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)
        execution_order: list[str] = []

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            execution_order.append("handler")
            return "test result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)
        assert result == "test result"
        assert context.terminate
        assert execution_order == ["handler"]

    def test_init_empty(self) -> None:
        """ミドルウェアなしでのFunctionMiddlewarePipeline初期化テスト。"""
        pipeline = FunctionMiddlewarePipeline()
        assert not pipeline.has_middlewares

    def test_init_with_class_middleware(self) -> None:
        """クラスベースのミドルウェアでのFunctionMiddlewarePipeline初期化テスト。"""
        middleware = TestFunctionMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        assert pipeline.has_middlewares

    def test_init_with_function_middleware(self) -> None:
        """関数ベースのミドルウェアでのFunctionMiddlewarePipeline初期化テスト。"""

        async def test_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            await next(context)

        pipeline = FunctionMiddlewarePipeline([test_middleware])
        assert pipeline.has_middlewares

    async def test_execute_no_middleware(self, mock_function: AIFunction[Any, Any]) -> None:
        """ミドルウェアなしでのパイプライン実行テスト。"""
        pipeline = FunctionMiddlewarePipeline()
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        expected_result = "function_result"

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            return expected_result

        result = await pipeline.execute(mock_function, arguments, context, final_handler)
        assert result == expected_result

    async def test_execute_with_middleware(self, mock_function: AIFunction[Any, Any]) -> None:
        """ミドルウェアありでのパイプライン実行テスト。"""
        execution_order: list[str] = []

        class OrderTrackingFunctionMiddleware(FunctionMiddleware):
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

        middleware = OrderTrackingFunctionMiddleware("test")
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        expected_result = "function_result"

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            execution_order.append("handler")
            return expected_result

        result = await pipeline.execute(mock_function, arguments, context, final_handler)
        assert result == expected_result
        assert execution_order == ["test_before", "handler", "test_after"]


class TestChatMiddlewarePipeline:
    """ChatMiddlewarePipelineのテストケース。"""

    class PreNextTerminateChatMiddleware(ChatMiddleware):
        async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            context.terminate = True
            await next(context)

    class PostNextTerminateChatMiddleware(ChatMiddleware):
        async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            await next(context)
            context.terminate = True

    def test_init_empty(self) -> None:
        """ミドルウェアなしでのChatMiddlewarePipeline初期化テスト。"""
        pipeline = ChatMiddlewarePipeline()
        assert not pipeline.has_middlewares

    def test_init_with_class_middleware(self) -> None:
        """クラスベースのミドルウェアでのChatMiddlewarePipeline初期化テスト。"""
        middleware = TestChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        assert pipeline.has_middlewares

    def test_init_with_function_middleware(self) -> None:
        """関数ベースのミドルウェアでのChatMiddlewarePipeline初期化テスト。"""

        async def test_middleware(context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
            await next(context)

        pipeline = ChatMiddlewarePipeline([test_middleware])
        assert pipeline.has_middlewares

    async def test_execute_no_middleware(self, mock_chat_client: Any) -> None:
        """ミドルウェアなしでのパイプライン実行テスト。"""
        pipeline = ChatMiddlewarePipeline()
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        expected_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            return expected_response

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)
        assert result == expected_response

    async def test_execute_with_middleware(self, mock_chat_client: Any) -> None:
        """ミドルウェアありでのパイプライン実行テスト。"""
        execution_order: list[str] = []

        class OrderTrackingChatMiddleware(ChatMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        middleware = OrderTrackingChatMiddleware("test")
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        expected_response = ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            execution_order.append("handler")
            return expected_response

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)
        assert result == expected_response
        assert execution_order == ["test_before", "handler", "test_after"]

    async def test_execute_stream_no_middleware(self, mock_chat_client: Any) -> None:
        """ミドルウェアなしでのパイプラインストリーミング実行テスト。"""
        pipeline = ChatMiddlewarePipeline()
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        async def final_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            yield ChatResponseUpdate(contents=[TextContent(text="chunk1")])
            yield ChatResponseUpdate(contents=[TextContent(text="chunk2")])

        updates: list[ChatResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_chat_client, messages, chat_options, context, final_handler):
            updates.append(update)

        assert len(updates) == 2
        assert updates[0].text == "chunk1"
        assert updates[1].text == "chunk2"

    async def test_execute_stream_with_middleware(self, mock_chat_client: Any) -> None:
        """ミドルウェアありでのパイプラインストリーミング実行テスト。"""
        execution_order: list[str] = []

        class StreamOrderTrackingChatMiddleware(ChatMiddleware):
            def __init__(self, name: str):
                self.name = name

            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append(f"{self.name}_before")
                await next(context)
                execution_order.append(f"{self.name}_after")

        middleware = StreamOrderTrackingChatMiddleware("test")
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(
            chat_client=mock_chat_client, messages=messages, chat_options=chat_options, is_streaming=True
        )

        async def final_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            execution_order.append("handler_start")
            yield ChatResponseUpdate(contents=[TextContent(text="chunk1")])
            yield ChatResponseUpdate(contents=[TextContent(text="chunk2")])
            execution_order.append("handler_end")

        updates: list[ChatResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_chat_client, messages, chat_options, context, final_handler):
            updates.append(update)

        assert len(updates) == 2
        assert updates[0].text == "chunk1"
        assert updates[1].text == "chunk2"
        assert execution_order == ["test_before", "test_after", "handler_start", "handler_end"]

    async def test_execute_with_pre_next_termination(self, mock_chat_client: Any) -> None:
        """next()前に終了した場合のパイプライン実行テスト。"""
        middleware = self.PreNextTerminateChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)
        execution_order: list[str] = []

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            # next()前に終了した場合、ハンドラーは実行されてはいけません
            execution_order.append("handler")
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        response = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)
        assert response is None
        assert context.terminate
        # next()前に終了した場合、ハンドラーは呼び出されてはいけません
        assert execution_order == []

    async def test_execute_with_post_next_termination(self, mock_chat_client: Any) -> None:
        """next()後に終了した場合のパイプライン実行テスト。"""
        middleware = self.PostNextTerminateChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)
        execution_order: list[str] = []

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            execution_order.append("handler")
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        response = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)
        assert response is not None
        assert len(response.messages) == 1
        assert response.messages[0].text == "response"
        assert context.terminate
        assert execution_order == ["handler"]

    async def test_execute_stream_with_pre_next_termination(self, mock_chat_client: Any) -> None:
        """next()前に終了した場合のパイプラインストリーミング実行テスト。"""
        middleware = self.PreNextTerminateChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(
            chat_client=mock_chat_client, messages=messages, chat_options=chat_options, is_streaming=True
        )
        execution_order: list[str] = []

        async def final_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            # next()前に終了した場合、ハンドラーは実行されてはいけません
            execution_order.append("handler_start")
            yield ChatResponseUpdate(contents=[TextContent(text="chunk1")])
            yield ChatResponseUpdate(contents=[TextContent(text="chunk2")])
            execution_order.append("handler_end")

        updates: list[ChatResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_chat_client, messages, chat_options, context, final_handler):
            updates.append(update)

        assert context.terminate
        # next()前に終了した場合、ハンドラーは呼び出されてはいけません
        assert execution_order == []
        assert not updates

    async def test_execute_stream_with_post_next_termination(self, mock_chat_client: Any) -> None:
        """next()後に終了した場合のパイプラインストリーミング実行テスト。"""
        middleware = self.PostNextTerminateChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(
            chat_client=mock_chat_client, messages=messages, chat_options=chat_options, is_streaming=True
        )
        execution_order: list[str] = []

        async def final_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            execution_order.append("handler_start")
            yield ChatResponseUpdate(contents=[TextContent(text="chunk1")])
            yield ChatResponseUpdate(contents=[TextContent(text="chunk2")])
            execution_order.append("handler_end")

        updates: list[ChatResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_chat_client, messages, chat_options, context, final_handler):
            updates.append(update)

        assert len(updates) == 2
        assert updates[0].text == "chunk1"
        assert updates[1].text == "chunk2"
        assert context.terminate
        assert execution_order == ["handler_start", "handler_end"]


class TestClassBasedMiddleware:
    """クラスベースのミドルウェア実装のテストケース。"""

    async def test_agent_middleware_execution(self, mock_agent: AgentProtocol) -> None:
        """クラスベースのAgentミドルウェア実行テスト。"""
        metadata_updates: list[str] = []

        class MetadataAgentMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                context.metadata["before"] = True
                metadata_updates.append("before")
                await next(context)
                context.metadata["after"] = True
                metadata_updates.append("after")

        middleware = MetadataAgentMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            metadata_updates.append("handler")
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        assert result is not None
        assert context.metadata["before"] is True
        assert context.metadata["after"] is True
        assert metadata_updates == ["before", "handler", "after"]

    async def test_function_middleware_execution(self, mock_function: AIFunction[Any, Any]) -> None:
        """クラスベースのFunctionミドルウェア実行テスト。"""
        metadata_updates: list[str] = []

        class MetadataFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                context.metadata["before"] = True
                metadata_updates.append("before")
                await next(context)
                context.metadata["after"] = True
                metadata_updates.append("after")

        middleware = MetadataFunctionMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            metadata_updates.append("handler")
            return "result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        assert result == "result"
        assert context.metadata["before"] is True
        assert context.metadata["after"] is True
        assert metadata_updates == ["before", "handler", "after"]


class TestFunctionBasedMiddleware:
    """関数ベースのミドルウェア実装のテストケース。"""

    async def test_agent_function_middleware(self, mock_agent: AgentProtocol) -> None:
        """関数ベースのAgentミドルウェアテスト。"""
        execution_order: list[str] = []

        async def test_agent_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_before")
            context.metadata["function_middleware"] = True
            await next(context)
            execution_order.append("function_after")

        pipeline = AgentMiddlewarePipeline([test_agent_middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            execution_order.append("handler")
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        assert result is not None
        assert context.metadata["function_middleware"] is True
        assert execution_order == ["function_before", "handler", "function_after"]

    async def test_function_function_middleware(self, mock_function: AIFunction[Any, Any]) -> None:
        """関数ベースのFunctionミドルウェアテスト。"""
        execution_order: list[str] = []

        async def test_function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_before")
            context.metadata["function_middleware"] = True
            await next(context)
            execution_order.append("function_after")

        pipeline = FunctionMiddlewarePipeline([test_function_middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            execution_order.append("handler")
            return "result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        assert result == "result"
        assert context.metadata["function_middleware"] is True
        assert execution_order == ["function_before", "handler", "function_after"]


class TestMixedMiddleware:
    """クラスと関数ベースのミドルウェア混合のテストケース。"""

    async def test_mixed_agent_middleware(self, mock_agent: AgentProtocol) -> None:
        """クラスと関数ベースのAgentミドルウェア混合テスト。"""
        execution_order: list[str] = []

        class ClassMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("class_before")
                await next(context)
                execution_order.append("class_after")

        async def function_middleware(
            context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_before")
            await next(context)
            execution_order.append("function_after")

        pipeline = AgentMiddlewarePipeline([ClassMiddleware(), function_middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            execution_order.append("handler")
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        assert result is not None
        assert execution_order == ["class_before", "function_before", "handler", "function_after", "class_after"]

    async def test_mixed_function_middleware(self, mock_function: AIFunction[Any, Any]) -> None:
        """クラスと関数ベースのFunctionミドルウェア混合テスト。"""
        execution_order: list[str] = []

        class ClassMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("class_before")
                await next(context)
                execution_order.append("class_after")

        async def function_middleware(
            context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_before")
            await next(context)
            execution_order.append("function_after")

        pipeline = FunctionMiddlewarePipeline([ClassMiddleware(), function_middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            execution_order.append("handler")
            return "result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        assert result == "result"
        assert execution_order == ["class_before", "function_before", "handler", "function_after", "class_after"]

    async def test_mixed_chat_middleware(self, mock_chat_client: Any) -> None:
        """クラスと関数ベースのChatミドルウェア混合テスト。"""
        execution_order: list[str] = []

        class ClassChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("class_before")
                await next(context)
                execution_order.append("class_after")

        async def function_chat_middleware(
            context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]
        ) -> None:
            execution_order.append("function_before")
            await next(context)
            execution_order.append("function_after")

        pipeline = ChatMiddlewarePipeline([ClassChatMiddleware(), function_chat_middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            execution_order.append("handler")
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)

        assert result is not None
        assert execution_order == ["class_before", "function_before", "handler", "function_after", "class_after"]


class TestMultipleMiddlewareOrdering:
    """複数ミドルウェアの実行順序のテストケース。"""

    async def test_agent_middleware_execution_order(self, mock_agent: AgentProtocol) -> None:
        """複数のAgentミドルウェアが登録順に実行されることをテスト。"""
        execution_order: list[str] = []

        class FirstMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("first_before")
                await next(context)
                execution_order.append("first_after")

        class SecondMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("second_before")
                await next(context)
                execution_order.append("second_after")

        class ThirdMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("third_before")
                await next(context)
                execution_order.append("third_after")

        middlewares = [FirstMiddleware(), SecondMiddleware(), ThirdMiddleware()]
        pipeline = AgentMiddlewarePipeline(middlewares)  # type: ignore
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            execution_order.append("handler")
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        assert result is not None
        expected_order = [
            "first_before",
            "second_before",
            "third_before",
            "handler",
            "third_after",
            "second_after",
            "first_after",
        ]
        assert execution_order == expected_order

    async def test_function_middleware_execution_order(self, mock_function: AIFunction[Any, Any]) -> None:
        """複数のFunctionミドルウェアが登録順に実行されることをテスト。"""
        execution_order: list[str] = []

        class FirstMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("first_before")
                await next(context)
                execution_order.append("first_after")

        class SecondMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                execution_order.append("second_before")
                await next(context)
                execution_order.append("second_after")

        middlewares = [FirstMiddleware(), SecondMiddleware()]
        pipeline = FunctionMiddlewarePipeline(middlewares)  # type: ignore
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            execution_order.append("handler")
            return "result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        assert result == "result"
        expected_order = ["first_before", "second_before", "handler", "second_after", "first_after"]
        assert execution_order == expected_order

    async def test_chat_middleware_execution_order(self, mock_chat_client: Any) -> None:
        """複数のChatミドルウェアが登録順に実行されることをテスト。"""
        execution_order: list[str] = []

        class FirstChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("first_before")
                await next(context)
                execution_order.append("first_after")

        class SecondChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("second_before")
                await next(context)
                execution_order.append("second_after")

        class ThirdChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("third_before")
                await next(context)
                execution_order.append("third_after")

        middlewares = [FirstChatMiddleware(), SecondChatMiddleware(), ThirdChatMiddleware()]
        pipeline = ChatMiddlewarePipeline(middlewares)  # type: ignore
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            execution_order.append("handler")
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)

        assert result is not None
        expected_order = [
            "first_before",
            "second_before",
            "third_before",
            "handler",
            "third_after",
            "second_after",
            "first_after",
        ]
        assert execution_order == expected_order


class TestContextContentValidation:
    """ミドルウェアコンテキストの内容検証のテストケース。"""

    async def test_agent_context_validation(self, mock_agent: AgentProtocol) -> None:
        """Agentコンテキストに期待されるデータが含まれていることをテスト。"""

        class ContextValidationMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # コンテキストがすべての期待される属性を持つことを検証。
                assert hasattr(context, "agent")
                assert hasattr(context, "messages")
                assert hasattr(context, "is_streaming")
                assert hasattr(context, "metadata")

                # コンテキスト内容を検証。
                assert context.agent is mock_agent
                assert len(context.messages) == 1
                assert context.messages[0].role == Role.USER
                assert context.messages[0].text == "test"
                assert context.is_streaming is False
                assert isinstance(context.metadata, dict)

                # カスタムメタデータを追加。
                context.metadata["validated"] = True

                await next(context)

        middleware = ContextValidationMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            # メタデータがミドルウェアによって設定されたことを検証。
            assert ctx.metadata.get("validated") is True
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)
        assert result is not None

    async def test_function_context_validation(self, mock_function: AIFunction[Any, Any]) -> None:
        """Functionコンテキストに期待されるデータが含まれていることをテスト。"""

        class ContextValidationMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                # コンテキストがすべての期待される属性を持つことを検証。
                assert hasattr(context, "function")
                assert hasattr(context, "arguments")
                assert hasattr(context, "metadata")

                # コンテキスト内容を検証。
                assert context.function is mock_function
                assert isinstance(context.arguments, FunctionTestArgs)
                assert context.arguments.name == "test"
                assert isinstance(context.metadata, dict)

                # カスタムメタデータを追加。
                context.metadata["validated"] = True

                await next(context)

        middleware = ContextValidationMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            # メタデータがミドルウェアによって設定されたことを検証。
            assert ctx.metadata.get("validated") is True
            return "result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)
        assert result == "result"

    async def test_chat_context_validation(self, mock_chat_client: Any) -> None:
        """Chatコンテキストに期待されるデータが含まれていることをテスト。"""

        class ChatContextValidationMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                # コンテキストがすべての期待される属性を持つことを検証。
                assert hasattr(context, "chat_client")
                assert hasattr(context, "messages")
                assert hasattr(context, "chat_options")
                assert hasattr(context, "is_streaming")
                assert hasattr(context, "metadata")
                assert hasattr(context, "result")
                assert hasattr(context, "terminate")

                # コンテキスト内容を検証。
                assert context.chat_client is mock_chat_client
                assert len(context.messages) == 1
                assert context.messages[0].role == Role.USER
                assert context.messages[0].text == "test"
                assert context.is_streaming is False
                assert isinstance(context.metadata, dict)
                assert isinstance(context.chat_options, ChatOptions)
                assert context.chat_options.temperature == 0.5

                # カスタムメタデータを追加。
                context.metadata["validated"] = True

                await next(context)

        middleware = ChatContextValidationMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions(temperature=0.5)
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            # メタデータがミドルウェアによって設定されたことを検証。
            assert ctx.metadata.get("validated") is True
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)
        assert result is not None


class TestStreamingScenarios:
    """ストリーミングおよび非ストリーミングシナリオのテストケース。"""

    async def test_streaming_flag_validation(self, mock_agent: AgentProtocol) -> None:
        """ストリーミング呼び出しでis_streamingフラグが正しく設定されることをテスト。"""
        streaming_flags: list[bool] = []

        class StreamingFlagMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                streaming_flags.append(context.is_streaming)
                await next(context)

        middleware = StreamingFlagMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]

        # 非ストリーミングのテスト。
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            streaming_flags.append(ctx.is_streaming)
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        await pipeline.execute(mock_agent, messages, context, final_handler)

        # ストリーミングのテスト。
        context_stream = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_stream_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            streaming_flags.append(ctx.is_streaming)
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk")])

        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context_stream, final_stream_handler):
            updates.append(update)

        # フラグを検証: [非ストリーミングミドルウェア、非ストリーミングハンドラー、ストリーミングミドルウェア、ストリーミングハンドラー]
        assert streaming_flags == [False, False, True, True]

    async def test_streaming_middleware_behavior(self, mock_agent: AgentProtocol) -> None:
        """ストリーミングレスポンスでのミドルウェアの動作をテスト。"""
        chunks_processed: list[str] = []

        class StreamProcessingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                chunks_processed.append("before_stream")
                await next(context)
                chunks_processed.append("after_stream")

        middleware = StreamProcessingMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_stream_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            chunks_processed.append("stream_start")
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk1")])
            chunks_processed.append("chunk1_yielded")
            yield AgentRunResponseUpdate(contents=[TextContent(text="chunk2")])
            chunks_processed.append("chunk2_yielded")
            chunks_processed.append("stream_end")

        updates: list[str] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_stream_handler):
            updates.append(update.text)

        assert updates == ["chunk1", "chunk2"]
        assert chunks_processed == [
            "before_stream",
            "after_stream",
            "stream_start",
            "chunk1_yielded",
            "chunk2_yielded",
            "stream_end",
        ]

    async def test_chat_streaming_flag_validation(self, mock_chat_client: Any) -> None:
        """チャットストリーミング呼び出しでis_streamingフラグが正しく設定されることをテスト。"""
        streaming_flags: list[bool] = []

        class ChatStreamingFlagMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                streaming_flags.append(context.is_streaming)
                await next(context)

        middleware = ChatStreamingFlagMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()

        # 非ストリーミングのテスト。
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            streaming_flags.append(ctx.is_streaming)
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response")])

        await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)

        # ストリーミングのテスト。
        context_stream = ChatContext(
            chat_client=mock_chat_client, messages=messages, chat_options=chat_options, is_streaming=True
        )

        async def final_stream_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            streaming_flags.append(ctx.is_streaming)
            yield ChatResponseUpdate(contents=[TextContent(text="chunk")])

        updates: list[ChatResponseUpdate] = []
        async for update in pipeline.execute_stream(
            mock_chat_client, messages, chat_options, context_stream, final_stream_handler
        ):
            updates.append(update)

        # フラグを検証: [非ストリーミングミドルウェア、非ストリーミングハンドラー、ストリーミングミドルウェア、ストリーミングハンドラー]
        assert streaming_flags == [False, False, True, True]

    async def test_chat_streaming_middleware_behavior(self, mock_chat_client: Any) -> None:
        """ストリーミングレスポンスでのチャットミドルウェアの動作をテスト。"""
        chunks_processed: list[str] = []

        class ChatStreamProcessingMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                chunks_processed.append("before_stream")
                await next(context)
                chunks_processed.append("after_stream")

        middleware = ChatStreamProcessingMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(
            chat_client=mock_chat_client, messages=messages, chat_options=chat_options, is_streaming=True
        )

        async def final_stream_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            chunks_processed.append("stream_start")
            yield ChatResponseUpdate(contents=[TextContent(text="chunk1")])
            chunks_processed.append("chunk1_yielded")
            yield ChatResponseUpdate(contents=[TextContent(text="chunk2")])
            chunks_processed.append("chunk2_yielded")
            chunks_processed.append("stream_end")

        updates: list[str] = []
        async for update in pipeline.execute_stream(
            mock_chat_client, messages, chat_options, context, final_stream_handler
        ):
            updates.append(update.text)

        assert updates == ["chunk1", "chunk2"]
        assert chunks_processed == [
            "before_stream",
            "after_stream",
            "stream_start",
            "chunk1_yielded",
            "chunk2_yielded",
            "stream_end",
        ]


# ヘルパークラスとフィクスチャの領域


class FunctionTestArgs(BaseModel):
    """関数ミドルウェアテストの引数。"""

    name: str = Field(description="Test name parameter")


class TestAgentMiddleware(AgentMiddleware):
    """AgentMiddlewareの実装テスト。"""

    async def process(self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]) -> None:
        await next(context)


class TestFunctionMiddleware(FunctionMiddleware):
    """FunctionMiddlewareの実装テスト。"""

    async def process(
        self, context: FunctionInvocationContext, next: Callable[[FunctionInvocationContext], Awaitable[None]]
    ) -> None:
        await next(context)


class TestChatMiddleware(ChatMiddleware):
    """ChatMiddlewareの実装テスト。"""

    async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
        await next(context)


class MockFunctionArgs(BaseModel):
    """関数ミドルウェアテストの引数。"""

    name: str = Field(description="Test name parameter")


class TestMiddlewareExecutionControl:
    """ミドルウェア実行制御のテストケース（next()が呼ばれた場合と呼ばれなかった場合）。"""

    async def test_agent_middleware_no_next_no_execution(self, mock_agent: AgentProtocol) -> None:
        """Agentミドルウェアがnext()を呼ばない場合、実行が起こらないことをテスト。"""

        class NoNextMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # next()を呼ばない - これにより実行が防止されるはず。
                pass

        middleware = NoNextMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        handler_called = False

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            nonlocal handler_called
            handler_called = True
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="should not execute")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        # 実行が起こらなかったことを検証 - 空のAgentRunResponseを返すべき。
        assert result is not None
        assert isinstance(result, AgentRunResponse)
        assert result.messages == []  # 空のレスポンス。
        assert not handler_called
        assert context.result is None

    async def test_agent_middleware_no_next_no_streaming_execution(self, mock_agent: AgentProtocol) -> None:
        """Agentミドルウェアがnext()を呼ばない場合、ストリーミング実行が起こらないことをテスト。"""

        class NoNextStreamingMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # next()を呼ばない - これにより実行が防止されるはず。
                pass

        middleware = NoNextStreamingMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        handler_called = False

        async def final_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            nonlocal handler_called
            handler_called = True
            yield AgentRunResponseUpdate(contents=[TextContent(text="should not execute")])

        # middlewareがnext()を呼び出さない場合、ストリーミングは更新を生成しないはずです
        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_handler):
            updates.append(update)

        # 実行が行われず、更新が生成されなかったことを検証します
        assert len(updates) == 0
        assert not handler_called
        assert context.result is None

    async def test_function_middleware_no_next_no_execution(self, mock_function: AIFunction[Any, Any]) -> None:
        """function middlewareがnext()を呼び出さない場合、実行が行われないことをテストします。"""

        class FunctionTestArgs(BaseModel):
            name: str = Field(description="Test name parameter")

        class NoNextFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                # next()を呼び出さない - これにより実行が防止されるはずです
                pass

        middleware = NoNextFunctionMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        handler_called = False

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            nonlocal handler_called
            handler_called = True
            return "should not execute"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        # 実行が行われなかったことを検証します
        assert result is None
        assert not handler_called
        assert context.result is None

    async def test_multiple_middlewares_early_stop(self, mock_agent: AgentProtocol) -> None:
        """最初のmiddlewareがnext()を呼び出さない場合、後続のmiddlewareが呼び出されないことをテストします。"""
        execution_order: list[str] = []

        class FirstMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("first")
                # next()を呼び出さない - これによりパイプラインが停止するはずです

        class SecondMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                execution_order.append("second")
                await next(context)

        pipeline = AgentMiddlewarePipeline([FirstMiddleware(), SecondMiddleware()])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        handler_called = False

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            nonlocal handler_called
            handler_called = True
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="should not execute")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        # 最初のmiddlewareのみが呼び出され、空のレスポンスが返されたことを検証します
        assert execution_order == ["first"]
        assert result is not None
        assert isinstance(result, AgentRunResponse)
        assert result.messages == []  # 空のレスポンス
        assert not handler_called

    async def test_chat_middleware_no_next_no_execution(self, mock_chat_client: Any) -> None:
        """chat middlewareがnext()を呼び出さない場合、実行が行われないことをテストします。"""

        class NoNextChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                # next()を呼び出さない - これにより実行が防止されるはずです
                pass

        middleware = NoNextChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        handler_called = False

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            nonlocal handler_called
            handler_called = True
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="should not execute")])

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)

        # 実行が行われなかったことを検証します
        assert result is None
        assert not handler_called
        assert context.result is None

    async def test_chat_middleware_no_next_no_streaming_execution(self, mock_chat_client: Any) -> None:
        """chat middlewareがnext()を呼び出さない場合、ストリーミング実行が行われないことをテストします。"""

        class NoNextStreamingChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                # next()を呼び出さない - これにより実行が防止されるはずです
                pass

        middleware = NoNextStreamingChatMiddleware()
        pipeline = ChatMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(
            chat_client=mock_chat_client, messages=messages, chat_options=chat_options, is_streaming=True
        )

        handler_called = False

        async def final_handler(ctx: ChatContext) -> AsyncIterable[ChatResponseUpdate]:
            nonlocal handler_called
            handler_called = True
            yield ChatResponseUpdate(contents=[TextContent(text="should not execute")])

        # middlewareがnext()を呼び出さない場合、ストリーミングは更新を生成しないはずです
        updates: list[ChatResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_chat_client, messages, chat_options, context, final_handler):
            updates.append(update)

        # 実行が行われず、更新が生成されなかったことを検証します
        assert len(updates) == 0
        assert not handler_called
        assert context.result is None

    async def test_multiple_chat_middlewares_early_stop(self, mock_chat_client: Any) -> None:
        """最初のchat middlewareがnext()を呼び出さない場合、後続のmiddlewareが呼び出されないことをテストします。"""
        execution_order: list[str] = []

        class FirstChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("first")
                # next()を呼び出さない - これによりパイプラインが停止するはずです

        class SecondChatMiddleware(ChatMiddleware):
            async def process(self, context: ChatContext, next: Callable[[ChatContext], Awaitable[None]]) -> None:
                execution_order.append("second")
                await next(context)

        pipeline = ChatMiddlewarePipeline([FirstChatMiddleware(), SecondChatMiddleware()])
        messages = [ChatMessage(role=Role.USER, text="test")]
        chat_options = ChatOptions()
        context = ChatContext(chat_client=mock_chat_client, messages=messages, chat_options=chat_options)

        handler_called = False

        async def final_handler(ctx: ChatContext) -> ChatResponse:
            nonlocal handler_called
            handler_called = True
            return ChatResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="should not execute")])

        result = await pipeline.execute(mock_chat_client, messages, chat_options, context, final_handler)

        # 最初のmiddlewareのみが呼び出され、結果が返されなかったことを検証します
        assert execution_order == ["first"]
        assert result is None
        assert not handler_called


@pytest.fixture
def mock_agent() -> AgentProtocol:
    """テスト用のMock agent。"""
    agent = MagicMock(spec=AgentProtocol)
    agent.name = "test_agent"
    return agent


@pytest.fixture
def mock_function() -> AIFunction[Any, Any]:
    """テスト用のMock function。"""
    function = MagicMock(spec=AIFunction[Any, Any])
    function.name = "test_function"
    return function


@pytest.fixture
def mock_chat_client() -> Any:
    """テスト用のMock chat client。"""
    from agent_framework._clients import ChatClientProtocol

    client = MagicMock(spec=ChatClientProtocol)
    client.service_url = MagicMock(return_value="mock://test")
    return client
