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
    ChatAgent,
    ChatMessage,
    Role,
    TextContent,
)
from agent_framework._middleware import (
    AgentMiddleware,
    AgentMiddlewarePipeline,
    AgentRunContext,
    FunctionInvocationContext,
    FunctionMiddleware,
    FunctionMiddlewarePipeline,
)
from agent_framework._tools import AIFunction

from .conftest import MockChatClient


class FunctionTestArgs(BaseModel):
    """function middlewareテスト用のテスト引数。"""

    name: str = Field(description="Test name parameter")


class TestResultOverrideMiddleware:
    """middlewareのresult override機能のテストケース。"""

    async def test_agent_middleware_response_override_non_streaming(self, mock_agent: AgentProtocol) -> None:
        """agent middlewareが非ストリーミング実行のレスポンスをオーバーライドできることをテストします。"""
        override_response = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="overridden response")])

        class ResponseOverrideMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # まずパイプラインを実行し、その後レスポンスをオーバーライドします
                await next(context)
                context.result = override_response

        middleware = ResponseOverrideMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        handler_called = False

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            nonlocal handler_called
            handler_called = True
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="original response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        # オーバーライドされたレスポンスが返されることを検証します
        assert result is not None
        assert result == override_response
        assert result.messages[0].text == "overridden response"
        # middlewareがnext()を呼び出したため、元のハンドラーが呼び出されたことを検証します
        assert handler_called

    async def test_agent_middleware_response_override_streaming(self, mock_agent: AgentProtocol) -> None:
        """agent middlewareがストリーミング実行のレスポンスをオーバーライドできることをテストします。"""

        async def override_stream() -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(contents=[TextContent(text="overridden")])
            yield AgentRunResponseUpdate(contents=[TextContent(text=" stream")])

        class StreamResponseOverrideMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # まずパイプラインを実行し、その後レスポンスストリームをオーバーライドします
                await next(context)
                context.result = override_stream()

        middleware = StreamResponseOverrideMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(contents=[TextContent(text="original")])

        updates: list[AgentRunResponseUpdate] = []
        async for update in pipeline.execute_stream(mock_agent, messages, context, final_handler):
            updates.append(update)

        # オーバーライドされたレスポンスストリームが返されることを検証します
        assert len(updates) == 2
        assert updates[0].text == "overridden"
        assert updates[1].text == " stream"

    async def test_function_middleware_result_override(self, mock_function: AIFunction[Any, Any]) -> None:
        """function middlewareが結果をオーバーライドできることをテストします。"""
        override_result = "overridden function result"

        class ResultOverrideMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                # まずパイプラインを実行し、その後結果をオーバーライドします
                await next(context)
                context.result = override_result

        middleware = ResultOverrideMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        handler_called = False

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            nonlocal handler_called
            handler_called = True
            return "original function result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        # オーバーライドされた結果が返されることを検証します
        assert result == override_result
        # middlewareがnext()を呼び出したため、元のハンドラーが呼び出されたことを検証します
        assert handler_called

    async def test_chat_agent_middleware_response_override(self) -> None:
        """ChatAgent統合でのresult override機能をテストします。"""
        mock_chat_client = MockChatClient()

        class ChatAgentResponseOverrideMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # 常に最初にnext()を呼び出して実行を許可します
                await next(context)
                # その後、内容に基づいて条件付きでオーバーライドします
                if any("special" in msg.text for msg in context.messages if msg.text):
                    context.result = AgentRunResponse(
                        messages=[ChatMessage(role=Role.ASSISTANT, text="Special response from middleware!")]
                    )

        # override middlewareを使ってChatAgentを作成します
        middleware = ChatAgentResponseOverrideMiddleware()
        agent = ChatAgent(chat_client=mock_chat_client, middleware=[middleware])

        # オーバーライドケースをテストします
        override_messages = [ChatMessage(role=Role.USER, text="Give me a special response")]
        override_response = await agent.run(override_messages)
        assert override_response.messages[0].text == "Special response from middleware!"
        # middlewareがnext()を呼び出したため、chat clientが呼び出されたことを検証します
        assert mock_chat_client.call_count == 1

        # 通常ケースをテストします
        normal_messages = [ChatMessage(role=Role.USER, text="Normal request")]
        normal_response = await agent.run(normal_messages)
        assert normal_response.messages[0].text == "test response"
        # 通常ケースでchat clientが呼び出されたことを検証します
        assert mock_chat_client.call_count == 2

    async def test_chat_agent_middleware_streaming_override(self) -> None:
        """ChatAgent統合でのストリーミング結果オーバーライド機能をテストします。"""
        mock_chat_client = MockChatClient()

        async def custom_stream() -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(contents=[TextContent(text="Custom")])
            yield AgentRunResponseUpdate(contents=[TextContent(text=" streaming")])
            yield AgentRunResponseUpdate(contents=[TextContent(text=" response!")])

        class ChatAgentStreamOverrideMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # 常に最初にnext()を呼び出して実行を許可します
                await next(context)
                # その後、内容に基づいて条件付きでオーバーライドします
                if any("custom stream" in msg.text for msg in context.messages if msg.text):
                    context.result = custom_stream()

        # override middlewareを使ってChatAgentを作成します
        middleware = ChatAgentStreamOverrideMiddleware()
        agent = ChatAgent(chat_client=mock_chat_client, middleware=[middleware])

        # ストリーミングオーバーライドケースをテストします
        override_messages = [ChatMessage(role=Role.USER, text="Give me a custom stream")]
        override_updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream(override_messages):
            override_updates.append(update)

        assert len(override_updates) == 3
        assert override_updates[0].text == "Custom"
        assert override_updates[1].text == " streaming"
        assert override_updates[2].text == " response!"

        # 通常のストリーミングケースをテストします
        normal_messages = [ChatMessage(role=Role.USER, text="Normal streaming request")]
        normal_updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream(normal_messages):
            normal_updates.append(update)

        assert len(normal_updates) == 2
        assert normal_updates[0].text == "test streaming response "
        assert normal_updates[1].text == "another update"

    async def test_agent_middleware_conditional_no_next(self, mock_agent: AgentProtocol) -> None:
        """agent middlewareが条件付きでnext()を呼び出さない場合、実行が行われないことをテストします。"""

        class ConditionalNoNextMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # メッセージに"execute"が含まれる場合のみnext()を呼び出します
                if any("execute" in msg.text for msg in context.messages if msg.text):
                    await next(context)
                # それ以外の場合はnext()を呼び出さない - 実行は行われないはずです

        middleware = ConditionalNoNextMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])

        handler_called = False

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            nonlocal handler_called
            handler_called = True
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="executed response")])

        # next()が呼び出されないケースのテスト
        no_execute_messages = [ChatMessage(role=Role.USER, text="Don't run this")]
        no_execute_context = AgentRunContext(agent=mock_agent, messages=no_execute_messages)
        no_execute_result = await pipeline.execute(mock_agent, no_execute_messages, no_execute_context, final_handler)

        # middlewareがnext()を呼び出さない場合、結果は空のAgentRunResponseであるべきです
        assert no_execute_result is not None
        assert isinstance(no_execute_result, AgentRunResponse)
        assert no_execute_result.messages == []  # 空のレスポンス
        assert not handler_called
        assert no_execute_context.result is None

        # 次のテストのためにリセットします
        handler_called = False

        # next()が呼び出されるケースのテスト
        execute_messages = [ChatMessage(role=Role.USER, text="Please execute this")]
        execute_context = AgentRunContext(agent=mock_agent, messages=execute_messages)
        execute_result = await pipeline.execute(mock_agent, execute_messages, execute_context, final_handler)

        assert execute_result is not None
        assert execute_result.messages[0].text == "executed response"
        assert handler_called

    async def test_function_middleware_conditional_no_next(self, mock_function: AIFunction[Any, Any]) -> None:
        """function middlewareが条件付きでnext()を呼び出さない場合、実行が行われないことをテストします。"""

        class ConditionalNoNextFunctionMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                # 引数名に"execute"が含まれる場合のみnext()を呼び出します
                args = context.arguments
                assert isinstance(args, FunctionTestArgs)
                if "execute" in args.name:
                    await next(context)
                # それ以外の場合はnext()を呼び出さない - 実行は行われないはずです

        middleware = ConditionalNoNextFunctionMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])

        handler_called = False

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            nonlocal handler_called
            handler_called = True
            return "executed function result"

        # next()が呼び出されないケースのテスト
        no_execute_args = FunctionTestArgs(name="test_no_action")
        no_execute_context = FunctionInvocationContext(function=mock_function, arguments=no_execute_args)
        no_execute_result = await pipeline.execute(mock_function, no_execute_args, no_execute_context, final_handler)

        # middlewareがnext()を呼び出さない場合、functionの結果はNoneであるべきです（関数はNoneを返すことがあります）
        assert no_execute_result is None
        assert not handler_called
        assert no_execute_context.result is None

        # 次のテストのためにリセットします
        handler_called = False

        # next()が呼び出されるケースのテスト
        execute_args = FunctionTestArgs(name="test_execute")
        execute_context = FunctionInvocationContext(function=mock_function, arguments=execute_args)
        execute_result = await pipeline.execute(mock_function, execute_args, execute_context, final_handler)

        assert execute_result == "executed function result"
        assert handler_called


class TestResultObservability:
    """middlewareのresult observability機能のテストケース。"""

    async def test_agent_middleware_response_observability(self, mock_agent: AgentProtocol) -> None:
        """middlewareが実行後にレスポンスを観察できることをテストします。"""
        observed_responses: list[AgentRunResponse] = []

        class ObservabilityMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # next()前はContextは空であるべきです
                assert context.result is None

                # nextを呼び出して実行します
                await next(context)

                # 現在、Contextには観察用のレスポンスが含まれているはずです
                assert context.result is not None
                assert isinstance(context.result, AgentRunResponse)
                observed_responses.append(context.result)

        middleware = ObservabilityMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="executed response")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        # レスポンスが観察されたことを検証します
        assert len(observed_responses) == 1
        assert observed_responses[0].messages[0].text == "executed response"
        assert result == observed_responses[0]

    async def test_function_middleware_result_observability(self, mock_function: AIFunction[Any, Any]) -> None:
        """middlewareが実行後にfunction結果を観察できることをテストします。"""
        observed_results: list[str] = []

        class ObservabilityMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                # next()前はContextは空であるべきです
                assert context.result is None

                # nextを呼び出して実行します
                await next(context)

                # 現在、Contextには観察用の結果が含まれているはずです
                assert context.result is not None
                observed_results.append(context.result)

        middleware = ObservabilityMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            return "executed function result"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        # 結果が観察されたことを検証します
        assert len(observed_results) == 1
        assert observed_results[0] == "executed function result"
        assert result == observed_results[0]

    async def test_agent_middleware_post_execution_override(self, mock_agent: AgentProtocol) -> None:
        """middlewareが実行を観察した後にレスポンスをオーバーライドできることをテストします。"""

        class PostExecutionOverrideMiddleware(AgentMiddleware):
            async def process(
                self, context: AgentRunContext, next: Callable[[AgentRunContext], Awaitable[None]]
            ) -> None:
                # まずnextを呼び出して実行します
                await next(context)

                # 現在、観察して条件付きでオーバーライドします
                assert context.result is not None
                assert isinstance(context.result, AgentRunResponse)

                if "modify" in context.result.messages[0].text:
                    # 観察後にオーバーライドします
                    context.result = AgentRunResponse(
                        messages=[ChatMessage(role=Role.ASSISTANT, text="modified after execution")]
                    )

        middleware = PostExecutionOverrideMiddleware()
        pipeline = AgentMiddlewarePipeline([middleware])
        messages = [ChatMessage(role=Role.USER, text="test")]
        context = AgentRunContext(agent=mock_agent, messages=messages)

        async def final_handler(ctx: AgentRunContext) -> AgentRunResponse:
            return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="response to modify")])

        result = await pipeline.execute(mock_agent, messages, context, final_handler)

        # 実行後にレスポンスが修正されたことを検証します
        assert result is not None
        assert result.messages[0].text == "modified after execution"

    async def test_function_middleware_post_execution_override(self, mock_function: AIFunction[Any, Any]) -> None:
        """middlewareが実行を観察した後にfunction結果をオーバーライドできることをテストします。"""

        class PostExecutionOverrideMiddleware(FunctionMiddleware):
            async def process(
                self,
                context: FunctionInvocationContext,
                next: Callable[[FunctionInvocationContext], Awaitable[None]],
            ) -> None:
                # まずnextを呼び出して実行します
                await next(context)

                # 現在、観察して条件付きでオーバーライドします
                assert context.result is not None

                if "modify" in context.result:
                    # 観察後にオーバーライドします
                    context.result = "modified after execution"

        middleware = PostExecutionOverrideMiddleware()
        pipeline = FunctionMiddlewarePipeline([middleware])
        arguments = FunctionTestArgs(name="test")
        context = FunctionInvocationContext(function=mock_function, arguments=arguments)

        async def final_handler(ctx: FunctionInvocationContext) -> str:
            return "result to modify"

        result = await pipeline.execute(mock_function, arguments, context, final_handler)

        # 実行後に結果が修正されたことを検証します
        assert result == "modified after execution"


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
