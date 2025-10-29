# Copyright (c) Microsoft. All rights reserved.

"""ストリーミングモードにおけるAgentExecutorのツール呼び出しと結果処理のテスト。"""

from collections.abc import AsyncIterable
from typing import Any

from agent_framework import (
    AgentExecutor,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    AgentThread,
    BaseAgent,
    ChatMessage,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    WorkflowBuilder,
)


class _ToolCallingAgent(BaseAgent):
    """ストリーミングモードでのツール呼び出しと結果をシミュレートするモックエージェント。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """非ストリーミング実行 - このテストでは使用しません。"""
        return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="done")])

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """ツール呼び出しと結果でストリーミングをシミュレートします。"""
        # 最初の更新：いくつかのテキスト
        yield AgentRunResponseUpdate(
            contents=[TextContent(text="Let me search for that...")],
            role=Role.ASSISTANT,
        )

        # 2回目の更新：ツール呼び出し（テキストなし！）
        yield AgentRunResponseUpdate(
            contents=[
                FunctionCallContent(
                    call_id="call_123",
                    name="search",
                    arguments={"query": "weather"},
                )
            ],
            role=Role.ASSISTANT,
        )

        # 3回目の更新：ツール結果（テキストなし！）
        yield AgentRunResponseUpdate(
            contents=[
                FunctionResultContent(
                    call_id="call_123",
                    result={"temperature": 72, "condition": "sunny"},
                )
            ],
            role=Role.TOOL,
        )

        # 4回目の更新：最終テキストレスポンス
        yield AgentRunResponseUpdate(
            contents=[TextContent(text="The weather is sunny, 72°F.")],
            role=Role.ASSISTANT,
        )


async def test_agent_executor_emits_tool_calls_in_streaming_mode() -> None:
    """AgentExecutorがFunctionCallContentとFunctionResultContentを含む更新を発行することをテストします。"""
    # 準備
    agent = _ToolCallingAgent(id="tool_agent", name="ToolAgent")
    agent_exec = AgentExecutor(agent, id="tool_exec")

    workflow = WorkflowBuilder().set_start_executor(agent_exec).build()

    # 実行：ストリーミングモードで実行
    events: list[AgentRunUpdateEvent] = []
    async for event in workflow.run_stream("What's the weather?"):
        if isinstance(event, AgentRunUpdateEvent):
            events.append(event)

    # 検証：4つのイベント（テキスト、関数呼び出し、関数結果、テキスト）を受け取るはずです
    assert len(events) == 4, f"Expected 4 events, got {len(events)}"

    # 最初のイベント：テキスト更新
    assert events[0].data is not None
    assert isinstance(events[0].data.contents[0], TextContent)
    assert "Let me search" in events[0].data.contents[0].text

    # 2番目のイベント：関数呼び出し
    assert events[1].data is not None
    assert isinstance(events[1].data.contents[0], FunctionCallContent)
    func_call = events[1].data.contents[0]
    assert func_call.call_id == "call_123"
    assert func_call.name == "search"

    # 3番目のイベント：関数結果
    assert events[2].data is not None
    assert isinstance(events[2].data.contents[0], FunctionResultContent)
    func_result = events[2].data.contents[0]
    assert func_result.call_id == "call_123"

    # 4番目のイベント：最終テキスト
    assert events[3].data is not None
    assert isinstance(events[3].data.contents[0], TextContent)
    assert "sunny" in events[3].data.contents[0].text
