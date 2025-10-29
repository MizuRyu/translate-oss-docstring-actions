# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable
from typing import Any

from pydantic import PrivateAttr
from typing_extensions import Never

from agent_framework import (
    AgentExecutor,
    AgentExecutorResponse,
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Executor,
    Role,
    SequentialBuilder,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowRunState,
    WorkflowStatusEvent,
    handler,
)


class _SimpleAgent(BaseAgent):
    """単一の assistant メッセージを返す Agent（非ストリーミングパス）。"""

    def __init__(self, *, reply_text: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reply_text = reply_text

    async def run(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=self._reply_text)])

    async def run_stream(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        # この Agent はストリーミングをサポートしない。単一の完全なレスポンスを yield する。
        yield AgentRunResponseUpdate(contents=[TextContent(text=self._reply_text)])


class _CaptureFullConversation(Executor):
    """AgentExecutorResponse.full_conversation をキャプチャし、ワークフローを完了させる。"""

    @handler
    async def capture(self, response: AgentExecutorResponse, ctx: WorkflowContext[Never, dict]) -> None:
        full = response.full_conversation
        # AgentExecutor の契約により full_conversation が必ず設定されることを保証。
        assert full is not None
        payload = {
            "length": len(full),
            "roles": [m.role for m in full],
            "texts": [m.text for m in full],
        }
        await ctx.yield_output(payload)
        pass


async def test_agent_executor_populates_full_conversation_non_streaming() -> None:
    # 準備: workflow.run() 使用時に AgentExecutor が非ストリーミングになる。
    agent = _SimpleAgent(id="agent1", name="A", reply_text="agent-reply")
    agent_exec = AgentExecutor(agent, id="agent1-exec")
    capturer = _CaptureFullConversation(id="capture")

    wf = WorkflowBuilder().set_start_executor(agent_exec).add_edge(agent_exec, capturer).build()

    # 実行: 非ストリーミングモードをテストするために run_stream() ではなく run() を使用。
    result = await wf.run("hello world")

    # run の結果から出力を抽出。
    outputs = result.get_outputs()
    assert len(outputs) == 1
    payload = outputs[0]

    # 検証: full_conversation に [user("hello world"), assistant("agent-reply")] が含まれていること。
    assert isinstance(payload, dict)
    assert payload["length"] == 2
    assert payload["roles"][0] == Role.USER and "hello world" in (payload["texts"][0] or "")
    assert payload["roles"][1] == Role.ASSISTANT and "agent-reply" in (payload["texts"][1] or "")


class _CaptureAgent(BaseAgent):
    """受信したメッセージを記録するストリーミング対応 Agent。"""

    _last_messages: list[ChatMessage] = PrivateAttr(default_factory=list)  # type: ignore

    def __init__(self, *, reply_text: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reply_text = reply_text

    async def run(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        # 非ストリーミング実行時に検証用にメッセージを正規化して記録。
        norm: list[ChatMessage] = []
        if messages:
            for m in messages:  # type: ignore[iteration-over-optional]
                if isinstance(m, ChatMessage):
                    norm.append(m)
                elif isinstance(m, str):
                    norm.append(ChatMessage(role=Role.USER, text=m))
        self._last_messages = norm
        return AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text=self._reply_text)])

    async def run_stream(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        # ストリーミング実行時に検証用にメッセージを正規化して記録。
        norm: list[ChatMessage] = []
        if messages:
            for m in messages:  # type: ignore[iteration-over-optional]
                if isinstance(m, ChatMessage):
                    norm.append(m)
                elif isinstance(m, str):
                    norm.append(ChatMessage(role=Role.USER, text=m))
        self._last_messages = norm
        yield AgentRunResponseUpdate(contents=[TextContent(text=self._reply_text)])


async def test_sequential_adapter_uses_full_conversation() -> None:
    # 準備: 2つのストリーミング Agent。2番目は受信したものを記録する。
    a1 = _CaptureAgent(id="agent1", name="A1", reply_text="A1 reply")
    a2 = _CaptureAgent(id="agent2", name="A2", reply_text="A2 reply")

    wf = SequentialBuilder().participants([a1, a2]).build()

    # 実行。
    async for ev in wf.run_stream("hello seq"):
        if isinstance(ev, WorkflowStatusEvent) and ev.state == WorkflowRunState.IDLE:
            break

    # 検証: 2番目の Agent はユーザープロンプトと A1 の assistant の返信を見ているはず。
    seen = a2._last_messages  # pyright: ignore[reportPrivateUsage]
    assert len(seen) == 2
    assert seen[0].role == Role.USER and "hello seq" in (seen[0].text or "")
    assert seen[1].role == Role.ASSISTANT and "A1 reply" in (seen[1].text or "")
