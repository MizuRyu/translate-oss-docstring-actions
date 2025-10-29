# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from agent_framework import (
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatClientProtocol,
    ChatMessage,
    Contents,
    Executor,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel

"""
Sample: Workflow as Agent with Reflection and Retry Pattern

Purpose:
This sample demonstrates how to wrap a workflow as an agent using WorkflowAgent.
It uses a reflection pattern where a Worker executor generates responses and a
Reviewer executor evaluates them. If the response is not approved, the Worker
regenerates the output based on feedback until the Reviewer approves it. Only
approved responses are emitted to the external consumer. The workflow completes when idle.

Key Concepts Demonstrated:
- WorkflowAgent: Wraps a workflow to behave like a regular agent.
- Cyclic workflow design (Worker ↔ Reviewer) for iterative improvement.
- AgentRunUpdateEvent: Mechanism for emitting approved responses externally.
- Structured output parsing for review feedback using Pydantic.
- State management for pending requests and retry logic.

Prerequisites:
- OpenAI account configured and accessible for OpenAIChatClient.
- Familiarity with WorkflowBuilder, Executor, WorkflowContext, and event handling.
- Understanding of how agent messages are generated, reviewed, and re-submitted.
"""


@dataclass
class ReviewRequest:
    """WorkerからReviewerへ評価のために渡される構造化リクエスト。"""

    request_id: str
    user_messages: list[ChatMessage]
    agent_messages: list[ChatMessage]


@dataclass
class ReviewResponse:
    """ReviewerからWorkerへ返される構造化レスポンス。"""

    request_id: str
    feedback: str
    approved: bool


class Reviewer(Executor):
    """Agentのレスポンスをレビューし、構造化フィードバックを提供するExecutor。"""

    def __init__(self, id: str, chat_client: ChatClientProtocol) -> None:
        super().__init__(id=id)
        self._chat_client = chat_client

    @handler
    async def review(self, request: ReviewRequest, ctx: WorkflowContext[ReviewResponse]) -> None:
        print(f"Reviewer: Evaluating response for request {request.request_id[:8]}...")

        # LLMが返す構造化スキーマを定義します。
        class _Response(BaseModel):
            feedback: str
            approved: bool

        # レビュー指示とコンテキストを構築します。
        messages = [
            ChatMessage(
                role=Role.SYSTEM,
                text=(
                    "You are a reviewer for an AI agent. Provide feedback on the "
                    "exchange between a user and the agent. Indicate approval only if:\n"
                    "- Relevance: response addresses the query\n"
                    "- Accuracy: information is correct\n"
                    "- Clarity: response is easy to understand\n"
                    "- Completeness: response covers all aspects\n"
                    "Do not approve until all criteria are satisfied."
                ),
            )
        ]
        # 会話履歴を追加します。
        messages.extend(request.user_messages)
        messages.extend(request.agent_messages)

        # 明示的なレビュー指示を追加します。
        messages.append(ChatMessage(role=Role.USER, text="Please review the agent's responses."))

        print("Reviewer: Sending review request to LLM...")
        response = await self._chat_client.get_response(messages=messages, response_format=_Response)

        parsed = _Response.model_validate_json(response.messages[-1].text)

        print(f"Reviewer: Review complete - Approved: {parsed.approved}")
        print(f"Reviewer: Feedback: {parsed.feedback}")

        # 構造化されたレビュー結果をWorkerに送信します。
        await ctx.send_message(
            ReviewResponse(request_id=request.request_id, feedback=parsed.feedback, approved=parsed.approved)
        )


class Worker(Executor):
    """レスポンスを生成し、必要に応じてフィードバックを取り込むExecutor。"""

    def __init__(self, id: str, chat_client: ChatClientProtocol) -> None:
        super().__init__(id=id)
        self._chat_client = chat_client
        self._pending_requests: dict[str, tuple[ReviewRequest, list[ChatMessage]]] = {}

    @handler
    async def handle_user_messages(self, user_messages: list[ChatMessage], ctx: WorkflowContext[ReviewRequest]) -> None:
        print("Worker: Received user messages, generating response...")

        # システムプロンプトでチャットを初期化します。
        messages = [ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant.")]
        messages.extend(user_messages)

        print("Worker: Calling LLM to generate response...")
        response = await self._chat_client.get_response(messages=messages)
        print(f"Worker: Response generated: {response.messages[-1].text}")

        # AgentのメッセージをContextに追加します。
        messages.extend(response.messages)

        # レビューリクエストを作成しReviewerに送信します。
        request = ReviewRequest(request_id=str(uuid4()), user_messages=user_messages, agent_messages=response.messages)
        print(f"Worker: Sending response for review (ID: {request.request_id[:8]})")
        await ctx.send_message(request)

        # 再試行の可能性のためにリクエストを追跡します。
        self._pending_requests[request.request_id] = (request, messages)

    @handler
    async def handle_review_response(self, review: ReviewResponse, ctx: WorkflowContext[ReviewRequest]) -> None:
        print(f"Worker: Received review for request {review.request_id[:8]} - Approved: {review.approved}")

        if review.request_id not in self._pending_requests:
            raise ValueError(f"Unknown request ID in review: {review.request_id}")

        request, messages = self._pending_requests.pop(review.request_id)

        if review.approved:
            print("Worker: Response approved. Emitting to external consumer...")
            contents: list[Contents] = []
            for message in request.agent_messages:
                contents.extend(message.contents)

            # AgentRunUpdateEventを通じて承認済み結果を外部コンシューマに送出します。
            await ctx.add_event(
                AgentRunUpdateEvent(self.id, data=AgentRunResponseUpdate(contents=contents, role=Role.ASSISTANT))
            )
            return

        print(f"Worker: Response not approved. Feedback: {review.feedback}")
        print("Worker: Regenerating response with feedback...")

        # レビューのフィードバックを取り込みます。
        messages.append(ChatMessage(role=Role.SYSTEM, text=review.feedback))
        messages.append(
            ChatMessage(role=Role.SYSTEM, text="Please incorporate the feedback and regenerate the response.")
        )
        messages.extend(request.user_messages)

        # 更新されたプロンプトで再試行します。
        response = await self._chat_client.get_response(messages=messages)
        print(f"Worker: New response generated: {response.messages[-1].text}")

        messages.extend(response.messages)

        # 再レビューのために更新されたリクエストを送信します。
        new_request = ReviewRequest(
            request_id=review.request_id, user_messages=request.user_messages, agent_messages=response.messages
        )
        await ctx.send_message(new_request)

        # さらなる評価のために新しいリクエストを追跡します。
        self._pending_requests[new_request.request_id] = (new_request, messages)


async def main() -> None:
    print("Starting Workflow Agent Demo")
    print("=" * 50)

    # チャットクライアントとExecutorを初期化します。
    print("Creating chat client and executors...")
    mini_chat_client = OpenAIChatClient(model_id="gpt-4.1-nano")
    chat_client = OpenAIChatClient(model_id="gpt-4.1")
    reviewer = Reviewer(id="reviewer", chat_client=chat_client)
    worker = Worker(id="worker", chat_client=mini_chat_client)

    print("Building workflow with Worker ↔ Reviewer cycle...")
    agent = (
        WorkflowBuilder()
        .add_edge(worker, reviewer)  # Worker sends responses to Reviewer
        .add_edge(reviewer, worker)  # Reviewer provides feedback to Worker
        .set_start_executor(worker)
        .build()
        .as_agent()  # Wrap workflow as an agent
    )

    print("Running workflow agent with user query...")
    print("Query: 'Write code for parallel reading 1 million files on disk and write to a sorted output file.'")
    print("-" * 50)

    # インクリメンタルな更新を観察するためにストリーミングモードでAgentを実行します。
    async for event in agent.run_stream(
        "Write code for parallel reading 1 million files on disk and write to a sorted output file."
    ):
        print(f"Agent Response: {event}")

    print("=" * 50)
    print("Workflow completed!")


if __name__ == "__main__":
    print("Initializing Workflow as Agent Sample...")
    asyncio.run(main())
