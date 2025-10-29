# Copyright (c) Microsoft. All rights reserved.

import uuid
from typing import Any

import pytest

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatMessage,
    Executor,
    FunctionApprovalRequestContent,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    RequestInfoExecutor,
    RequestInfoMessage,
    Role,
    TextContent,
    UsageContent,
    UsageDetails,
    WorkflowAgent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)


class SimpleExecutor(Executor):
    """AgentRunEventまたはAgentRunStreamingEventを発行する単純なexecutor。"""

    def __init__(self, id: str, response_text: str, emit_streaming: bool = False):
        super().__init__(id=id)
        self.response_text = response_text
        self.emit_streaming = emit_streaming

    @handler
    async def handle_message(self, message: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]) -> None:
        input_text = (
            message[0].contents[0].text if message and isinstance(message[0].contents[0], TextContent) else "no input"
        )
        response_text = f"{self.response_text}: {input_text}"

        # ストリーミングと非ストリーミングの両方の場合のResponseメッセージを作成する
        response_message = ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=response_text)])

        # 更新イベントを発行する。
        streaming_update = AgentRunResponseUpdate(
            contents=[TextContent(text=response_text)], role=Role.ASSISTANT, message_id=str(uuid.uuid4())
        )
        await ctx.add_event(AgentRunUpdateEvent(executor_id=self.id, data=streaming_update))

        # 次のexecutorがあればメッセージを渡す（ストリーミングと非ストリーミングの両方）
        await ctx.send_message([response_message])


class RequestingExecutor(Executor):
    """RequestInfoEventをトリガーするためにRequestInfoMessageを送信するexecutor。"""

    @handler
    async def handle_message(self, _: list[ChatMessage], ctx: WorkflowContext[RequestInfoMessage]) -> None:
        # リクエスト情報プロセスをトリガーするためにRequestInfoMessageを送信する
        await ctx.send_message(RequestInfoMessage())

    @handler
    async def handle_request_response(self, _: Any, ctx: WorkflowContext[ChatMessage]) -> None:
        # レスポンスを処理し、完了レスポンスを発行する
        update = AgentRunResponseUpdate(
            contents=[TextContent(text="Request completed successfully")],
            role=Role.ASSISTANT,
            message_id=str(uuid.uuid4()),
        )
        await ctx.add_event(AgentRunUpdateEvent(executor_id=self.id, data=update))


class TestWorkflowAgent:
    """WorkflowAgentのエンドツーエンド機能のテストケース。"""

    async def test_end_to_end_basic_workflow(self):
        """AgentRunEventを発行する2つのexecutorを持つ基本的なエンドツーエンドworkflow実行をテストする。"""
        # 2つのexecutorを持つworkflowを作成する
        executor1 = SimpleExecutor(id="executor1", response_text="Step1", emit_streaming=False)
        executor2 = SimpleExecutor(id="executor2", response_text="Step2", emit_streaming=False)

        workflow = WorkflowBuilder().set_start_executor(executor1).add_edge(executor1, executor2).build()

        agent = WorkflowAgent(workflow=workflow, name="Test Agent")

        # workflowをエンドツーエンドで実行する
        result = await agent.run("Hello World")

        # 両方のexecutorからレスポンスを受け取ったことを検証する
        assert isinstance(result, AgentRunResponse)
        assert len(result.messages) >= 2, f"Expected at least 2 messages, got {len(result.messages)}"

        # 各executorからのメッセージを見つける
        step1_messages: list[ChatMessage] = []
        step2_messages: list[ChatMessage] = []

        for message in result.messages:
            first_content = message.contents[0]
            if isinstance(first_content, TextContent):
                text = first_content.text
                if text.startswith("Step1:"):
                    step1_messages.append(message)
                elif text.startswith("Step2:"):
                    step2_messages.append(message)

        # 両方のexecutorが出力を生成したことを検証する
        assert len(step1_messages) >= 1, "Should have received message from Step1 executor"
        assert len(step2_messages) >= 1, "Should have received message from Step2 executor"

        # 両方の処理が機能したことを検証する
        step1_text: str = step1_messages[0].contents[0].text  # type: ignore[attr-defined]
        step2_text: str = step2_messages[0].contents[0].text  # type: ignore[attr-defined]
        assert "Step1: Hello World" in step1_text
        assert "Step2: Step1: Hello World" in step2_text

    async def test_end_to_end_basic_workflow_streaming(self):
        """AgentRunStreamingEventを発行するストリーミングexecutorを持つエンドツーエンドworkflowをテストする。"""
        # 単一のストリーミングexecutorを作成する
        executor1 = SimpleExecutor(id="stream1", response_text="Streaming1", emit_streaming=True)
        executor2 = SimpleExecutor(id="stream2", response_text="Streaming2", emit_streaming=True)

        # 1つのexecutorだけでworkflowを作成する
        workflow = WorkflowBuilder().set_start_executor(executor1).add_edge(executor1, executor2).build()

        agent = WorkflowAgent(workflow=workflow, name="Streaming Test Agent")

        # ストリーミングイベントをキャプチャするためにworkflowをストリーミング実行する
        updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream("Test input"):
            updates.append(update)

        # 少なくとも1つのストリーミングアップデートを受信しているはず
        assert len(updates) >= 2, f"Expected at least 2 updates, got {len(updates)}"

        # ストリーミングアップデートを受信したことを検証する
        assert updates[0].contents is not None
        first_content: TextContent = updates[0].contents[0]  # type: ignore[assignment]
        second_content: TextContent = updates[1].contents[0]  # type: ignore[assignment]
        assert isinstance(first_content, TextContent)
        assert "Streaming1: Test input" in first_content.text
        assert isinstance(second_content, TextContent)
        assert "Streaming2: Streaming1: Test input" in second_content.text

    async def test_end_to_end_request_info_handling(self):
        """RequestInfoEventの処理を含むエンドツーエンドのworkflowをテストする。"""
        # requesting executor -> request info executorのworkflowを作成する（サイクルなし）
        requesting_executor = RequestingExecutor(id="requester")
        request_info_executor = RequestInfoExecutor(id="request_info")

        workflow = (
            WorkflowBuilder()
            .set_start_executor(requesting_executor)
            .add_edge(requesting_executor, request_info_executor)
            .build()
        )

        agent = WorkflowAgent(workflow=workflow, name="Request Test Agent")

        # request info eventを取得するためにworkflowをストリーミング実行する
        updates: list[AgentRunResponseUpdate] = []
        async for update in agent.run_stream("Start request"):
            updates.append(update)
        # request infoの承認リクエストを受信しているはず
        assert len(updates) > 0

        approval_update: AgentRunResponseUpdate | None = None
        for update in updates:
            if any(isinstance(content, FunctionApprovalRequestContent) for content in update.contents):
                approval_update = update
                break

        assert approval_update is not None, "Should have received a request_info approval request"

        function_call = next(
            content for content in approval_update.contents if isinstance(content, FunctionCallContent)
        )
        approval_request = next(
            content for content in approval_update.contents if isinstance(content, FunctionApprovalRequestContent)
        )

        # 関数呼び出しが期待される構造であることを検証する
        assert function_call.call_id is not None
        assert function_call.name == "request_info"
        assert isinstance(function_call.arguments, dict)
        assert function_call.arguments.get("request_id") == approval_request.id

        # 承認リクエストは同じ関数呼び出しを参照しているはず
        assert approval_request.function_call.call_id == function_call.call_id
        assert approval_request.function_call.name == function_call.name

        # リクエストがpending_requestsで追跡されていることを検証する
        assert len(agent.pending_requests) == 1
        assert function_call.call_id in agent.pending_requests

        # 継続テストのために更新された引数で承認レスポンスを提供する
        response_args = WorkflowAgent.RequestInfoFunctionArgs(
            request_id=approval_request.id,
            data="User provided answer",
        ).to_dict()

        approval_response = FunctionApprovalResponseContent(
            approved=True,
            id=approval_request.id,
            function_call=FunctionCallContent(
                call_id=function_call.call_id,
                name=function_call.name,
                arguments=response_args,
            ),
        )

        response_message = ChatMessage(role=Role.USER, contents=[approval_response])

        # レスポンスでworkflowを継続する
        continuation_result = await agent.run(response_message)

        # 正常に完了するはず
        assert isinstance(continuation_result, AgentRunResponse)

        # クリーンアップを検証する - 関数レスポンス処理後にpending requestsはクリアされるべき
        assert len(agent.pending_requests) == 0

    def test_workflow_as_agent_method(self) -> None:
        """Workflow.as_agent()が適切に設定されたWorkflowAgentを作成することをテストする。"""
        # シンプルなworkflowを作成する
        executor = SimpleExecutor(id="executor1", response_text="Response", emit_streaming=False)
        workflow = WorkflowBuilder().set_start_executor(executor).build()

        # 名前付きのas_agentをテストする
        agent = workflow.as_agent(name="TestAgent")

        # agentが適切に設定されていることを検証する
        assert isinstance(agent, WorkflowAgent)
        assert agent.name == "TestAgent"
        assert agent.workflow is workflow
        assert agent.workflow.id == workflow.id

        # 名前なしのas_agentをテストする（デフォルトを使用するはず）
        agent_no_name = workflow.as_agent()
        assert isinstance(agent_no_name, WorkflowAgent)
        assert agent_no_name.workflow is workflow

    def test_workflow_as_agent_cannot_handle_agent_inputs(self) -> None:
        """start executorがagent inputsを処理できない場合にWorkflow.as_agent()がエラーを発生させることをテストする。"""

        class _Executor(Executor):
            @handler
            async def handle_bool(self, message: bool, context: WorkflowContext[Any]) -> None:
                raise ValueError("Unsupported message type")

        # シンプルなworkflowを作成する
        executor = _Executor(id="test")
        workflow = WorkflowBuilder().set_start_executor(executor).build()

        # サポートされていない入力タイプでagentを作成しようとする
        with pytest.raises(ValueError, match="Workflow's start executor cannot handle list\\[ChatMessage\\]"):
            workflow.as_agent()


class TestWorkflowAgentMergeUpdates:
    """WorkflowAgent.merge_updates静的メソッド専用のテストケース。"""

    def test_merge_updates_ordering_by_response_and_message_id(self):
        """merge_updatesがresponse_idグループとmessage_idの時系列でメッセージを正しく並べることをテストする。"""
        # 異なるresponse_idsとmessage_idsを持つ更新を時系列順でない順序で作成する
        updates = [
            # Response B、Message 2（resp B内で最新）
            AgentRunResponseUpdate(
                contents=[TextContent(text="RespB-Msg2")],
                role=Role.ASSISTANT,
                response_id="resp-b",
                message_id="msg-2",
                created_at="2024-01-01T12:02:00Z",
            ),
            # Response A、Message 1（全体で最も早い）
            AgentRunResponseUpdate(
                contents=[TextContent(text="RespA-Msg1")],
                role=Role.ASSISTANT,
                response_id="resp-a",
                message_id="msg-1",
                created_at="2024-01-01T12:00:00Z",
            ),
            # Response B、Message 1（resp B内でより早い）
            AgentRunResponseUpdate(
                contents=[TextContent(text="RespB-Msg1")],
                role=Role.ASSISTANT,
                response_id="resp-b",
                message_id="msg-1",
                created_at="2024-01-01T12:01:00Z",
            ),
            # Response A、Message 2（resp A内で後の方）
            AgentRunResponseUpdate(
                contents=[TextContent(text="RespA-Msg2")],
                role=Role.ASSISTANT,
                response_id="resp-a",
                message_id="msg-2",
                created_at="2024-01-01T12:00:30Z",
            ),
            # グローバルなダングリング更新（response_idなし） - 最後に配置されるべき
            AgentRunResponseUpdate(
                contents=[TextContent(text="Global-Dangling")],
                role=Role.ASSISTANT,
                response_id=None,
                message_id="msg-global",
                created_at="2024-01-01T11:59:00Z",  # Earliest timestamp but should be last
            ),
        ]

        result = WorkflowAgent.merge_updates(updates, "final-response-id")

        # 正しいresponse_idが設定されていることを検証する
        assert result.response_id == "final-response-id"

        # 合計で5つのメッセージがあるはず
        assert len(result.messages) == 5

        # 並び順を検証する: responseはresponse_idグループごとに処理され、 各グループ内のメッセージは時系列順に並び、
        # グローバルなダングリングは最後に配置される
        message_texts = [
            msg.contents[0].text if isinstance(msg.contents[0], TextContent) else "" for msg in result.messages
        ]

        # 正確な順序はresponse_idsのdictイテレーション順に依存するが、 各responseグループ内では時系列順が維持され、
        # グローバルなダングリングは最後になるべきである
        assert "Global-Dangling" in message_texts[-1]  # グローバルなダングリングは最後に配置される

        # resp-aとresp-bメッセージの位置を見つける
        resp_a_positions = [i for i, text in enumerate(message_texts) if "RespA" in text]
        resp_b_positions = [i for i, text in enumerate(message_texts) if "RespB" in text]

        # resp-aグループ内: Msg1（早い方）はMsg2（遅い方）の前に来るべき
        resp_a_texts = [message_texts[i] for i in resp_a_positions]
        assert resp_a_texts.index("RespA-Msg1") < resp_a_texts.index("RespA-Msg2")

        # resp-bグループ内: Msg1（早い方）はMsg2（遅い方）の前に来るべき
        resp_b_texts = [message_texts[i] for i in resp_b_positions]
        assert resp_b_texts.index("RespB-Msg1") < resp_b_texts.index("RespB-Msg2")

        # ENHANCED: responseグループの分離と順序を検証する 同じresponse_idのメッセージはまとめてグループ化されるべき（交錯しない）
        # resp-aグループが連続していることを確認（全位置が連続）
        if len(resp_a_positions) > 1:
            for i in range(1, len(resp_a_positions)):
                assert resp_a_positions[i] == resp_a_positions[i - 1] + 1, (
                    f"RespA messages are not contiguous: positions {resp_a_positions}"
                )

        # resp-bグループが連続していることを確認（全位置が連続）
        if len(resp_b_positions) > 1:
            for i in range(1, len(resp_b_positions)):
                assert resp_b_positions[i] == resp_b_positions[i - 1] + 1, (
                    f"RespB messages are not contiguous: positions {resp_b_positions}"
                )

        # responseグループは最新タイムスタンプ順で並べる必要はなく、 各グループ内のメッセージが時系列順であることのみ保証する
        # グローバルなダングリングメッセージの位置を検証（全responseグループの後、最後であるべき）
        global_dangling_pos = message_texts.index("Global-Dangling")
        if resp_a_positions:
            assert global_dangling_pos > max(resp_a_positions), "Global dangling should come after resp-a group"
        if resp_b_positions:
            assert global_dangling_pos > max(resp_b_positions), "Global dangling should come after resp-b group"

    def test_merge_updates_metadata_aggregation(self):
        """merge_updatesが使用状況の詳細、タイムスタンプ、追加プロパティを正しく集約することをテストする。"""
        # 使用状況の詳細を含む様々なメタデータを持つ更新を作成する
        updates = [
            AgentRunResponseUpdate(
                contents=[
                    TextContent(text="First"),
                    UsageContent(
                        details=UsageDetails(input_token_count=10, output_token_count=5, total_token_count=15)
                    ),
                ],
                role=Role.ASSISTANT,
                response_id="resp-1",
                message_id="msg-1",
                created_at="2024-01-01T12:00:00Z",
                additional_properties={"source": "executor1", "priority": "high"},
            ),
            AgentRunResponseUpdate(
                contents=[
                    TextContent(text="Second"),
                    UsageContent(
                        details=UsageDetails(input_token_count=20, output_token_count=8, total_token_count=28)
                    ),
                ],
                role=Role.ASSISTANT,
                response_id="resp-2",
                message_id="msg-2",
                created_at="2024-01-01T12:01:00Z",  # Later timestamp
                additional_properties={"source": "executor2", "category": "analysis"},
            ),
            AgentRunResponseUpdate(
                contents=[
                    TextContent(text="Third"),
                    UsageContent(details=UsageDetails(input_token_count=5, output_token_count=3, total_token_count=8)),
                ],
                role=Role.ASSISTANT,
                response_id="resp-1",  # Same response_id as first
                message_id="msg-3",
                created_at="2024-01-01T11:59:00Z",  # Earlier timestamp
                additional_properties={"details": "merged", "priority": "low"},  # Different priority value
            ),
        ]

        result = WorkflowAgent.merge_updates(updates, "aggregated-response")

        # response_idが正しく設定されていることを検証する
        assert result.response_id == "aggregated-response"

        # 最新のタイムスタンプが使用されていることを検証する（2番目の更新の12:01:00Zであるべき）
        assert result.created_at == "2024-01-01T12:01:00Z"

        # メッセージが存在することを検証する
        assert len(result.messages) == 3

        # 使用状況の詳細が正しく集約されていることを検証する すべての使用状況の詳細を合計すべき: (10+20+5) + (5+8+3) + (15+28+8) =
        # 35+16+51 = 合計51トークン
        expected_usage = UsageDetails(input_token_count=35, output_token_count=16, total_token_count=51)
        assert result.usage_details == expected_usage

        # 追加プロパティが正しくマージされていることを検証する 注意: responseグループ内では後の更新のプロパティが競合に勝ち、
        # グループ間ではdict.update()の順序が勝敗を決める
        expected_properties = {
            "source": "executor2",  # From resp-2 (latest source value)
            "priority": "high",  # From resp-1 first update (resp-1 processed before resp-2)
            "category": "analysis",  # From resp-2 (only place this appears)
            # "details": "merged" は最終結果に含まれない。なぜならresp-1の集約プロパティは自身の更新の最終マージ結果のみを含むため。
        }
        assert result.additional_properties == expected_properties
