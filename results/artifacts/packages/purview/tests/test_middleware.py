# Copyright (c) Microsoft. All rights reserved.

"""Purview middleware のテスト。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import AgentRunContext, AgentRunResponse, ChatMessage, Role
from azure.core.credentials import AccessToken

from agent_framework_purview import PurviewPolicyMiddleware, PurviewSettings


class TestPurviewPolicyMiddleware:
    """PurviewPolicyMiddleware の機能をテスト。"""

    @pytest.fixture
    def mock_credential(self) -> AsyncMock:
        """モックの非同期クレデンシャルを作成する。"""
        credential = AsyncMock()
        credential.get_token = AsyncMock(return_value=AccessToken("fake-token", 9999999999))
        return credential

    @pytest.fixture
    def settings(self) -> PurviewSettings:
        """テスト用の設定を作成する。"""
        return PurviewSettings(app_name="Test App", tenant_id="test-tenant")

    @pytest.fixture
    def middleware(self, mock_credential: AsyncMock, settings: PurviewSettings) -> PurviewPolicyMiddleware:
        """PurviewPolicyMiddleware インスタンスを作成する。"""
        return PurviewPolicyMiddleware(mock_credential, settings)

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """モックの Agent を作成する。"""
        agent = MagicMock()
        agent.name = "test-agent"
        return agent

    def test_middleware_initialization(self, mock_credential: AsyncMock, settings: PurviewSettings) -> None:
        """PurviewPolicyMiddleware の初期化をテスト。"""
        middleware = PurviewPolicyMiddleware(mock_credential, settings)

        assert middleware._client is not None
        assert middleware._processor is not None

    async def test_middleware_allows_clean_prompt(
        self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock
    ) -> None:
        """ポリシーチェックを通過するプロンプトをミドルウェアが許可することをテスト。"""
        context = AgentRunContext(agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Hello, how are you?")])

        with patch.object(middleware._processor, "process_messages", return_value=(False, "user-123")):
            next_called = False

            async def mock_next(ctx: AgentRunContext) -> None:
                nonlocal next_called
                next_called = True
                ctx.result = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="I'm good, thanks!")])

            await middleware.process(context, mock_next)

            assert next_called
            assert context.result is not None
            assert not context.terminate

    async def test_middleware_blocks_prompt_on_policy_violation(
        self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock
    ) -> None:
        """ポリシー違反のプロンプトをミドルウェアがブロックすることをテスト。"""
        context = AgentRunContext(
            agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Sensitive information")]
        )

        with patch.object(middleware._processor, "process_messages", return_value=(True, "user-123")):
            next_called = False

            async def mock_next(ctx: AgentRunContext) -> None:
                nonlocal next_called
                next_called = True

            await middleware.process(context, mock_next)

            assert not next_called
            assert context.result is not None
            assert context.terminate
            assert len(context.result.messages) == 1
            assert context.result.messages[0].role == Role.SYSTEM
            assert "blocked by policy" in context.result.messages[0].text.lower()

    async def test_middleware_checks_response(self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock) -> None:
        """ミドルウェアが Agent のレスポンスをポリシー違反についてチェックすることをテスト。"""
        context = AgentRunContext(agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Hello")])

        call_count = 0

        async def mock_process_messages(messages, activity, user_id=None):
            nonlocal call_count
            call_count += 1
            should_block = call_count != 1
            return (should_block, "user-123")

        with patch.object(middleware._processor, "process_messages", side_effect=mock_process_messages):

            async def mock_next(ctx: AgentRunContext) -> None:
                ctx.result = AgentRunResponse(
                    messages=[ChatMessage(role=Role.ASSISTANT, text="Here's some sensitive information")]
                )

            await middleware.process(context, mock_next)

            assert call_count == 2
            assert context.result is not None
            assert len(context.result.messages) == 1
            assert context.result.messages[0].role == Role.SYSTEM
            assert "blocked by policy" in context.result.messages[0].text.lower()

    async def test_middleware_handles_result_without_messages(
        self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock
    ) -> None:
        """messages 属性を持たない結果をミドルウェアが処理することをテスト。"""
        context = AgentRunContext(agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Hello")])

        with patch.object(middleware._processor, "process_messages", return_value=(False, "user-123")):

            async def mock_next(ctx: AgentRunContext) -> None:
                ctx.result = "Some non-standard result"

            await middleware.process(context, mock_next)

            assert context.result == "Some non-standard result"

    async def test_middleware_processor_receives_correct_activity(
        self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock
    ) -> None:
        """ミドルウェアが正しいアクティビティタイプをプロセッサに渡すことをテスト。"""
        from agent_framework_purview._models import Activity

        context = AgentRunContext(agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Test")])

        with patch.object(middleware._processor, "process_messages", return_value=(False, "user-123")) as mock_process:

            async def mock_next(ctx: AgentRunContext) -> None:
                ctx.result = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Response")])

            await middleware.process(context, mock_next)

            assert mock_process.call_count == 2
            for call in mock_process.call_args_list:
                assert call[0][1] == Activity.UPLOAD_TEXT

    async def test_middleware_handles_pre_check_exception(
        self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock
    ) -> None:
        """pre-check 内の例外がログに記録されるが処理を停止しないことをテスト。"""
        context = AgentRunContext(agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Test")])

        with patch.object(
            middleware._processor, "process_messages", side_effect=Exception("Pre-check error")
        ) as mock_process:

            async def mock_next(ctx: AgentRunContext) -> None:
                ctx.result = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Response")])

            await middleware.process(context, mock_next)

            # 2回呼ばれているはず（pre-check が例外を発生、その後 post-check も例外を発生）
            assert mock_process.call_count == 2
            # Context は終了されるべきでない
            assert not context.terminate
            # 結果は mock_next によって設定されるべき
            assert context.result is not None

    async def test_middleware_handles_post_check_exception(
        self, middleware: PurviewPolicyMiddleware, mock_agent: MagicMock
    ) -> None:
        """post-check 内の例外がログに記録されるが結果に影響しないことをテスト。"""
        context = AgentRunContext(agent=mock_agent, messages=[ChatMessage(role=Role.USER, text="Test")])

        call_count = 0

        async def mock_process_messages(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (False, "user-123")  # Pre-check が成功する
            raise Exception("Post-check error")  # Post-check が失敗する

        with patch.object(middleware._processor, "process_messages", side_effect=mock_process_messages):

            async def mock_next(ctx: AgentRunContext) -> None:
                ctx.result = AgentRunResponse(messages=[ChatMessage(role=Role.ASSISTANT, text="Response")])

            await middleware.process(context, mock_next)

            # 2回呼ばれているはず（pre と post）
            assert call_count == 2
            # 結果はまだ設定されているはず
            assert context.result is not None
            assert hasattr(context.result, "messages")
