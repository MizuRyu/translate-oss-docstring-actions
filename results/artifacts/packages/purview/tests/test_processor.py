# Copyright (c) Microsoft. All rights reserved.

"""Purview processor のテスト。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role

from agent_framework_purview import PurviewAppLocation, PurviewLocationType, PurviewSettings
from agent_framework_purview._models import (
    Activity,
    DlpAction,
    DlpActionInfo,
    ProcessContentResponse,
    RestrictionAction,
)
from agent_framework_purview._processor import ScopedContentProcessor, _is_valid_guid


class TestGuidValidation:
    """GUID 検証ヘルパーをテスト。"""

    def test_valid_guid(self) -> None:
        """有効な GUID で _is_valid_guid をテスト。"""
        assert _is_valid_guid("12345678-1234-1234-1234-123456789012")
        assert _is_valid_guid("a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d")

    def test_invalid_guid(self) -> None:
        """無効な GUID で _is_valid_guid をテスト。"""
        assert not _is_valid_guid("not-a-guid")
        assert not _is_valid_guid("")
        assert not _is_valid_guid(None)


class TestScopedContentProcessor:
    """ScopedContentProcessor の機能をテスト。"""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """モックの Purview client を作成する。"""
        client = AsyncMock()
        client.get_user_info_from_token = AsyncMock(
            return_value={
                "tenant_id": "12345678-1234-1234-1234-123456789012",
                "user_id": "12345678-1234-1234-1234-123456789012",
                "client_id": "12345678-1234-1234-1234-123456789012",
            }
        )
        return client

    @pytest.fixture
    def settings_with_defaults(self) -> PurviewSettings:
        """デフォルト値を持つ設定を作成する。"""
        app_location = PurviewAppLocation(
            location_type=PurviewLocationType.APPLICATION, location_value="12345678-1234-1234-1234-123456789012"
        )
        return PurviewSettings(
            app_name="Test App",
            tenant_id="12345678-1234-1234-1234-123456789012",
            purview_app_location=app_location,
        )

    @pytest.fixture
    def settings_without_defaults(self) -> PurviewSettings:
        """デフォルト値を持たない設定を作成する（トークン情報が必要）。"""
        return PurviewSettings(app_name="Test App")

    @pytest.fixture
    def processor(self, mock_client: AsyncMock, settings_with_defaults: PurviewSettings) -> ScopedContentProcessor:
        """モッククライアントを使って ScopedContentProcessor を作成する。"""
        return ScopedContentProcessor(mock_client, settings_with_defaults)

    async def test_processor_initialization(
        self, mock_client: AsyncMock, settings_with_defaults: PurviewSettings
    ) -> None:
        """ScopedContentProcessor の初期化をテスト。"""
        processor = ScopedContentProcessor(mock_client, settings_with_defaults)

        assert processor._client == mock_client
        assert processor._settings == settings_with_defaults

    async def test_process_messages_with_defaults(self, processor: ScopedContentProcessor) -> None:
        """デフォルトを持つ設定で process_messages をテスト。"""
        messages = [
            ChatMessage(role=Role.USER, text="Hello"),
            ChatMessage(role=Role.ASSISTANT, text="Hi there"),
        ]

        with patch.object(processor, "_map_messages", return_value=([], None)) as mock_map:
            should_block, user_id = await processor.process_messages(messages, Activity.UPLOAD_TEXT)

            assert should_block is False
            assert user_id is None
            mock_map.assert_called_once_with(messages, Activity.UPLOAD_TEXT, None)

    async def test_process_messages_blocks_content(
        self, processor: ScopedContentProcessor, process_content_request_factory
    ) -> None:
        """コンテンツがブロックされるべき場合に process_messages が True を返すことをテスト。"""
        messages = [ChatMessage(role=Role.USER, text="Sensitive content")]

        mock_request = process_content_request_factory("Sensitive content")

        mock_response = ProcessContentResponse(**{
            "policyActions": [DlpActionInfo(action=DlpAction.BLOCK_ACCESS, restrictionAction=RestrictionAction.BLOCK)]
        })

        with (
            patch.object(processor, "_map_messages", return_value=([mock_request], "user-123")),
            patch.object(processor, "_process_with_scopes", return_value=mock_response),
        ):
            should_block, user_id = await processor.process_messages(messages, Activity.UPLOAD_TEXT)

            assert should_block is True
            assert user_id == "user-123"

    async def test_map_messages_creates_requests(
        self, processor: ScopedContentProcessor, mock_client: AsyncMock
    ) -> None:
        """_map_messages が ProcessContentRequest オブジェクトを作成することをテスト。"""
        messages = [
            ChatMessage(
                role=Role.USER,
                text="Test message",
                message_id="msg-123",
                author_name="12345678-1234-1234-1234-123456789012",
            ),
        ]

        requests, user_id = await processor._map_messages(messages, Activity.UPLOAD_TEXT)

        assert len(requests) == 1
        assert requests[0].user_id == "12345678-1234-1234-1234-123456789012"
        assert requests[0].tenant_id == "12345678-1234-1234-1234-123456789012"
        assert user_id == "12345678-1234-1234-1234-123456789012"

    async def test_map_messages_without_defaults_gets_token_info(self, mock_client: AsyncMock) -> None:
        """設定に一部のデフォルトがない場合に _map_messages がトークン情報を取得することをテスト。"""
        settings = PurviewSettings(app_name="Test App", tenant_id="12345678-1234-1234-1234-123456789012")
        processor = ScopedContentProcessor(mock_client, settings)
        messages = [ChatMessage(role=Role.USER, text="Test", message_id="msg-123")]

        requests, user_id = await processor._map_messages(messages, Activity.UPLOAD_TEXT)

        mock_client.get_user_info_from_token.assert_called_once()
        assert len(requests) == 1
        assert user_id is not None

    async def test_map_messages_raises_on_missing_tenant_id(self, mock_client: AsyncMock) -> None:
        """tenant_id が判別できない場合に _map_messages が ValueError を発生させることをテスト。"""
        settings = PurviewSettings(app_name="Test App")  # tenant_id がない
        processor = ScopedContentProcessor(mock_client, settings)

        mock_client.get_user_info_from_token = AsyncMock(
            return_value={"user_id": "test-user", "client_id": "test-client"}
        )

        messages = [ChatMessage(role=Role.USER, text="Test", message_id="msg-123")]

        with pytest.raises(ValueError, match="Tenant id required"):
            await processor._map_messages(messages, Activity.UPLOAD_TEXT)

    async def test_check_applicable_scopes_no_scopes(
        self, processor: ScopedContentProcessor, process_content_request_factory
    ) -> None:
        """スコープが返されない場合の _check_applicable_scopes をテスト。"""
        from agent_framework_purview._models import ProtectionScopesResponse

        request = process_content_request_factory()
        response = ProtectionScopesResponse(**{"value": None})

        should_process, actions = processor._check_applicable_scopes(request, response)

        assert should_process is False
        assert actions == []

    async def test_check_applicable_scopes_with_block_action(
        self, processor: ScopedContentProcessor, process_content_request_factory
    ) -> None:
        """ブロックアクションを識別する _check_applicable_scopes をテスト。"""
        from agent_framework_purview._models import (
            PolicyLocation,
            PolicyScope,
            ProtectionScopeActivities,
            ProtectionScopesResponse,
        )

        request = process_content_request_factory()

        block_action = DlpActionInfo(action=DlpAction.BLOCK_ACCESS, restrictionAction=RestrictionAction.BLOCK)
        scope_location = PolicyLocation(**{
            "@odata.type": "microsoft.graph.policyLocationApplication",
            "value": "app-id",
        })
        scope = PolicyScope(**{
            "policyActions": [block_action],
            "activities": ProtectionScopeActivities.UPLOAD_TEXT,
            "locations": [scope_location],
        })
        response = ProtectionScopesResponse(**{"value": [scope]})

        should_process, actions = processor._check_applicable_scopes(request, response)

        assert should_process is True
        assert len(actions) == 1
        assert actions[0].action == DlpAction.BLOCK_ACCESS

    async def test_combine_policy_actions(self, processor: ScopedContentProcessor) -> None:
        """アクションリストをマージする _combine_policy_actions をテスト。"""
        action1 = DlpActionInfo(action=DlpAction.BLOCK_ACCESS, restrictionAction=RestrictionAction.BLOCK)
        action2 = DlpActionInfo(action=DlpAction.OTHER, restrictionAction=RestrictionAction.OTHER)

        combined = processor._combine_policy_actions([action1], [action2])

        assert len(combined) == 2
        assert action1 in combined
        assert action2 in combined

    async def test_process_with_scopes_calls_client_methods(
        self, processor: ScopedContentProcessor, mock_client: AsyncMock, process_content_request_factory
    ) -> None:
        """get_protection_scopes と process_content を呼び出す _process_with_scopes をテスト。"""
        from agent_framework_purview._models import (
            ContentActivitiesResponse,
            ProtectionScopesResponse,
        )

        request = process_content_request_factory()

        mock_client.get_protection_scopes = AsyncMock(return_value=ProtectionScopesResponse(**{"value": []}))
        mock_client.process_content = AsyncMock(
            return_value=ProcessContentResponse(**{"id": "response-123", "protectionScopeState": "notModified"})
        )
        mock_client.send_content_activities = AsyncMock(return_value=ContentActivitiesResponse(**{"error": None}))

        response = await processor._process_with_scopes(request)

        mock_client.get_protection_scopes.assert_called_once()
        mock_client.process_content.assert_not_called()
        mock_client.send_content_activities.assert_called_once()
        assert response.id is None

    async def test_map_messages_with_user_id_in_additional_properties(self, mock_client: AsyncMock) -> None:
        """message の additional_properties から user_id を抽出することをテスト。"""
        settings = PurviewSettings(
            app_name="Test App",
            tenant_id="12345678-1234-1234-1234-123456789012",
            purview_app_location=PurviewAppLocation(
                location_type=PurviewLocationType.APPLICATION, location_value="app-id"
            ),
        )
        processor = ScopedContentProcessor(mock_client, settings)

        messages = [
            ChatMessage(
                role=Role.USER,
                text="Test message",
                additional_properties={"user_id": "22345678-1234-1234-1234-123456789012"},
            ),
        ]

        requests, user_id = await processor._map_messages(messages, Activity.UPLOAD_TEXT)

        assert len(requests) == 1
        assert user_id == "22345678-1234-1234-1234-123456789012"
        assert requests[0].user_id == "22345678-1234-1234-1234-123456789012"

    async def test_map_messages_with_provided_user_id_fallback(self, mock_client: AsyncMock) -> None:
        """他のソースがない場合に provided_user_id を使用することをテスト。"""
        settings = PurviewSettings(
            app_name="Test App",
            tenant_id="12345678-1234-1234-1234-123456789012",
            purview_app_location=PurviewAppLocation(
                location_type=PurviewLocationType.APPLICATION, location_value="app-id"
            ),
        )
        processor = ScopedContentProcessor(mock_client, settings)

        messages = [ChatMessage(role=Role.USER, text="Test message")]

        requests, user_id = await processor._map_messages(
            messages, Activity.UPLOAD_TEXT, provided_user_id="32345678-1234-1234-1234-123456789012"
        )

        assert len(requests) == 1
        assert user_id == "32345678-1234-1234-1234-123456789012"
        assert requests[0].user_id == "32345678-1234-1234-1234-123456789012"

    async def test_map_messages_returns_empty_when_no_user_id(self, mock_client: AsyncMock) -> None:
        """user_id が解決できない場合に空の結果が返されることをテスト。"""
        settings = PurviewSettings(
            app_name="Test App",
            tenant_id="12345678-1234-1234-1234-123456789012",
            purview_app_location=PurviewAppLocation(
                location_type=PurviewLocationType.APPLICATION, location_value="app-id"
            ),
        )
        processor = ScopedContentProcessor(mock_client, settings)

        messages = [ChatMessage(role=Role.USER, text="Test message")]

        requests, user_id = await processor._map_messages(messages, Activity.UPLOAD_TEXT)

        assert len(requests) == 0
        assert user_id is None

    async def test_process_content_sends_activities_when_not_applicable(
        self, mock_client: AsyncMock, process_content_request_factory
    ) -> None:
        """スコープが適用されない場合にコンテンツアクティビティが送信されることをテスト。"""
        settings = PurviewSettings(
            app_name="Test App",
            tenant_id="12345678-1234-1234-1234-123456789012",
            purview_app_location=PurviewAppLocation(
                location_type=PurviewLocationType.APPLICATION, location_value="app-id"
            ),
        )
        processor = ScopedContentProcessor(mock_client, settings)

        pc_request = process_content_request_factory()

        # 適用可能なスコープがないように get_protection_scopes をモック。
        mock_ps_response = MagicMock()
        mock_ps_response.scopes = []
        mock_client.get_protection_scopes.return_value = mock_ps_response

        # 成功を返すように send_content_activities をモック。
        mock_ca_response = MagicMock()
        mock_ca_response.error = None
        mock_client.send_content_activities.return_value = mock_ca_response

        response = await processor._process_with_scopes(pc_request)

        mock_client.get_protection_scopes.assert_called_once()
        mock_client.process_content.assert_not_called()
        mock_client.send_content_activities.assert_called_once()
        # コンテンツアクティビティが成功した場合、レスポンスにエラーがないこと（processing_errors は None または空でよい）。
        assert response.processing_errors is None or response.processing_errors == []

    async def test_process_content_handles_activities_error(
        self, mock_client: AsyncMock, process_content_request_factory
    ) -> None:
        """コンテンツアクティビティが失敗した場合のエラー処理をテスト。"""
        settings = PurviewSettings(
            app_name="Test App",
            tenant_id="12345678-1234-1234-1234-123456789012",
            purview_app_location=PurviewAppLocation(
                location_type=PurviewLocationType.APPLICATION, location_value="app-id"
            ),
        )
        processor = ScopedContentProcessor(mock_client, settings)

        pc_request = process_content_request_factory()

        # 適用可能なスコープがないように get_protection_scopes をモック。
        mock_ps_response = MagicMock()
        mock_ps_response.scopes = []
        mock_client.get_protection_scopes.return_value = mock_ps_response

        # エラーを返すように send_content_activities をモック。
        mock_ca_response = MagicMock()
        mock_ca_response.error = "Test error message"
        mock_client.send_content_activities.return_value = mock_ca_response

        response = await processor._process_with_scopes(pc_request)

        assert len(response.processing_errors) == 1
        assert response.processing_errors[0].message == "Test error message"
