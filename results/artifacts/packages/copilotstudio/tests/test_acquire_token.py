# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
from agent_framework.exceptions import ServiceException

from agent_framework_copilotstudio._acquire_token import DEFAULT_SCOPES, acquire_token


class TestAcquireToken:
    """トークン取得機能のテストクラス。"""

    def test_acquire_token_missing_client_id(self) -> None:
        """client_idが欠落している場合にacquire_tokenがServiceExceptionを発生させることをテスト。"""
        with pytest.raises(ServiceException, match="Client ID is required for token acquisition"):
            acquire_token(client_id="", tenant_id="test-tenant-id")

    def test_acquire_token_missing_tenant_id(self) -> None:
        """tenant_idが欠落している場合にacquire_tokenがServiceExceptionを発生させることをテスト。"""
        with pytest.raises(ServiceException, match="Tenant ID is required for token acquisition"):
            acquire_token(client_id="test-client-id", tenant_id="")

    def test_acquire_token_none_client_id(self) -> None:
        """client_idがNoneの場合にacquire_tokenがServiceExceptionを発生させることをテスト。"""
        with pytest.raises(ServiceException, match="Client ID is required for token acquisition"):
            acquire_token(client_id=None, tenant_id="test-tenant-id")  # type: ignore

    def test_acquire_token_none_tenant_id(self) -> None:
        """tenant_idがNoneの場合にacquire_tokenがServiceExceptionを発生させることをテスト。"""
        with pytest.raises(ServiceException, match="Tenant ID is required for token acquisition"):
            acquire_token(client_id="test-client-id", tenant_id=None)  # type: ignore

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_silent_success(self, mock_pca_class: MagicMock) -> None:
        """正常なサイレントトークン取得のテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_account = MagicMock()
        mock_pca.get_accounts.return_value = [mock_account]

        mock_token_response = {"access_token": "test-access-token-12345"}
        mock_pca.acquire_token_silent.return_value = mock_token_response

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
        )

        assert result == "test-access-token-12345"
        mock_pca_class.assert_called_once_with(
            client_id="test-client-id",
            authority="https://login.microsoftonline.com/test-tenant-id",
            token_cache=None,
        )
        mock_pca.get_accounts.assert_called_once_with(username=None)
        mock_pca.acquire_token_silent.assert_called_once_with(scopes=DEFAULT_SCOPES, account=mock_account)

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_silent_success_with_username(self, mock_pca_class: MagicMock) -> None:
        """username付きの正常なサイレントトークン取得のテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_account = MagicMock()
        mock_pca.get_accounts.return_value = [mock_account]

        mock_token_response = {"access_token": "test-access-token-12345"}
        mock_pca.acquire_token_silent.return_value = mock_token_response

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            username="test-user@example.com",
        )

        assert result == "test-access-token-12345"
        mock_pca.get_accounts.assert_called_once_with(username="test-user@example.com")
        mock_pca.acquire_token_silent.assert_called_once_with(scopes=DEFAULT_SCOPES, account=mock_account)

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_silent_success_with_custom_scopes(self, mock_pca_class: MagicMock) -> None:
        """カスタムスコープ付きの正常なサイレントトークン取得のテスト。"""
        # セットアップ。
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_account = MagicMock()
        mock_pca.get_accounts.return_value = [mock_account]

        mock_token_response = {"access_token": "test-access-token-12345"}
        mock_pca.acquire_token_silent.return_value = mock_token_response

        custom_scopes = ["https://custom.api.com/.default"]

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            scopes=custom_scopes,
        )

        assert result == "test-access-token-12345"
        mock_pca.acquire_token_silent.assert_called_once_with(scopes=custom_scopes, account=mock_account)

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_interactive_success_no_accounts(self, mock_pca_class: MagicMock) -> None:
        """キャッシュされたアカウントが存在しない場合の正常なインタラクティブトークン取得のテスト。"""
        # セットアップ。
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_pca.get_accounts.return_value = []  # キャッシュされたアカウントなし。

        mock_token_response = {"access_token": "test-interactive-token-67890"}
        mock_pca.acquire_token_interactive.return_value = mock_token_response

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
        )

        assert result == "test-interactive-token-67890"
        mock_pca.acquire_token_interactive.assert_called_once_with(scopes=DEFAULT_SCOPES)

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_fallback_to_interactive_after_silent_fails(self, mock_pca_class: MagicMock) -> None:
        """サイレント取得が失敗した場合のインタラクティブ認証へのフォールバックをテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_account = MagicMock()
        mock_pca.get_accounts.return_value = [mock_account]

        # エラー応答でサイレント取得が失敗。
        mock_silent_error_response = {"error": "invalid_grant", "error_description": "Token expired"}
        mock_pca.acquire_token_silent.return_value = mock_silent_error_response

        # インタラクティブ取得が成功。
        mock_interactive_response = {"access_token": "test-interactive-token-67890"}
        mock_pca.acquire_token_interactive.return_value = mock_interactive_response

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
        )

        assert result == "test-interactive-token-67890"
        mock_pca.acquire_token_silent.assert_called_once_with(scopes=DEFAULT_SCOPES, account=mock_account)
        mock_pca.acquire_token_interactive.assert_called_once_with(scopes=DEFAULT_SCOPES)

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_fallback_to_interactive_after_silent_exception(self, mock_pca_class: MagicMock) -> None:
        """サイレント取得が例外をスローした場合のインタラクティブ認証へのフォールバックをテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_account = MagicMock()
        mock_pca.get_accounts.return_value = [mock_account]

        # サイレント取得が例外をスロー。
        mock_pca.acquire_token_silent.side_effect = Exception("Network error")

        # インタラクティブ取得が成功。
        mock_interactive_response = {"access_token": "test-interactive-token-67890"}
        mock_pca.acquire_token_interactive.return_value = mock_interactive_response

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
        )

        assert result == "test-interactive-token-67890"
        mock_pca.acquire_token_silent.assert_called_once_with(scopes=DEFAULT_SCOPES, account=mock_account)
        mock_pca.acquire_token_interactive.assert_called_once_with(scopes=DEFAULT_SCOPES)

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_interactive_error_response(self, mock_pca_class: MagicMock) -> None:
        """インタラクティブ認証からのエラー応答をacquire_tokenが処理することをテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_pca.get_accounts.return_value = []  # キャッシュされたアカウントなし。

        # インタラクティブ取得がエラーを返す。
        mock_error_response = {"error": "access_denied", "error_description": "User denied consent"}
        mock_pca.acquire_token_interactive.return_value = mock_error_response

        with pytest.raises(ServiceException, match="Authentication token cannot be acquired"):
            acquire_token(
                client_id="test-client-id",
                tenant_id="test-tenant-id",
            )

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_interactive_exception(self, mock_pca_class: MagicMock) -> None:
        """インタラクティブ認証からの例外をacquire_tokenが処理することをテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_pca.get_accounts.return_value = []  # キャッシュされたアカウントなし。

        # インタラクティブ取得が例外をスロー。
        mock_pca.acquire_token_interactive.side_effect = Exception("Authentication service unavailable")

        with pytest.raises(ServiceException, match="Failed to acquire authentication token"):
            acquire_token(
                client_id="test-client-id",
                tenant_id="test-tenant-id",
            )

    @patch("agent_framework_copilotstudio._acquire_token.PublicClientApplication")
    def test_acquire_token_with_token_cache(self, mock_pca_class: MagicMock) -> None:
        """カスタムトークンキャッシュを使用したacquire_tokenのテスト。"""
        mock_pca = MagicMock()
        mock_pca_class.return_value = mock_pca

        mock_account = MagicMock()
        mock_pca.get_accounts.return_value = [mock_account]

        mock_token_response = {"access_token": "test-cached-token"}
        mock_pca.acquire_token_silent.return_value = mock_token_response

        mock_token_cache = MagicMock()

        result = acquire_token(
            client_id="test-client-id",
            tenant_id="test-tenant-id",
            token_cache=mock_token_cache,
        )

        assert result == "test-cached-token"
        mock_pca_class.assert_called_once_with(
            client_id="test-client-id",
            authority="https://login.microsoftonline.com/test-tenant-id",
            token_cache=mock_token_cache,
        )

    def test_default_scopes_constant(self) -> None:
        """DEFAULT_SCOPES定数が正しく定義されていることをテスト。"""
        assert DEFAULT_SCOPES == ["https://api.powerplatform.com/.default"]
        assert isinstance(DEFAULT_SCOPES, list)
        assert len(DEFAULT_SCOPES) == 1
