# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.core.exceptions import ClientAuthenticationError

from agent_framework.azure._entra_id_authentication import (
    get_entra_auth_token,
    get_entra_auth_token_async,
)
from agent_framework.exceptions import ServiceInvalidAuthError


@pytest.fixture
def mock_credential() -> MagicMock:
    """同期の TokenCredential をモックする。"""
    mock_cred = MagicMock()
    # .token 属性を持つモックトークンオブジェクトを作成する
    mock_token = MagicMock()
    mock_token.token = "test-access-token-12345"
    mock_cred.get_token.return_value = mock_token
    return mock_cred


@pytest.fixture
def mock_async_credential() -> MagicMock:
    """非同期の AsyncTokenCredential をモックする。"""
    mock_cred = MagicMock()
    # .token 属性を持つモックトークンオブジェクトを作成する
    mock_token = MagicMock()
    mock_token.token = "test-async-access-token-12345"
    mock_cred.get_token = AsyncMock(return_value=mock_token)
    return mock_cred


def test_get_entra_auth_token_success(mock_credential: MagicMock) -> None:
    """同期関数でのトークン取得成功をテストする。"""

    token_endpoint = "https://test-endpoint.com/.default"

    result = get_entra_auth_token(mock_credential, token_endpoint)

    # アサート - 結果をチェックする
    assert result == "test-access-token-12345"
    mock_credential.get_token.assert_called_once_with(token_endpoint)


async def test_get_entra_auth_token_async_success(mock_async_credential: MagicMock) -> None:
    """非同期関数でのトークン取得成功をテストする。"""

    token_endpoint = "https://test-endpoint.com/.default"

    result = await get_entra_auth_token_async(mock_async_credential, token_endpoint)

    # アサート - 結果をチェックする
    assert result == "test-async-access-token-12345"
    mock_async_credential.get_token.assert_called_once_with(token_endpoint)


def test_get_entra_auth_token_missing_endpoint(mock_credential: MagicMock) -> None:
    """トークンエンドポイントがない場合に ServiceInvalidAuthError が発生することをテストする。"""
    # 空文字列でテストする
    with pytest.raises(ServiceInvalidAuthError, match="A token endpoint must be provided"):
        get_entra_auth_token(mock_credential, "")

    # None でテストする
    with pytest.raises(ServiceInvalidAuthError, match="A token endpoint must be provided"):
        get_entra_auth_token(mock_credential, None)  # type: ignore


async def test_get_entra_auth_token_async_missing_endpoint(mock_async_credential: MagicMock) -> None:
    """非同期関数でトークンエンドポイントがない場合に ServiceInvalidAuthError が発生することをテストする。"""
    # 空文字列でテストする
    with pytest.raises(ServiceInvalidAuthError, match="A token endpoint must be provided"):
        await get_entra_auth_token_async(mock_async_credential, "")

    # None でテストする
    with pytest.raises(ServiceInvalidAuthError, match="A token endpoint must be provided"):
        await get_entra_auth_token_async(mock_async_credential, None)  # type: ignore


def test_get_entra_auth_token_auth_failure(mock_credential: MagicMock) -> None:
    """Azure 認証失敗時に None が返ることをテストする。"""

    mock_credential.get_token.side_effect = ClientAuthenticationError("Auth failed")
    token_endpoint = "https://test-endpoint.com/.default"

    result = get_entra_auth_token(mock_credential, token_endpoint)

    # アサート - 認証失敗時に None を返すべき
    assert result is None
    mock_credential.get_token.assert_called_once_with(token_endpoint)


async def test_get_entra_auth_token_async_auth_failure(mock_async_credential: MagicMock) -> None:
    """非同期関数で Azure 認証失敗時に None が返ることをテストする。"""

    mock_async_credential.get_token.side_effect = ClientAuthenticationError("Auth failed")
    token_endpoint = "https://test-endpoint.com/.default"

    result = await get_entra_auth_token_async(mock_async_credential, token_endpoint)

    # アサート - 認証失敗時に None を返すべき
    assert result is None
    mock_async_credential.get_token.assert_called_once_with(token_endpoint)


def test_get_entra_auth_token_none_token_response(mock_credential: MagicMock) -> None:
    """None のトークンレスポンスが None を返すことをテストする。"""
    mock_credential.get_token.return_value = None
    token_endpoint = "https://test-endpoint.com/.default"

    result = get_entra_auth_token(mock_credential, token_endpoint)

    # アサート
    assert result is None
    mock_credential.get_token.assert_called_once_with(token_endpoint)


async def test_get_entra_auth_token_async_none_token_response(mock_async_credential: MagicMock) -> None:
    """非同期関数で None のトークンレスポンスが None を返すことをテストする。"""
    mock_async_credential.get_token.return_value = None
    token_endpoint = "https://test-endpoint.com/.default"

    result = await get_entra_auth_token_async(mock_async_credential, token_endpoint)

    # アサート
    assert result is None
    mock_async_credential.get_token.assert_called_once_with(token_endpoint)


def test_get_entra_auth_token_with_kwargs(mock_credential: MagicMock) -> None:
    """kwargs が get_token に渡されることをテストする。"""

    token_endpoint = "https://test-endpoint.com/.default"
    extra_kwargs = {"scopes": ["read", "write"], "tenant_id": "test-tenant"}

    result = get_entra_auth_token(mock_credential, token_endpoint, **extra_kwargs)

    # アサート
    assert result == "test-access-token-12345"
    mock_credential.get_token.assert_called_once_with(token_endpoint, **extra_kwargs)


async def test_get_entra_auth_token_async_with_kwargs(mock_async_credential: MagicMock) -> None:
    """kwargs が非同期 get_token に渡されることをテストする。"""

    token_endpoint = "https://test-endpoint.com/.default"
    extra_kwargs = {"scopes": ["read", "write"], "tenant_id": "test-tenant"}

    result = await get_entra_auth_token_async(mock_async_credential, token_endpoint, **extra_kwargs)

    # アサート
    assert result == "test-async-access-token-12345"
    mock_async_credential.get_token.assert_called_once_with(token_endpoint, **extra_kwargs)
