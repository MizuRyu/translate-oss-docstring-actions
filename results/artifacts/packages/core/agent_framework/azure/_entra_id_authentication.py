# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import TYPE_CHECKING, Any

from azure.core.exceptions import ClientAuthenticationError

from ..exceptions import ServiceInvalidAuthError

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
    from azure.core.credentials_async import AsyncTokenCredential

logger: logging.Logger = logging.getLogger(__name__)


def get_entra_auth_token(
    credential: "TokenCredential",
    token_endpoint: str,
    **kwargs: Any,
) -> str | None:
    """指定されたトークンエンドポイントの Microsoft Entra 認証トークンを取得します。

    トークンエンドポイントは環境変数、.env ファイル、または引数で指定可能です。指定がない場合は None がデフォルトです。

    Args:
        credential: 認証に使用する Azure 資格情報。
        token_endpoint: 認証トークンを取得するためのトークンエンドポイント。

    Keyword Args:
        **kwargs: トークン取得メソッドに渡す追加のキーワード引数。

    Returns:
        Azure トークン、または取得できなかった場合は None。

    """
    if not token_endpoint:
        raise ServiceInvalidAuthError(
            "A token endpoint must be provided either in settings, as an environment variable, or as an argument."
        )

    try:
        auth_token = credential.get_token(token_endpoint, **kwargs)
    except ClientAuthenticationError as ex:
        logger.error(f"Failed to retrieve Azure token for the specified endpoint: `{token_endpoint}`, with error: {ex}")
        return None

    return auth_token.token if auth_token else None


async def get_entra_auth_token_async(
    credential: "AsyncTokenCredential", token_endpoint: str, **kwargs: Any
) -> str | None:
    """指定されたトークンエンドポイントの非同期 Microsoft Entra 認証トークンを取得します。

    トークンエンドポイントは環境変数、.env ファイル、または引数で指定可能です。指定がない場合は None がデフォルトです。

    Args:
        credential: 認証に使用する非同期 Azure 資格情報。
        token_endpoint: 認証トークンを取得するためのトークンエンドポイント。

    Keyword Args:
        **kwargs: トークン取得メソッドに渡す追加のキーワード引数。

    Returns:
        Azure トークン、または取得できなかった場合は None。

    """
    if not token_endpoint:
        raise ServiceInvalidAuthError(
            "A token endpoint must be provided either in settings, as an environment variable, or as an argument."
        )

    try:
        auth_token = await credential.get_token(token_endpoint, **kwargs)
    except ClientAuthenticationError as ex:
        logger.error(f"Failed to retrieve Azure token for the specified endpoint: `{token_endpoint}`, with error: {ex}")
        return None

    return auth_token.token if auth_token else None
