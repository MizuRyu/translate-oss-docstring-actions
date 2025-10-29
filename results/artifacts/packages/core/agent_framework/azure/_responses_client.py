# Copyright (c) Microsoft. All rights reserved.

from collections.abc import Mapping
from typing import Any, TypeVar
from urllib.parse import urljoin

from azure.core.credentials import TokenCredential
from openai.lib.azure import AsyncAzureADTokenProvider, AsyncAzureOpenAI
from pydantic import ValidationError

from agent_framework import use_chat_middleware, use_function_invocation
from agent_framework.exceptions import ServiceInitializationError
from agent_framework.observability import use_observability
from agent_framework.openai._responses_client import OpenAIBaseResponsesClient

from ._shared import (
    AzureOpenAIConfigMixin,
    AzureOpenAISettings,
)

TAzureOpenAIResponsesClient = TypeVar("TAzureOpenAIResponsesClient", bound="AzureOpenAIResponsesClient")


@use_function_invocation
@use_observability
@use_chat_middleware
class AzureOpenAIResponsesClient(AzureOpenAIConfigMixin, OpenAIBaseResponsesClient):
    """Azure Responses completion クラス。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        deployment_name: str | None = None,
        endpoint: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        ad_token: str | None = None,
        ad_token_provider: AsyncAzureADTokenProvider | None = None,
        token_endpoint: str | None = None,
        credential: TokenCredential | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncAzureOpenAI | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        instruction_role: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Azure OpenAI Responsesクライアントを初期化します。

        キーワード引数:
            api_key: APIキー。指定された場合、環境変数や.envファイルの値を上書きします。
                環境変数AZURE_OPENAI_API_KEYでも設定可能です。
            deployment_name: デプロイメント名。指定された場合、
                環境変数や.envファイルの(responses_deployment_name)の値を上書きします。
                環境変数AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAMEでも設定可能です。
            endpoint: デプロイメントのエンドポイント。指定された場合、
                環境変数や.envファイルの値を上書きします。
                環境変数AZURE_OPENAI_ENDPOINTでも設定可能です。
            base_url: デプロイメントのベースURL。指定された場合、
                環境変数や.envファイルの値を上書きします。現在、base_urlは必ず"/openai/v1/"で終わる必要があります。
                環境変数AZURE_OPENAI_BASE_URLでも設定可能です。
            api_version: デプロイメントのAPIバージョン。指定された場合、
                環境変数や.envファイルの値を上書きします。現在、api_versionは"preview"である必要があります。
                環境変数AZURE_OPENAI_API_VERSIONでも設定可能です。
            ad_token: Azure Active Directoryのトークン。
            ad_token_provider: Azure Active Directoryのトークンプロバイダー。
            token_endpoint: Azureトークンをリクエストするためのトークンエンドポイント。
                環境変数AZURE_OPENAI_TOKEN_ENDPOINTでも設定可能です。
            credential: 認証用のAzureクレデンシャル。
            default_headers: HTTPリクエスト用の文字列キーと文字列値のマッピングであるデフォルトヘッダー。
            async_client: 使用する既存のクライアント。
            env_file_path: 環境変数の代わりに環境設定ファイルをフォールバックとして使用。
            env_file_encoding: 環境設定ファイルのエンコーディング。デフォルトは'utf-8'。
            instruction_role: 'instruction'メッセージに使用するロール。例えば、要約プロンプトは`developer`や`system`を使用可能。
            kwargs: 追加のキーワード引数。

        Examples:
            .. code-block:: python

                from agent_framework.azure import AzureOpenAIResponsesClient

                # 環境変数を使用する場合
                # AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com を設定
                # AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-4o を設定
                # AZURE_OPENAI_API_KEY=your-key を設定
                client = AzureOpenAIResponsesClient()

                # またはパラメータを直接渡す場合
                client = AzureOpenAIResponsesClient(
                    endpoint="https://your-endpoint.openai.azure.com", deployment_name="gpt-4o", api_key="your-key"
                )

                # または.envファイルから読み込む場合
                client = AzureOpenAIResponsesClient(env_file_path="path/to/.env")

        """
        if model_id := kwargs.pop("model_id", None) and not deployment_name:
            deployment_name = str(model_id)
        try:
            azure_openai_settings = AzureOpenAISettings(
                # pydanticの設定は値があるか確認し、なければ環境変数や.envファイルを試みます。
                api_key=api_key,  # type: ignore
                base_url=base_url,  # type: ignore
                endpoint=endpoint,  # type: ignore
                responses_deployment_name=deployment_name,
                api_version=api_version,
                env_file_path=env_file_path,
                env_file_encoding=env_file_encoding,
                token_endpoint=token_endpoint,
                default_api_version="preview",
            )
            # TODO(peterychang): これはプレビュー中の機能でbase_urlが正しく設定されていることを保証するための一時的なハックです。
            # ただし、これはAzure上にいる場合のみ行うべきです。プライベートデプロイメントでは必要ないかもしれません。
            if (
                not azure_openai_settings.base_url
                and azure_openai_settings.endpoint
                and azure_openai_settings.endpoint.host
                and azure_openai_settings.endpoint.host.endswith(".openai.azure.com")
            ):
                azure_openai_settings.base_url = urljoin(str(azure_openai_settings.endpoint), "/openai/v1/")  # type: ignore
        except ValidationError as exc:
            raise ServiceInitializationError(f"Failed to validate settings: {exc}") from exc

        if not azure_openai_settings.responses_deployment_name:
            raise ServiceInitializationError(
                "Azure OpenAI deployment name is required. Set via 'deployment_name' parameter "
                "or 'AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME' environment variable."
            )

        super().__init__(
            deployment_name=azure_openai_settings.responses_deployment_name,
            endpoint=azure_openai_settings.endpoint,
            base_url=azure_openai_settings.base_url,
            api_version=azure_openai_settings.api_version,  # type: ignore
            api_key=azure_openai_settings.api_key.get_secret_value() if azure_openai_settings.api_key else None,
            ad_token=ad_token,
            ad_token_provider=ad_token_provider,
            token_endpoint=azure_openai_settings.token_endpoint,
            credential=credential,
            default_headers=default_headers,
            client=async_client,
            instruction_role=instruction_role,
        )
