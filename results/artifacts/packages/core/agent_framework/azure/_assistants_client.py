# Copyright (c) Microsoft. All rights reserved.

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from openai.lib.azure import AsyncAzureADTokenProvider, AsyncAzureOpenAI
from pydantic import ValidationError

from ..exceptions import ServiceInitializationError
from ..openai import OpenAIAssistantsClient
from ._shared import AzureOpenAISettings

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

__all__ = ["AzureOpenAIAssistantsClient"]


class AzureOpenAIAssistantsClient(OpenAIAssistantsClient):
    """Azure OpenAI Assistants クライアント。"""

    DEFAULT_AZURE_API_VERSION: ClassVar[str] = "2024-05-01-preview"

    def __init__(
        self,
        *,
        deployment_name: str | None = None,
        assistant_id: str | None = None,
        assistant_name: str | None = None,
        thread_id: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        ad_token: str | None = None,
        ad_token_provider: AsyncAzureADTokenProvider | None = None,
        token_endpoint: str | None = None,
        credential: "TokenCredential | None" = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncAzureOpenAI | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Azure OpenAI Assistants クライアントを初期化します。

        Keyword Args:
            deployment_name: 使用するモデルの Azure OpenAI デプロイメント名。
                環境変数 AZURE_OPENAI_CHAT_DEPLOYMENT_NAME でも設定可能です。
            assistant_id: 使用する Azure OpenAI アシスタントの ID。
                指定しない場合、新しいアシスタントが作成され（リクエスト後に削除されます）。
            assistant_name: 新しいアシスタント作成時に使用する名前。
            thread_id: 会話に使用するデフォルトのスレッドID。リクエスト時の conversation_id プロパティで上書き可能。
                指定しない場合、新しいスレッドが作成され（リクエスト後に削除されます）。
            api_key: 使用する API キー。指定すると環境変数や .env ファイルの値を上書きします。
                環境変数 AZURE_OPENAI_API_KEY でも設定可能です。
            endpoint: デプロイメントのエンドポイント。指定すると環境変数や .env ファイルの値を上書きします。
                環境変数 AZURE_OPENAI_ENDPOINT でも設定可能です。
            base_url: デプロイメントのベース URL。指定すると環境変数や .env ファイルの値を上書きします。
                環境変数 AZURE_OPENAI_BASE_URL でも設定可能です。
            api_version: デプロイメントの API バージョン。指定すると環境変数や .env ファイルの値を上書きします。
                環境変数 AZURE_OPENAI_API_VERSION でも設定可能です。
            ad_token: Azure Active Directory トークン。
            ad_token_provider: Azure Active Directory トークンプロバイダー。
            token_endpoint: Azure トークンを要求するトークンエンドポイント。
                環境変数 AZURE_OPENAI_TOKEN_ENDPOINT でも設定可能です。
            credential: 認証に使用する Azure 資格情報。
            default_headers: HTTP リクエストのための文字列キーから文字列値へのデフォルトヘッダーのマッピング。
            async_client: 使用する既存のクライアント。
            env_file_path: 環境変数の代わりに環境設定ファイルを使用。
            env_file_encoding: 環境設定ファイルのエンコーディング。

        Examples:
            .. code-block:: python

                from agent_framework.azure import AzureOpenAIAssistantsClient

                # 環境変数を使用する場合
                # Set AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
                # Set AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4
                # Set AZURE_OPENAI_API_KEY=your-key
                client = AzureOpenAIAssistantsClient()

                # またはパラメータを直接渡す場合
                client = AzureOpenAIAssistantsClient(
                    endpoint="https://your-endpoint.openai.azure.com", deployment_name="gpt-4", api_key="your-key"
                )

                # または .env ファイルから読み込む場合
                client = AzureOpenAIAssistantsClient(env_file_path="path/to/.env")

        """
        try:
            azure_openai_settings = AzureOpenAISettings(
                # pydantic の設定は値があるか確認し、なければ環境変数や .env ファイルを試みます
                api_key=api_key,  # type: ignore
                base_url=base_url,  # type: ignore
                endpoint=endpoint,  # type: ignore
                chat_deployment_name=deployment_name,
                api_version=api_version,
                env_file_path=env_file_path,
                env_file_encoding=env_file_encoding,
                token_endpoint=token_endpoint,
                default_api_version=self.DEFAULT_AZURE_API_VERSION,
            )
        except ValidationError as ex:
            raise ServiceInitializationError("Failed to create Azure OpenAI settings.", ex) from ex

        if not azure_openai_settings.chat_deployment_name:
            raise ServiceInitializationError(
                "Azure OpenAI deployment name is required. Set via 'deployment_name' parameter "
                "or 'AZURE_OPENAI_CHAT_DEPLOYMENT_NAME' environment variable."
            )

        # 認証を処理します: まず API キー、次に AD トークン、最後に Entra ID を試みます
        if (
            not async_client
            and not azure_openai_settings.api_key
            and not ad_token
            and not ad_token_provider
            and azure_openai_settings.token_endpoint
            and credential
        ):
            ad_token = azure_openai_settings.get_azure_auth_token(credential)

        if not async_client and not azure_openai_settings.api_key and not ad_token and not ad_token_provider:
            raise ServiceInitializationError("The Azure OpenAI API key, ad_token, or ad_token_provider is required.")

        # 提供されていなければ Azure クライアントを作成します
        if not async_client:
            client_params: dict[str, Any] = {
                "api_version": azure_openai_settings.api_version,
                "default_headers": default_headers,
            }

            if azure_openai_settings.api_key:
                client_params["api_key"] = azure_openai_settings.api_key.get_secret_value()
            elif ad_token:
                client_params["azure_ad_token"] = ad_token
            elif ad_token_provider:
                client_params["azure_ad_token_provider"] = ad_token_provider

            if azure_openai_settings.base_url:
                client_params["base_url"] = str(azure_openai_settings.base_url)
            elif azure_openai_settings.endpoint:
                client_params["azure_endpoint"] = str(azure_openai_settings.endpoint)

            async_client = AsyncAzureOpenAI(**client_params)

        super().__init__(
            model_id=azure_openai_settings.chat_deployment_name,
            assistant_id=assistant_id,
            assistant_name=assistant_name,
            thread_id=thread_id,
            async_client=async_client,  # type: ignore[reportArgumentType]
            default_headers=default_headers,
        )
