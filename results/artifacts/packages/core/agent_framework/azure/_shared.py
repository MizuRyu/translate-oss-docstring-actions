# Copyright (c) Microsoft. All rights reserved.

import logging
import sys
from collections.abc import Awaitable, Callable, Mapping
from copy import copy
from typing import Any, ClassVar, Final

from azure.core.credentials import TokenCredential
from openai.lib.azure import AsyncAzureOpenAI
from pydantic import SecretStr, model_validator

from .._pydantic import AFBaseSettings, HTTPsUrl
from .._telemetry import APP_INFO, prepend_agent_framework_to_user_agent
from ..exceptions import ServiceInitializationError
from ..openai._shared import OpenAIBase
from ._entra_id_authentication import get_entra_auth_token

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


DEFAULT_AZURE_API_VERSION: Final[str] = "2024-10-21"
DEFAULT_AZURE_TOKEN_ENDPOINT: Final[str] = "https://cognitiveservices.azure.com/.default"  # noqa: S105


class AzureOpenAISettings(AFBaseSettings):
    """AzureOpenAIのモデル設定。

    設定はまずプレフィックス'AZURE_OPENAI_'の環境変数から読み込まれます。
    環境変数が見つからない場合、エンコーディング'utf-8'の.envファイルから読み込むことができます。
    .envファイルにも設定が見つからない場合は無視されますが、検証は失敗し設定が不足していることを警告します。

    キーワード引数:
        endpoint: Azureデプロイメントのエンドポイント。この値は
            AzureポータルのKeys & Endpointセクションで確認できます。
            エンドポイントはopenai.azure.comで終わる必要があります。
            base_urlとendpointの両方が指定された場合はbase_urlが使用されます。
            環境変数AZURE_OPENAI_ENDPOINTで設定可能です。
        chat_deployment_name: Azure Chatデプロイメントの名前。この値は
            モデルをデプロイした際に選択したカスタム名に対応します。
            AzureポータルのResource Management > Deployments、または
            Azure AI FoundryのManagement > Deploymentsで確認できます。
            環境変数AZURE_OPENAI_CHAT_DEPLOYMENT_NAMEで設定可能です。
        responses_deployment_name: Azure Responsesデプロイメントの名前。この値は
            モデルをデプロイした際に選択したカスタム名に対応します。
            AzureポータルのResource Management > Deployments、または
            Azure AI FoundryのManagement > Deploymentsで確認できます。
            環境変数AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAMEで設定可能です。
        api_key: AzureデプロイメントのAPIキー。この値は
            AzureポータルのKeys & Endpointセクションで確認できます。
            KEY1またはKEY2のいずれかを使用可能です。
            環境変数AZURE_OPENAI_API_KEYで設定可能です。
        api_version: 使用するAPIバージョン。デフォルトは`default_api_version`です。
            環境変数AZURE_OPENAI_API_VERSIONで設定可能です。
        base_url: AzureデプロイメントのURL。この値は
            AzureポータルのKeys & Endpointセクションで確認できます。
            base_urlはendpointに続き/openai/deployments/{deployment_name}/を含みます。
            エンドポイントのみを指定したい場合はendpointを使用してください。
            環境変数AZURE_OPENAI_BASE_URLで設定可能です。
        token_endpoint: 認証トークンを取得するためのトークンエンドポイント。
            デフォルトは`default_token_endpoint`です。
            環境変数AZURE_OPENAI_TOKEN_ENDPOINTで設定可能です。
        default_api_version: 指定がない場合に使用するデフォルトのAPIバージョン。
            デフォルト値は"2024-10-21"です。
        default_token_endpoint: 指定がない場合に使用するデフォルトのトークンエンドポイント。
            デフォルト値は"https://cognitiveservices.azure.com/.default"です。
        env_file_path: 設定を読み込む.envファイルのパス。
        env_file_encoding: .envファイルのエンコーディング。デフォルトは'utf-8'。

    Examples:
        .. code-block:: python

            from agent_framework.azure import AzureOpenAISettings

            # 環境変数を使用する場合
            # AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com を設定
            # AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4 を設定
            # AZURE_OPENAI_API_KEY=your-key を設定
            settings = AzureOpenAISettings()

            # またはパラメータを直接渡す場合
            settings = AzureOpenAISettings(
                endpoint="https://your-endpoint.openai.azure.com", chat_deployment_name="gpt-4", api_key="your-key"
            )

            # または.envファイルから読み込む場合
            settings = AzureOpenAISettings(env_file_path="path/to/.env")

    """

    env_prefix: ClassVar[str] = "AZURE_OPENAI_"

    chat_deployment_name: str | None = None
    responses_deployment_name: str | None = None
    endpoint: HTTPsUrl | None = None
    base_url: HTTPsUrl | None = None
    api_key: SecretStr | None = None
    api_version: str | None = None
    token_endpoint: str | None = None
    default_api_version: str = DEFAULT_AZURE_API_VERSION
    default_token_endpoint: str = DEFAULT_AZURE_TOKEN_ENDPOINT

    def get_azure_auth_token(
        self, credential: "TokenCredential", token_endpoint: str | None = None, **kwargs: Any
    ) -> str | None:
        """Azure OpenAIで使用するために、指定されたトークンエンドポイントからMicrosoft Entra認証トークンを取得します。

        トークンに必要なロールは`Cognitive Services OpenAI Contributor`です。
        トークンエンドポイントは環境変数、.envファイル、または引数で指定可能です。
        指定がない場合、デフォルトはNoneです。
        `token_endpoint`引数は`token_endpoint`属性より優先されます。

        引数:
            credential: 使用するAzure ADクレデンシャル。
            token_endpoint: 使用するトークンエンドポイント。デフォルトは`https://cognitiveservices.azure.com/.default`。

        キーワード引数:
            **kwargs: トークン取得メソッドに渡す追加のキーワード引数。

        戻り値:
            Azureトークン、または取得できなかった場合はNone。

        例外:
            ServiceInitializationError: トークンエンドポイントが指定されていない場合に発生。

        """
        endpoint_to_use = token_endpoint or self.token_endpoint or self.default_token_endpoint
        return get_entra_auth_token(credential, endpoint_to_use, **kwargs)

    @model_validator(mode="after")
    def _validate_fields(self) -> Self:
        self.api_version = self.api_version or self.default_api_version
        self.token_endpoint = self.token_endpoint or self.default_token_endpoint
        return self


class AzureOpenAIConfigMixin(OpenAIBase):
    """Azure OpenAIサービスへの接続を設定するための内部クラス。"""

    OTEL_PROVIDER_NAME: ClassVar[str] = "azure.ai.openai"
    # 注: INJECTABLE = {"client"} はOpenAIBaseから継承されています。

    def __init__(
        self,
        deployment_name: str,
        endpoint: HTTPsUrl | None = None,
        base_url: HTTPsUrl | None = None,
        api_version: str = DEFAULT_AZURE_API_VERSION,
        api_key: str | None = None,
        ad_token: str | None = None,
        ad_token_provider: Callable[[], str | Awaitable[str]] | None = None,
        token_endpoint: str | None = None,
        credential: TokenCredential | None = None,
        default_headers: Mapping[str, str] | None = None,
        client: AsyncAzureOpenAI | None = None,
        instruction_role: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Azure OpenAIサービスへの接続を設定するための内部クラス。

        `validate_call`デコレーターは任意の型を許容する設定で使用されます。
        これは`HTTPsUrl`や`OpenAIModelTypes`のような型に必要です。

        引数:
            deployment_name: デプロイメントの名前。
            endpoint: デプロイメントの特定のエンドポイントURL。
            base_url: AzureサービスのベースURL。
            api_version: Azure APIのバージョン。デフォルトは定義されたDEFAULT_AZURE_API_VERSION。
            api_key: AzureサービスのAPIキー。
            ad_token: 認証用のAzure ADトークン。
            ad_token_provider: Azure ADトークンを提供する呼び出し可能またはコルーチン関数。
            token_endpoint: トークン取得に使用するAzure ADトークンエンドポイント。
            credential: 認証用のAzureクレデンシャル。
            default_headers: HTTPリクエストのデフォルトヘッダー。
            client: 使用する既存のクライアント。
            instruction_role: 'instruction'メッセージに使用するロール。例えば、要約プロンプトは`developer`や`system`を使用可能。
            kwargs: 追加のキーワード引数。


        """
        # APP_INFOが存在する場合、ヘッダーにマージします。
        merged_headers = dict(copy(default_headers)) if default_headers else {}
        if APP_INFO:
            merged_headers.update(APP_INFO)
            merged_headers = prepend_agent_framework_to_user_agent(merged_headers)
        if not client:
            # clientがNoneで、api_keyがNoneで、ad_tokenがNoneで、ad_token_providerもNoneの場合、 Azure
            # OpenAI設定で指定されたデフォルトエンドポイントを使用してad_tokenを取得しようとします。
            if not api_key and not ad_token_provider and not ad_token and token_endpoint and credential:
                ad_token = get_entra_auth_token(credential, token_endpoint)

            if not api_key and not ad_token and not ad_token_provider:
                raise ServiceInitializationError(
                    "Please provide either api_key, ad_token or ad_token_provider or a client."
                )

            if not endpoint and not base_url:
                raise ServiceInitializationError("Please provide an endpoint or a base_url")

            args: dict[str, Any] = {
                "default_headers": merged_headers,
            }
            if api_version:
                args["api_version"] = api_version
            if ad_token:
                args["azure_ad_token"] = ad_token
            if ad_token_provider:
                args["azure_ad_token_provider"] = ad_token_provider
            if api_key:
                args["api_key"] = api_key
            if base_url:
                args["base_url"] = str(base_url)
            if endpoint and not base_url:
                args["azure_endpoint"] = str(endpoint)
            if deployment_name:
                args["azure_deployment"] = deployment_name
            if "websocket_base_url" in kwargs:
                args["websocket_base_url"] = kwargs.pop("websocket_base_url")

            client = AsyncAzureOpenAI(**args)

        # シリアライズのために設定をインスタンス属性として保存します。
        self.endpoint = str(endpoint)
        self.base_url = str(base_url)
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.instruction_role = instruction_role
        # default_headersを保存しますが、シリアライズのためにUSER_AGENT_KEYは除外します。
        if default_headers:
            from .._telemetry import USER_AGENT_KEY

            def_headers = {k: v for k, v in default_headers.items() if k != USER_AGENT_KEY}
        else:
            def_headers = None
        self.default_headers = def_headers

        super().__init__(model_id=deployment_name, client=client, **kwargs)
