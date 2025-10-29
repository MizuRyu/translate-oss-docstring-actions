# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any, TypeVar

from azure.core.credentials import TokenCredential
from openai.lib.azure import AsyncAzureADTokenProvider, AsyncAzureOpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from pydantic import ValidationError

from agent_framework import (
    ChatResponse,
    ChatResponseUpdate,
    CitationAnnotation,
    TextContent,
    use_chat_middleware,
    use_function_invocation,
)
from agent_framework.exceptions import ServiceInitializationError
from agent_framework.observability import use_observability
from agent_framework.openai._chat_client import OpenAIBaseChatClient

from ._shared import (
    AzureOpenAIConfigMixin,
    AzureOpenAISettings,
)

if sys.version_info >= (3, 12):
    from typing import override  # type: ignore # pragma: no cover
else:
    from typing_extensions import override  # type: ignore[import] # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)

TChatResponse = TypeVar("TChatResponse", ChatResponse, ChatResponseUpdate)
TAzureOpenAIChatClient = TypeVar("TAzureOpenAIChatClient", bound="AzureOpenAIChatClient")


@use_function_invocation
@use_observability
@use_chat_middleware
class AzureOpenAIChatClient(AzureOpenAIConfigMixin, OpenAIBaseChatClient):
    """Azure OpenAI Chat completion クラス。"""

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
        """Azure OpenAI Chat completion クライアントを初期化します。

        Keyword Args:
            api_key: API キー。指定すると環境変数や .env ファイルの値を上書きします。
                環境変数 AZURE_OPENAI_API_KEY でも設定可能です。
            deployment_name: デプロイメント名。指定すると環境変数や .env ファイルの値（chat_deployment_name）を上書きします。
                環境変数 AZURE_OPENAI_CHAT_DEPLOYMENT_NAME でも設定可能です。
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
            env_file_encoding: 環境設定ファイルのエンコーディング。デフォルトは 'utf-8'。
            instruction_role: 'instruction' メッセージに使用するロール。例えば要約プロンプトは `developer` や `system` を使用可能。
            kwargs: その他のキーワードパラメータ。

        Examples:
            .. code-block:: python

                from agent_framework.azure import AzureOpenAIChatClient

                # 環境変数を使用する場合
                # Set AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
                # Set AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4
                # Set AZURE_OPENAI_API_KEY=your-key
                client = AzureOpenAIChatClient()

                # またはパラメータを直接渡す場合
                client = AzureOpenAIChatClient(
                    endpoint="https://your-endpoint.openai.azure.com", deployment_name="gpt-4", api_key="your-key"
                )

                # または .env ファイルから読み込む場合
                client = AzureOpenAIChatClient(env_file_path="path/to/.env")

        """
        try:
            # 引数から None の値を除外する
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
            )
        except ValidationError as exc:
            raise ServiceInitializationError(f"Failed to validate settings: {exc}") from exc

        if not azure_openai_settings.chat_deployment_name:
            raise ServiceInitializationError(
                "Azure OpenAI deployment name is required. Set via 'deployment_name' parameter "
                "or 'AZURE_OPENAI_CHAT_DEPLOYMENT_NAME' environment variable."
            )

        super().__init__(
            deployment_name=azure_openai_settings.chat_deployment_name,
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
            **kwargs,
        )

    @override
    def _parse_text_from_choice(self, choice: Choice | ChunkChoice) -> TextContent | None:
        """choice を TextContent オブジェクトに解析します。

        OpenAIBaseChatClient からオーバーライドし、Azure On Your Data 機能に対応しています。
        ドキュメントは以下を参照してください:
        https://learn.microsoft.com/en-us/azure/ai-foundry/openai/references/on-your-data?tabs=python#context

        """
        message = choice.message if isinstance(choice, Choice) else choice.delta
        if hasattr(message, "refusal") and message.refusal:
            return TextContent(text=message.refusal, raw_representation=choice)
        if not message.content:
            return None
        text_content = TextContent(text=message.content, raw_representation=choice)
        if not message.model_extra or "context" not in message.model_extra:
            return text_content

        context: dict[str, Any] | str = message.context  # type: ignore[assignment, union-attr]
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except json.JSONDecodeError:
                logger.warning("Context is not a valid JSON string, ignoring context.")
                return text_content
        if not isinstance(context, dict):
            logger.warning("Context is not a valid dictionary, ignoring context.")
            return text_content
        # `all_retrieved_documents` は現在使用されていませんが、テキストコンテンツの raw_representation
        # から取得可能です。
        if intent := context.get("intent"):
            text_content.additional_properties = {"intent": intent}
        if citations := context.get("citations"):
            text_content.annotations = []
            for citation in citations:
                text_content.annotations.append(
                    CitationAnnotation(
                        title=citation.get("title", ""),
                        url=citation.get("url", ""),
                        snippet=citation.get("content", ""),
                        file_id=citation.get("filepath", ""),
                        tool_name="Azure-on-your-Data",
                        additional_properties={"chunk_id": citation.get("chunk_id", "")},
                        raw_representation=citation,
                    )
                )
        return text_content
