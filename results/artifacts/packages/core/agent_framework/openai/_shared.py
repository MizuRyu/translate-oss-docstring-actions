# Copyright (c) Microsoft. All rights reserved.

import logging
from collections.abc import Awaitable, Callable, Mapping
from copy import copy
from typing import Any, ClassVar, Union

import openai
from openai import (
    AsyncOpenAI,
    AsyncStream,
    _legacy_response,  # type: ignore
)
from openai.types import Completion
from openai.types.audio import Transcription
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.images_response import ImagesResponse
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent
from packaging import version
from pydantic import SecretStr

from .._logging import get_logger
from .._pydantic import AFBaseSettings
from .._serialization import SerializationMixin
from .._telemetry import APP_INFO, USER_AGENT_KEY, prepend_agent_framework_to_user_agent
from .._types import ChatOptions
from ..exceptions import ServiceInitializationError

logger: logging.Logger = get_logger("agent_framework.openai")


RESPONSE_TYPE = Union[
    ChatCompletion,
    Completion,
    AsyncStream[ChatCompletionChunk],
    AsyncStream[Completion],
    list[Any],
    ImagesResponse,
    Response,
    AsyncStream[ResponseStreamEvent],
    Transcription,
    _legacy_response.HttpxBinaryResponseContent,
]

OPTION_TYPE = Union[ChatOptions, dict[str, Any]]


__all__ = [
    "OpenAISettings",
]


def _check_openai_version_for_callable_api_key() -> None:
    """OpenAIのバージョンが呼び出し可能なAPIキーをサポートしているか確認します。

    呼び出し可能なAPIキーはOpenAI >= 1.106.0が必要です。
    バージョンが古すぎる場合は、役立つメッセージ付きでServiceInitializationErrorを発生させます。

    """
    try:
        current_version = version.parse(openai.__version__)
        min_required_version = version.parse("1.106.0")

        if current_version < min_required_version:
            raise ServiceInitializationError(
                f"Callable API keys require OpenAI SDK >= 1.106.0, but you have {openai.__version__}. "
                f"Please upgrade with 'pip install openai>=1.106.0' or provide a string API key instead. "
                f"Note: If you're using mem0ai, you may need to upgrade to mem0ai>=0.1.118 "
                f"to allow newer OpenAI versions."
            )
    except ServiceInitializationError:
        raise  # 自分たちの例外を再スローします
    except Exception as e:
        logger.warning(f"Could not check OpenAI version for callable API key support: {e}")


class OpenAISettings(AFBaseSettings):
    """OpenAIの環境設定。

    設定はまず 'OPENAI_' プレフィックスの環境変数から読み込まれます。
    環境変数が見つからない場合は、エンコーディング 'utf-8' の .env ファイルから設定を読み込むことができます。
    .env ファイルにも設定が見つからない場合は無視されますが、検証は失敗し設定が不足していることを通知します。

    キーワード引数:
        api_key: OpenAI APIキー。詳細は https://platform.openai.com/account/api-keys を参照。
            環境変数 OPENAI_API_KEY で設定可能。
        base_url: OpenAI APIのベースURL。
            環境変数 OPENAI_BASE_URL で設定可能。
        org_id: 通常はオプションですが、アカウントが複数の組織に属する場合は必要です。
            環境変数 OPENAI_ORG_ID で設定可能。
        chat_model_id: 使用するOpenAIチャットモデルID。例: gpt-3.5-turbo や gpt-4。
            環境変数 OPENAI_CHAT_MODEL_ID で設定可能。
        responses_model_id: 使用するOpenAIレスポンスモデルID。例: gpt-4o や o1。
            環境変数 OPENAI_RESPONSES_MODEL_ID で設定可能。
        env_file_path: 設定を読み込む .env ファイルのパス。
        env_file_encoding: .env ファイルのエンコーディング。デフォルトは 'utf-8'。

    Examples:
        .. code-block:: python

            from agent_framework.openai import OpenAISettings

            # 環境変数を使用する場合
            # OPENAI_API_KEY=sk-... を設定
            # OPENAI_CHAT_MODEL_ID=gpt-4 を設定
            settings = OpenAISettings()

            # またはパラメータを直接渡す場合
            settings = OpenAISettings(api_key="sk-...", chat_model_id="gpt-4")

            # または .env ファイルから読み込む場合
            settings = OpenAISettings(env_file_path="path/to/.env")

    """

    env_prefix: ClassVar[str] = "OPENAI_"

    api_key: SecretStr | Callable[[], str | Awaitable[str]] | None = None
    base_url: str | None = None
    org_id: str | None = None
    chat_model_id: str | None = None
    responses_model_id: str | None = None


class OpenAIBase(SerializationMixin):
    """OpenAIクライアントの基底クラスです。"""

    INJECTABLE: ClassVar[set[str]] = {"client"}

    def __init__(self, *, client: AsyncOpenAI, model_id: str, **kwargs: Any) -> None:
        """OpenAIBaseを初期化します。

        キーワード引数:
            client: AsyncOpenAIクライアントのインスタンス。
            model_id: 使用するAIモデルID（空でなく、空白は削除されます）。
            **kwargs: 追加のキーワード引数。

        """
        if not model_id or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self.client = client
        self.model_id = model_id.strip()

        # super().__init__() を呼び出してMROチェーンを継続します（例: BaseChatClient）
        # 他の基底クラスに属する既知のkwargsを抽出します
        additional_properties = kwargs.pop("additional_properties", None)
        middleware = kwargs.pop("middleware", None)
        instruction_role = kwargs.pop("instruction_role", None)

        # super().__init__() の引数を構築します
        super_kwargs = {}
        if additional_properties is not None:
            super_kwargs["additional_properties"] = additional_properties
        if middleware is not None:
            super_kwargs["middleware"] = middleware

        # フィルタリングされたkwargsで super().__init__() を呼び出します
        super().__init__(**super_kwargs)

        # instruction_role と残りの kwargs をインスタンス属性として保存します
        if instruction_role is not None:
            self.instruction_role = instruction_role
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_api_key(
        self, api_key: str | SecretStr | Callable[[], str | Awaitable[str]] | None
    ) -> str | Callable[[], str | Awaitable[str]] | None:
        """クライアント初期化に適切なAPIキーの値を取得します。

        引数:
            api_key: 文字列、SecretStr、呼び出し可能、または None のAPIキー引数。

        戻り値:
            呼び出し可能なAPIキーの場合はそのまま呼び出し可能を返します。
            SecretStrの場合は文字列の値を返します。
            文字列またはNoneの場合はそのまま返します。

        """
        if isinstance(api_key, SecretStr):
            return api_key.get_secret_value()

        # 呼び出し可能なAPIキーのバージョン互換性をチェックします
        if callable(api_key):
            _check_openai_version_for_callable_api_key()

        return api_key  # 呼び出し可能、文字列、またはNoneをOpenAI SDKに直接渡します


class OpenAIConfigMixin(OpenAIBase):
    """OpenAIサービスへの接続を設定する内部クラスです。"""

    OTEL_PROVIDER_NAME: ClassVar[str] = "openai"  # type: ignore[reportIncompatibleVariableOverride, misc]

    def __init__(
        self,
        model_id: str,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """OpenAIサービス用のクライアントを初期化します。

        このコンストラクタはOpenAIのAPIと対話するクライアントを設定し、
        チャットやテキスト補完などの異なるタイプのAIモデルとのやり取りを可能にします。

        引数:
            model_id: OpenAIモデルの識別子。空であってはなりません。
                既定値があります。
            api_key: 認証用のOpenAI APIキー、またはAPIキーを返す呼び出し可能。
                空であってはなりません。（オプション）
            org_id: OpenAIの組織ID。通常はオプションですが、アカウントが複数の組織に属する場合は必要です。
            default_headers: HTTPリクエスト用のデフォルトヘッダー。（オプション）
            client: 既存のOpenAIクライアント。（オプション）
            instruction_role: 'instruction'メッセージに使用するロール。例として要約プロンプトは `developer` や `system` を使うことがあります。（オプション）
            base_url: 使用するベースURL。指定した場合はOpenAIコネクタの標準値を上書きします。
                カスタムクライアントを指定した場合は使用されません。
            kwargs: 追加のキーワード引数。


        """
        # APP_INFOが存在する場合、ヘッダーにマージします
        merged_headers = dict(copy(default_headers)) if default_headers else {}
        if APP_INFO:
            merged_headers.update(APP_INFO)
            merged_headers = prepend_agent_framework_to_user_agent(merged_headers)

        # 基底クラスのメソッドを使って呼び出し可能なAPIキーを処理します
        api_key_value = self._get_api_key(api_key)

        if not client:
            if not api_key:
                raise ServiceInitializationError("Please provide an api_key")
            args: dict[str, Any] = {"api_key": api_key_value, "default_headers": merged_headers}
            if org_id:
                args["organization"] = org_id
            if base_url:
                args["base_url"] = base_url
            client = AsyncOpenAI(**args)

        # シリアライズのために設定をインスタンス属性として保存します
        self.org_id = org_id
        self.base_url = str(base_url)
        # default_headersを保存しますが、シリアライズのためにUSER_AGENT_KEYは除外します
        if default_headers:
            self.default_headers: dict[str, Any] | None = {
                k: v for k, v in default_headers.items() if k != USER_AGENT_KEY
            }
        else:
            self.default_headers = None

        args = {
            "model_id": model_id,
            "client": client,
        }
        if instruction_role:
            args["instruction_role"] = instruction_role

        # additional_properties と middleware を kwargs 経由で BaseChatClient に渡すことを保証します
        # これらは BaseChatClient.__init__ で kwargs 経由で消費されます
        super().__init__(**args, **kwargs)
