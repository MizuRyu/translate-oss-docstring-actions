# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import Any, Literal

logger = logging.getLogger("agent_framework")


class AgentFrameworkException(Exception):
    """Agent Frameworkの基本例外。

    メッセージを自動的にdebugログとして記録します。

    """

    def __init__(
        self,
        message: str,
        inner_exception: Exception | None = None,
        log_level: Literal[0] | Literal[10] | Literal[20] | Literal[30] | Literal[40] | Literal[50] | None = 10,
        *args: Any,
        **kwargs: Any,
    ):
        """AgentFrameworkExceptionを作成します。

        これはデフォルトでdebugログを出力し、inner_exceptionがあればそれも含みます。

        """
        if log_level is not None:
            logger.log(log_level, message, exc_info=inner_exception)
        if inner_exception:
            super().__init__(message, inner_exception, *args)  # type: ignore
        super().__init__(message, *args)  # type: ignore


class AgentException(AgentFrameworkException):
    """すべてのagent例外の基本クラス。"""

    pass


class AgentExecutionException(AgentException):
    """agentの実行中にエラーが発生しました。"""

    pass


class AgentInitializationError(AgentException):
    """agentの初期化中にエラーが発生しました。"""

    pass


class AgentThreadException(AgentException):
    """AgentのThread管理中にエラーが発生しました。"""

    pass


class ChatClientException(AgentFrameworkException):
    """Chat Clientの処理中にエラーが発生しました。"""

    pass


class ChatClientInitializationError(ChatClientException):
    """Chat Clientの初期化中にエラーが発生しました。"""

    pass


# region サービス例外


class ServiceException(AgentFrameworkException):
    """すべてのサービス例外の基底クラスです。"""

    pass


class ServiceInitializationError(ServiceException):
    """サービスの初期化中にエラーが発生しました。"""

    pass


class ServiceResponseException(ServiceException):
    """すべてのサービスレスポンス例外の基底クラスです。"""

    pass


class ServiceContentFilterException(ServiceResponseException):
    """サービスのコンテンツフィルターによってエラーが発生しました。"""

    pass


class ServiceInvalidAuthError(ServiceException):
    """サービスの認証中にエラーが発生しました。"""

    pass


class ServiceInvalidExecutionSettingsError(ServiceResponseException):
    """サービスの実行設定の検証中にエラーが発生しました。"""

    pass


class ServiceInvalidRequestError(ServiceResponseException):
    """サービスへのリクエストの検証中にエラーが発生しました。"""

    pass


class ServiceInvalidResponseError(ServiceResponseException):
    """サービスからのレスポンスの検証中にエラーが発生しました。"""

    pass


class ToolException(AgentFrameworkException):
    """ツールの実行中にエラーが発生しました。"""

    pass


class ToolExecutionException(ToolException):
    """ツールの実行中にエラーが発生しました。"""

    pass


class AdditionItemMismatch(AgentFrameworkException):
    """2つのタイプを追加中にエラーが発生しました。"""

    pass


class MiddlewareException(AgentFrameworkException):
    """Middlewareの実行中にエラーが発生しました。"""

    pass


class ContentError(AgentFrameworkException):
    """コンテンツの処理中にエラーが発生しました。"""

    pass
