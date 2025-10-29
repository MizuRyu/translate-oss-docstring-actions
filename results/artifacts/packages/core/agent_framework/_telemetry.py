# Copyright (c) Microsoft. All rights reserved.

import os
from typing import Any, Final

from . import __version__ as version_info
from ._logging import get_logger

logger = get_logger()

__all__ = [
    "AGENT_FRAMEWORK_USER_AGENT",
    "APP_INFO",
    "USER_AGENT_KEY",
    "USER_AGENT_TELEMETRY_DISABLED_ENV_VAR",
    "prepend_agent_framework_to_user_agent",
]

# この環境変数が存在しない場合、ユーザーエージェントのテレメトリが有効になることに注意してください。
USER_AGENT_TELEMETRY_DISABLED_ENV_VAR = "AGENT_FRAMEWORK_USER_AGENT_DISABLED"
IS_TELEMETRY_ENABLED = os.environ.get(USER_AGENT_TELEMETRY_DISABLED_ENV_VAR, "false").lower() not in ["true", "1"]

APP_INFO = (
    {
        "agent-framework-version": f"python/{version_info}",  # type: ignore[has-type]
    }
    if IS_TELEMETRY_ENABLED
    else None
)
USER_AGENT_KEY: Final[str] = "User-Agent"
HTTP_USER_AGENT: Final[str] = "agent-framework-python"
AGENT_FRAMEWORK_USER_AGENT = f"{HTTP_USER_AGENT}/{version_info}"  # type: ignore[has-type]


def prepend_agent_framework_to_user_agent(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    """ヘッダーのUser-Agentに "agent-framework" を先頭に追加します。

    環境変数 ``AGENT_FRAMEWORK_USER_AGENT_DISABLED`` によってユーザーエージェントのテレメトリが無効化されている場合、
    User-Agentヘッダーにはagent-framework情報が含まれません。
    Noneが渡された場合は空の辞書として返されるか、そのまま送信されます。

    Args:
        headers: 既存のヘッダー辞書。

    Returns:
        headersがNoneの場合は "User-Agent" に "agent-framework-python/{version}" を設定した新しい辞書。
        既存のUser-Agentに "agent-framework-python/{version}" を先頭に追加した修正済みヘッダー辞書。

    Examples:
        .. code-block:: python

            from agent_framework import prepend_agent_framework_to_user_agent

            # 新しいヘッダーにagent-frameworkを追加
            headers = prepend_agent_framework_to_user_agent()
            print(headers["User-Agent"])  # "agent-framework-python/0.1.0"

            # 既存のヘッダーに先頭追加
            existing = {"User-Agent": "my-app/1.0"}
            headers = prepend_agent_framework_to_user_agent(existing)
            print(headers["User-Agent"])  # "agent-framework-python/0.1.0 my-app/1.0"

    """
    if not IS_TELEMETRY_ENABLED:
        return headers or {}
    if not headers:
        return {USER_AGENT_KEY: AGENT_FRAMEWORK_USER_AGENT}
    headers[USER_AGENT_KEY] = (
        f"{AGENT_FRAMEWORK_USER_AGENT} {headers[USER_AGENT_KEY]}"
        if USER_AGENT_KEY in headers
        else AGENT_FRAMEWORK_USER_AGENT
    )

    return headers
