# Copyright (c) Microsoft. All rights reserved.
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture


@fixture
def exclude_list(request: Any) -> list[str]:
    """除外する環境変数のリストを返すフィクスチャ。"""
    return request.param if hasattr(request, "param") else []


@fixture
def override_env_param_dict(request: Any) -> dict[str, str]:
    """上書きする環境変数の辞書を返すフィクスチャ。"""
    return request.param if hasattr(request, "param") else {}


@fixture()
def azure_ai_unit_test_env(monkeypatch, exclude_list, override_env_param_dict):  # type: ignore
    """AzureAISettings 用の環境変数を設定するフィクスチャ。"""

    if exclude_list is None:
        exclude_list = []

    if override_env_param_dict is None:
        override_env_param_dict = {}

    env_vars = {
        "AZURE_AI_PROJECT_ENDPOINT": "https://test-project.cognitiveservices.azure.com/",
        "AZURE_AI_MODEL_DEPLOYMENT_NAME": "test-gpt-4o",
    }

    env_vars.update(override_env_param_dict)  # type: ignore

    for key, value in env_vars.items():
        if key in exclude_list:
            monkeypatch.delenv(key, raising=False)  # type: ignore
            continue
        monkeypatch.setenv(key, value)  # type: ignore

    return env_vars


@fixture
def mock_ai_project_client() -> MagicMock:
    """モック AIProjectClient を提供するフィクスチャ。"""
    mock_client = MagicMock()

    # モック agents プロパティ。
    mock_client.agents = MagicMock()
    mock_client.agents.create_agent = AsyncMock()
    mock_client.agents.delete_agent = AsyncMock()

    # モック agent 作成レスポンス。
    mock_agent = MagicMock()
    mock_agent.id = "test-agent-id"
    mock_client.agents.create_agent.return_value = mock_agent

    # モック threads プロパティ。
    mock_client.agents.threads = MagicMock()
    mock_client.agents.threads.create = AsyncMock()
    mock_client.agents.messages.create = AsyncMock()

    # モック runs プロパティ。
    mock_client.agents.runs = MagicMock()
    mock_client.agents.runs.list = AsyncMock()
    mock_client.agents.runs.cancel = AsyncMock()
    mock_client.agents.runs.stream = AsyncMock()
    mock_client.agents.runs.submit_tool_outputs_stream = AsyncMock()

    return mock_client


@fixture
def mock_azure_credential() -> MagicMock:
    """モック AsyncTokenCredential を提供するフィクスチャ。"""
    return MagicMock()
