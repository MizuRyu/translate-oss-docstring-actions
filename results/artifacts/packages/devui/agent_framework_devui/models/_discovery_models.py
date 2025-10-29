# Copyright (c) Microsoft. All rights reserved.

"""エンティティ情報のためのDiscovery APIモデル。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EnvVarRequirement(BaseModel):
    """エンティティの環境変数要件。"""

    name: str
    description: str
    required: bool = True
    example: str | None = None


class EntityInfo(BaseModel):
    """discoveryおよび詳細ビューのためのエンティティ情報。"""

    # 常に存在（コアエンティティデータ）
    id: str
    type: str  # "agent"、"workflow"
    name: str
    description: str | None = None
    framework: str
    tools: list[str | dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ソース情報
    source: str = "directory"  # "directory"または"in_memory"

    # 環境変数の要件
    required_env_vars: list[EnvVarRequirement] | None = None

    # エージェント固有のフィールド（オプション、利用可能な場合に設定）
    instructions: str | None = None
    model_id: str | None = None
    chat_client_type: str | None = None
    context_providers: list[str] | None = None
    middleware: list[str] | None = None

    # workflow固有のフィールド（詳細情報リクエスト時のみ設定）
    executors: list[str] | None = None
    workflow_dump: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
    input_type_name: str | None = None
    start_executor_id: str | None = None


class DiscoveryResponse(BaseModel):
    """エンティティdiscoveryのレスポンスモデル。"""

    entities: list[EntityInfo] = Field(default_factory=list)
