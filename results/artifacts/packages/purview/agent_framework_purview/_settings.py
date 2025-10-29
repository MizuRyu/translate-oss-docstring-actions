# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from enum import Enum

from agent_framework._pydantic import AFBaseSettings
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class PurviewLocationType(str, Enum):
    """Purview ポリシー評価のための場所のタイプ。"""

    APPLICATION = "application"
    URI = "uri"
    DOMAIN = "domain"


class PurviewAppLocation(BaseModel):
    """Purview ポリシー評価のためのアプリの場所を表す識別子。"""

    location_type: PurviewLocationType = Field(..., description="The location type.")
    location_value: str = Field(..., description="The location value.")

    def get_policy_location(self) -> dict[str, str]:
        ns = "microsoft.graph"
        if self.location_type == PurviewLocationType.APPLICATION:
            dt = f"{ns}.policyLocationApplication"
        elif self.location_type == PurviewLocationType.URI:
            dt = f"{ns}.policyLocationUrl"
        elif self.location_type == PurviewLocationType.DOMAIN:
            dt = f"{ns}.policyLocationDomain"
        else:  # pragma: no cover - defensive
            raise ValueError("Invalid Purview location type")
        return {"@odata.type": dt, "value": self.location_value}


class PurviewSettings(AFBaseSettings):
    """Purview 統合の設定で、.NET PurviewSettings をミラーリングしています。

    Attributes:
        app_name: 公開アプリ名。
        tenant_id: リクエストを行うユーザーのオプションのテナントID（guid）。
        purview_app_location: ポリシー評価のためのオプションのアプリ場所。
        graph_base_uri: Microsoft Graph のベースURI。
        blocked_prompt_message: ポリシーによりプロンプトがブロックされた場合に返すカスタムメッセージ。
        blocked_response_message: ポリシーによりレスポンスがブロックされた場合に返すカスタムメッセージ。

    """

    app_name: str = Field(...)
    tenant_id: str | None = Field(default=None)
    purview_app_location: PurviewAppLocation | None = Field(default=None)
    graph_base_uri: str = Field(default="https://graph.microsoft.com/v1.0/")
    process_inline: bool = Field(default=False, description="Process content inline if supported.")
    blocked_prompt_message: str = Field(
        default="Prompt blocked by policy",
        description="Message to return when a prompt is blocked by policy.",
    )
    blocked_response_message: str = Field(
        default="Response blocked by policy",
        description="Message to return when a response is blocked by policy.",
    )

    model_config = SettingsConfigDict(populate_by_name=True, validate_assignment=True)

    def get_scopes(self) -> list[str]:
        from urllib.parse import urlparse

        host = urlparse(self.graph_base_uri).hostname or "graph.microsoft.com"
        return [f"https://{host}/.default"]
