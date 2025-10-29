# Copyright (c) Microsoft. All rights reserved.
"""Purviewポリシー適用サンプル（Python）。

以下を示します:
1. 基本的なchat agentの作成
2. AGENT middleware（agentレベル）によるPurviewポリシー評価の追加
3. CHAT middleware（chat-clientレベル）によるPurviewポリシー評価の追加
4. スレッド会話の実行と結果の表示

環境変数:
- AZURE_OPENAI_ENDPOINT (必須)
- AZURE_OPENAI_DEPLOYMENT_NAME (任意、デフォルトは gpt-4o-mini)
- PURVIEW_CLIENT_APP_ID (必須)
- PURVIEW_USE_CERT_AUTH (任意、証明書認証を使用する場合は "true" に設定)
- PURVIEW_TENANT_ID (証明書認証時に必須)
- PURVIEW_CERT_PATH (証明書認証時に必須)
- PURVIEW_CERT_PASSWORD (任意)
- PURVIEW_DEFAULT_USER_ID (任意、Purview評価用のユーザーID)
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from agent_framework import AgentRunResponse, ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import (
    AzureCliCredential,
    CertificateCredential,
    InteractiveBrowserCredential,
)

# Purview統合の構成要素
from agent_framework.microsoft import (
    PurviewPolicyMiddleware,
    PurviewChatPolicyMiddleware,
    PurviewSettings,
)

JOKER_NAME = "Joker"
JOKER_INSTRUCTIONS = "You are good at telling jokes. Keep responses concise."


def _get_env(name: str, *, required: bool = True, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"Environment variable {name} is required")
    return val  # type: ignore[return-value]


def build_credential() -> Any:
    """Purview認証用のAzure credentialを選択します。

    サポートされているモード:
    1. CertificateCredential (PURVIEW_USE_CERT_AUTH=trueの場合)
    2. InteractiveBrowserCredential (PURVIEW_CLIENT_APP_IDが必要)

    """
    client_id = _get_env("PURVIEW_CLIENT_APP_ID", required=True)
    use_cert_auth = _get_env("PURVIEW_USE_CERT_AUTH", required=False, default="false").lower() == "true"

    if not client_id:
        raise RuntimeError(
            "PURVIEW_CLIENT_APP_ID is required for interactive browser authentication; "
            "set PURVIEW_USE_CERT_AUTH=true for certificate mode instead."
        )

    if use_cert_auth:
        tenant_id = _get_env("PURVIEW_TENANT_ID")
        cert_path = _get_env("PURVIEW_CERT_PATH")
        cert_password = _get_env("PURVIEW_CERT_PASSWORD", required=False, default=None)
        print(f"Using Certificate Authentication (tenant: {tenant_id}, cert: {cert_path})")
        return CertificateCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            certificate_path=cert_path,
            password=cert_password,
        )

    print(f"Using Interactive Browser Authentication (client_id: {client_id})")
    return InteractiveBrowserCredential(client_id=client_id)


async def run_with_agent_middleware() -> None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        print("Skipping run: AZURE_OPENAI_ENDPOINT not set")
        return

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    user_id = os.environ.get("PURVIEW_DEFAULT_USER_ID")
    chat_client = AzureOpenAIChatClient(deployment_name=deployment, endpoint=endpoint, credential=AzureCliCredential())

    purview_agent_middleware = PurviewPolicyMiddleware(
        build_credential(),
        PurviewSettings(
            app_name="Agent Framework Sample App",
        ),
    )

    agent = ChatAgent(
        chat_client=chat_client,
        instructions=JOKER_INSTRUCTIONS,
        name=JOKER_NAME,
        middleware=purview_agent_middleware,
    )

    print("-- Agent Middleware Path --")
    first: AgentRunResponse = await agent.run(ChatMessage(role=Role.USER, text="Tell me a joke about a pirate.", additional_properties={"user_id": user_id}))
    print("First response (agent middleware):\n", first)

    second: AgentRunResponse = await agent.run(ChatMessage(role=Role.USER, text="That was funny. Tell me another one.", additional_properties={"user_id": user_id}))
    print("Second response (agent middleware):\n", second)


async def run_with_chat_middleware() -> None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        print("Skipping chat middleware run: AZURE_OPENAI_ENDPOINT not set")
        return

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", default="gpt-4o-mini")
    user_id = os.environ.get("PURVIEW_DEFAULT_USER_ID")
    
    chat_client = AzureOpenAIChatClient(
        deployment_name=deployment,
        endpoint=endpoint,
        credential=AzureCliCredential(),
        middleware=[
            PurviewChatPolicyMiddleware(
                build_credential(),
                PurviewSettings(
                    app_name="Agent Framework Sample App (Chat)",
                ),
            )
        ],
    )

    agent = ChatAgent(
        chat_client=chat_client,
        instructions=JOKER_INSTRUCTIONS,
        name=JOKER_NAME,
    )

    print("-- Chat Middleware Path --")
    first: AgentRunResponse = await agent.run(
        ChatMessage(
            role=Role.USER,
            text="Give me a short clean joke.",
            additional_properties={"user_id": user_id},
        )
    )
    print("First response (chat middleware):\n", first)

    second: AgentRunResponse = await agent.run(
        ChatMessage(
            role=Role.USER,
            text="One more please.",
            additional_properties={"user_id": user_id},
        )
    )
    print("Second response (chat middleware):\n", second)


async def main() -> None:
    print("== Purview Agent Sample (Agent & Chat Middleware) ==")
    try:
        await run_with_agent_middleware()
    except Exception as ex:  # pragma: no cover - demo resilience
        print(f"Agent middleware path failed: {ex}")

    try:
        await run_with_chat_middleware()
    except Exception as ex:  # pragma: no cover - demo resilience
        print(f"Chat middleware path failed: {ex}")


if __name__ == "__main__":
    asyncio.run(main())
