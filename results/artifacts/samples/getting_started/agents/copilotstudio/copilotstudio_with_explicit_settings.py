# 著作権 (c) Microsoft。無断転載を禁じます。

import asyncio
import os

from agent_framework.microsoft import CopilotStudioAgent, acquire_token
from microsoft_agents.copilotstudio.client import AgentType, ConnectionSettings, CopilotClient, PowerPlatformCloud

"""
Copilot Studio Agent with Explicit Settings Example

This sample demonstrates explicit configuration of CopilotStudioAgent with manual
token management and custom ConnectionSettings for production environments.
"""

# 必要な環境変数: COPILOTSTUDIOAGENT__ENVIRONMENTID - Copilotがデプロイされている環境ID
# COPILOTSTUDIOAGENT__SCHEMANAME - CopilotのAgent識別子/スキーマ名 COPILOTSTUDIOAGENT__AGENTAPPID
# - 認証用クライアントID COPILOTSTUDIOAGENT__TENANTID - 認証用テナントID


async def example_with_connection_settings() -> None:
    """明示的なConnectionSettingsとCopilotClientを使用した例。"""
    print("=== Copilot Studio Agent with Connection Settings ===")

    # 環境変数からの設定
    environment_id = os.environ["COPILOTSTUDIOAGENT__ENVIRONMENTID"]
    agent_identifier = os.environ["COPILOTSTUDIOAGENT__SCHEMANAME"]
    client_id = os.environ["COPILOTSTUDIOAGENT__AGENTAPPID"]
    tenant_id = os.environ["COPILOTSTUDIOAGENT__TENANTID"]

    # acquire_token関数を使ってトークンを取得します
    token = acquire_token(
        client_id=client_id,
        tenant_id=tenant_id,
    )

    # 接続設定を作成します
    settings = ConnectionSettings(
        environment_id=environment_id,
        agent_identifier=agent_identifier,
        cloud=PowerPlatformCloud.PROD,  # Or PowerPlatformCloud.GOV, PowerPlatformCloud.HIGH, etc.
        copilot_agent_type=AgentType.PUBLISHED,  # Or AgentType.PREBUILT
        custom_power_platform_cloud=None,  # Optional: for custom cloud endpoints
    )

    # 明示的な設定でCopilotClientを作成します
    client = CopilotClient(settings=settings, token=token)

    # 明示的なクライアントでAgentを作成します
    agent = CopilotStudioAgent(client=client)

    # シンプルなクエリを実行します
    query = "What is the capital of Italy?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Agent: {result}")


async def example_with_explicit_parameters() -> None:
    """すべてのパラメーターを明示的に指定したCopilotStudioAgentの例です。"""
    print("\n=== Copilot Studio Agent with All Explicit Parameters ===")

    # 環境変数からの設定
    environment_id = os.environ["COPILOTSTUDIOAGENT__ENVIRONMENTID"]
    agent_identifier = os.environ["COPILOTSTUDIOAGENT__SCHEMANAME"]
    client_id = os.environ["COPILOTSTUDIOAGENT__AGENTAPPID"]
    tenant_id = os.environ["COPILOTSTUDIOAGENT__TENANTID"]

    # すべてのパラメーターを明示的に指定してAgentを作成します
    agent = CopilotStudioAgent(
        environment_id=environment_id,
        agent_identifier=agent_identifier,
        client_id=client_id,
        tenant_id=tenant_id,
        cloud=PowerPlatformCloud.PROD,
        agent_type=AgentType.PUBLISHED,
    )

    # シンプルなクエリを実行します
    query = "What is the capital of Japan?"
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Agent: {result}")


async def main() -> None:
    await example_with_connection_settings()
    await example_with_explicit_parameters()


if __name__ == "__main__":
    asyncio.run(main())
