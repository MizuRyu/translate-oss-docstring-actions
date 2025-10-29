# 著作権 (c) Microsoft。無断転載を禁じます。

import os
import asyncio

from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

"""
Azure OpenAI Responses Client with local Model Context Protocol (MCP) Example

This sample demonstrates integration of Azure OpenAI Responses Client with local Model Context Protocol (MCP)
servers.
"""


# --- 以下のコードはStreamable HTTP経由でMicrosoft Learn MCPサーバーを使用します --- ---
# ユーザーはこれらの環境変数を設定するか、以下の値を希望のローカルMCPサーバーに編集できます
MCP_NAME = os.environ.get("MCP_NAME", "Microsoft Learn MCP")  # 例の名前
MCP_URL = os.environ.get("MCP_URL", "https://learn.microsoft.com/api/mcp")   # 例のエンドポイント

# Azure OpenAI Responses認証用の環境変数 AZURE_OPENAI_ENDPOINT="<your-azure openai-endpoint>"
# AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME="<your-deployment-name>"
# AZURE_OPENAI_API_VERSION="<your-api-version>"  # 例: "2025-03-01-preview"

async def main():
    """Azure OpenAI Responses Agent用のローカルMCPツールの例です。"""
    # 認証: Azure CLIを使用
    credential = AzureCliCredential()

    # Azure OpenAI ResponsesをバックエンドにしたAgentを構築します
    # （endpoint/deployment/api_versionは上記の環境変数からも取得可能）
    responses_client = AzureOpenAIResponsesClient(
        credential=credential,
    )

    agent: ChatAgent = responses_client.create_agent(
        name="DocsAgent",
        instructions=(
            "You are a helpful assistant that can help with Microsoft documentation questions."
        ),
    )

    # MCPサーバーに接続します（Streamable HTTP）
    async with MCPStreamableHTTPTool(
        name=MCP_NAME,
        url=MCP_URL,
        
    ) as mcp_tool:
        # 最初のクエリ — 助けになる場合はAgentがMCPツールを使用することを期待します
        q1 = "How to create an Azure storage account using az cli?"
        r1 = await agent.run(q1, tools=mcp_tool)
        print("\n=== Answer 1 ===\n", r1.text)

        # フォローアップクエリ（接続は再利用されます）
        q2 = "What is Microsoft Agent Framework?"
        r2 = await agent.run(q2, tools=mcp_tool)
        print("\n=== Answer 2 ===\n", r2.text)


if __name__ == "__main__":
    asyncio.run(main())
