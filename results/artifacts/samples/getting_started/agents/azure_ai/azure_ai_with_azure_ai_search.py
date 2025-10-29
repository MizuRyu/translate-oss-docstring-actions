# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, HostedFileSearchTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

"""
Azure AI Agent with Azure AI Search Example

This sample demonstrates how to create an Azure AI agent that uses Azure AI Search
to search through indexed hotel data and answer user questions about hotels.

Prerequisites:
1. Set AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME environment variables
2. Ensure you have an Azure AI Search connection configured in your Azure AI project
3. The search index "hotels-sample-index" should exist in your Azure AI Search service
   (you can create this using the Azure portal with sample hotel data)

Environment variables:
- AZURE_AI_PROJECT_ENDPOINT: Your Azure AI project endpoint
- AZURE_AI_MODEL_DEPLOYMENT_NAME: The name of your model deployment
"""

# Azure AI Search が hotels-sample-index で動作していることを検証するテストクエリ。
USER_INPUTS = [
    "Search the hotel database for Stay-Kay City Hotel and give me detailed information.",
]


async def main() -> None:
    """Azure AI Search 機能を持つ Azure AI エージェントを示すメイン関数。"""

    # 1. HostedFileSearchTool を使って Azure AI Search ツールを作成する このツールはプロジェクトのデフォルト Azure AI
    # Search 接続を自動的に使用します。
    azure_ai_search_tool = HostedFileSearchTool(
        additional_properties={
            "index_name": "hotels-sample-index",  # Name of your search index
            "query_type": "simple",  # Use simple search
            "top_k": 10,  # Get more comprehensive results
        },
    )

    # 2. AzureAIAgentClient を非同期コンテキストマネージャーとして使用し、自動クリーンアップを行う
    async with (
        AzureAIAgentClient(async_credential=AzureCliCredential()) as client,
        ChatAgent(
            chat_client=client,
            name="HotelSearchAgent",
            instructions=("You are a helpful travel assistant that searches hotel information."),
            tools=azure_ai_search_tool,
        ) as agent,
    ):
        print("=== Azure AI Agent with Azure AI Search ===")
        print("This agent can search through hotel data to help you find accommodations.\n")

        # 3. エージェントとの会話をシミュレートする
        for user_input in USER_INPUTS:
            print(f"User: {user_input}")
            print("Agent: ", end="", flush=True)

            # より良いユーザー体験のためにレスポンスをストリームする
            async for chunk in agent.run_stream(user_input):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print("\n" + "=" * 50 + "\n")

        print("Hotel search conversation completed!")


if __name__ == "__main__":
    asyncio.run(main())
