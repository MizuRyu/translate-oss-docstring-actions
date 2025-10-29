# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential

"""
Azure AI Agent with Existing Agent Example

This sample demonstrates working with pre-existing Azure AI Agents by providing
agent IDs, showing agent reuse patterns for production scenarios.
"""


async def main() -> None:
    print("=== Azure AI Chat Client with Existing Agent ===")

    # クライアントを作成する
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential) as client,
    ):
        azure_ai_agent = await client.agents.create_agent(
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            # デフォルトの指示でリモートエージェントを作成する これらの指示は作成されたエージェントに対して毎回保持されます。
            instructions="End each response with [END].",
        )

        chat_client = AzureAIAgentClient(project_client=client, agent_id=azure_ai_agent.id)

        try:
            async with ChatAgent(
                chat_client=chat_client,
                # ここでの指示はこの ChatAgent インスタンスにのみ適用されます これらの指示は既存のリモートエージェントの指示と組み合わされます。
                # 実行時の最終的な指示は次のようになります: "'End each response with [END]. Respond with
                # 'Hello World' only'"
                instructions="Respond with 'Hello World' only",
            ) as agent:
                query = "How are you?"
                print(f"User: {query}")
                result = await agent.run(query)
                # ローカルとリモートの指示に基づき、結果は 'Hello World [END]' になります。
                print(f"Agent: {result}\n")
        finally:
            # エージェントを手動でクリーンアップする
            await client.agents.delete_agent(azure_ai_agent.id)


if __name__ == "__main__":
    asyncio.run(main())
