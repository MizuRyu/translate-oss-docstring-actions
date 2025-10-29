# Copyright (c) Microsoft. All rights reserved.
"""Semantic KernelとAgent Frameworkの両方を使用してAzure AI Agentを作成します。

前提条件:
- 展開済みモデルを持つAzure AI Agentリソース。
- AzureCliCredentialがサポートするログイン済みのAzure CLIまたはその他の認証情報。
"""

import asyncio


async def run_semantic_kernel() -> None:
    from azure.identity.aio import AzureCliCredential
    from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings

    async with AzureCliCredential() as credential:
        async with AzureAIAgent.create_client(credential=credential) as client:
            settings = AzureAIAgentSettings()  # リージョン/デプロイメントのための環境変数を読み取ります。
            # SKはリモートAgentの定義を構築し、それをAzureAIAgentでラップします。
            definition = await client.agents.create_agent(
                model=settings.model_deployment_name,
                name="Support",
                instructions="Answer customer questions in one paragraph.",
            )
            agent = AzureAIAgent(client=client, definition=definition)
            response = await agent.get_response("How do I upgrade my plan?")
            print("[SK]", response.message.content)


async def run_agent_framework() -> None:
    from azure.identity.aio import AzureCliCredential
    from agent_framework.azure import AzureAIAgentClient

    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential).create_agent(
            name="Support",
            instructions="Answer customer questions in one paragraph.",
        ) as agent:
            # AFクライアントはリモートAgent用の非同期コンテキストマネージャを返します。
            reply = await agent.run("How do I upgrade my plan?")
            print("[AF]", reply.text)


async def main() -> None:
    await run_semantic_kernel()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
