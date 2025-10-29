# Copyright (c) Microsoft. All rights reserved.
"""SKとAgent Frameworkを使って基本的なResponses API呼び出しを行います。"""

import asyncio


async def run_semantic_kernel() -> None:
    from azure.identity import AzureCliCredential
    from semantic_kernel.agents import AzureResponsesAgent
    from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

    credential = AzureCliCredential()
    try:
        client = AzureResponsesAgent.create_client(credential=credential)
        # SKのレスポンスAgentはAzure OpenAIのホストされたResponses APIをラップします。
        agent = AzureResponsesAgent(
            ai_model_id=AzureOpenAISettings().responses_deployment_name,
            client=client,
            instructions="Answer in one concise sentence.",
            name="Expert",
        )
        response = await agent.get_response("Why is the sky blue?")
        print("[SK]", response.message.content)
    finally:
        await credential.close()


async def run_agent_framework() -> None:
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIResponsesClient

    # AFのChatAgentはOpenAIResponsesClientを直接差し替え可能です。
    chat_agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="Answer in one concise sentence.",
        name="Expert",
    )
    reply = await chat_agent.run("Why is the sky blue?")
    print("[AF]", reply.text)


async def main() -> None:
    await run_semantic_kernel()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
