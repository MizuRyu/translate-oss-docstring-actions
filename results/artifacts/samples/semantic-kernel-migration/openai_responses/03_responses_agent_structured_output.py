# Copyright (c) Microsoft. All rights reserved.
"""SKとAFでResponses APIから構造化JSON出力を要求します。"""

import asyncio

from pydantic import BaseModel


class ReleaseBrief(BaseModel):
    feature: str
    benefit: str
    launch_date: str


async def run_semantic_kernel() -> None:
    from azure.identity import AzureCliCredential
    from semantic_kernel.agents import AzureResponsesAgent
    from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

    credential = AzureCliCredential()
    try:
        client = AzureResponsesAgent.create_client(credential=credential)
        # response_formatはモデルからスキーマ制約された出力を要求します。
        agent = AzureResponsesAgent(
            ai_model_id=AzureOpenAISettings().responses_deployment_name,
            client=client,
            instructions="Return launch briefs as structured JSON.",
            name="ProductMarketer",
            text=AzureResponsesAgent.configure_response_format(ReleaseBrief),
        )
        response = await agent.get_response(
            "Draft a launch brief for the Contoso Note app.",
            response_format=ReleaseBrief,
        )
        print("[SK]", response.message.content)
    finally:
        await credential.close()


async def run_agent_framework() -> None:
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIResponsesClient

    chat_agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="Return launch briefs as structured JSON.",
        name="ProductMarketer",
    )
    # AFは呼び出し時に同じresponse_formatペイロードを転送します。
    reply = await chat_agent.run(
        "Draft a launch brief for the Contoso Note app.",
        response_format=ReleaseBrief,
    )
    print("[AF]", reply.text)


async def main() -> None:
    await run_semantic_kernel()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
