# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field

"""
Azure AI Agent with Existing Thread Example

This sample demonstrates working with pre-existing conversation threads
by providing thread IDs for thread reuse patterns.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    print("=== Azure AI Chat Client with Existing Thread ===")

    # クライアントを作成する
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential) as client,
    ):
        # 永続するスレッドを作成する
        created_thread = await client.agents.threads.create()

        try:
            async with ChatAgent(
                # ここでクライアントを渡すのはオプションです。ポータルから agent_id を取得すれば、上記の2行なしで直接使用できます。
                chat_client=AzureAIAgentClient(project_client=client),
                instructions="You are a helpful weather agent.",
                tools=get_weather,
            ) as agent:
                thread = agent.get_new_thread(service_thread_id=created_thread.id)
                assert thread.is_initialized
                result = await agent.run("What's the weather like in Tokyo?", thread=thread)
                print(f"Result: {result}\n")
        finally:
            # スレッドを手動でクリーンアップする
            await client.agents.threads.delete(created_thread.id)


if __name__ == "__main__":
    asyncio.run(main())
