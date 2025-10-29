# 著作権 (c) Microsoft。無断転載を禁じます。

import asyncio

from agent_framework import ChatMessage, TextContent, UriContent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

"""
Azure OpenAI Responses Client with Image Analysis Example

This sample demonstrates using Azure OpenAI Responses for image analysis and vision tasks,
showing multi-modal messages combining text and image content.
"""


async def main():
    print("=== Azure Responses Agent with Image Analysis ===")

    # 1. ビジョン機能を持つAzure Responses Agentを作成します
    agent = AzureOpenAIResponsesClient(credential=AzureCliCredential()).create_agent(
        name="VisionAgent",
        instructions="You are a helpful agent that can analyze images.",
    )

    # 2. テキストと画像の両方のコンテンツを含むシンプルなメッセージを作成します
    user_message = ChatMessage(
        role="user",
        contents=[
            TextContent(text="What do you see in this image?"),
            UriContent(
                uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                media_type="image/jpeg",
            ),
        ],
    )

    # 3. Agentのレスポンスを取得します
    print("User: What do you see in this image? [Image provided]")
    result = await agent.run(user_message)
    print(f"Agent: {result.text}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
