# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatMessage, DataContent, Role, TextContent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


def create_sample_image() -> str:
    """テスト用のシンプルな1x1ピクセルPNG画像を作成する。"""
    # これはPNG形式の小さな赤いピクセルです。
    png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    return f"data:image/png;base64,{png_data}"

async def test_image() -> None:
    """Azure OpenAIを使った画像解析のテスト。"""
    # 認証には、ターミナルで`az
    # login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。AZURE_OPENAI_ENDPOINTとAZURE_OPENAI_CHAT_DEPLOYMENT_NAMEの環境変数が設定されている必要があります。
    # または、deployment_nameを明示的に渡すこともできます： client =
    # AzureOpenAIChatClient(credential=AzureCliCredential(),
    # deployment_name="your-deployment-name")
    client = AzureOpenAIChatClient(credential=AzureCliCredential())

    image_uri = create_sample_image()
    message = ChatMessage(
        role=Role.USER,
        contents=[TextContent(text="What's in this image?"), DataContent(uri=image_uri, media_type="image/png")],
    )

    response = await client.get_response(message)
    print(f"Image Response: {response}")


async def main() -> None:
    print("=== Testing Azure OpenAI Multimodal ===")
    print("Testing image analysis (supported by Chat Completions API)")
    await test_image()

if __name__ == "__main__":
    asyncio.run(main())
