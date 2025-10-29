# Copyright (c) Microsoft. All rights reserved.

import asyncio
from pathlib import Path

from agent_framework import ChatMessage, DataContent, Role, TextContent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

ASSETS_DIR = Path(__file__).resolve().parent.parent / "sample_assets"


def load_sample_pdf() -> bytes:
    """テスト用にバンドルされたサンプルPDFを読み込む。"""
    pdf_path = ASSETS_DIR / "sample.pdf"
    return pdf_path.read_bytes()


def create_sample_image() -> str:
    """テスト用のシンプルな1x1ピクセルPNG画像を作成する。"""
    # これはPNG形式の小さな赤いピクセルです。
    png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    return f"data:image/png;base64,{png_data}"


async def test_image() -> None:
    """Azure OpenAI Responses APIを使った画像解析のテスト。"""
    # 認証には、ターミナルで`az
    # login`コマンドを実行するか、AzureCliCredentialを好みの認証オプションに置き換えてください。AZURE_OPENAI_ENDPOINTとAZURE_OPENAI_RESPONSES_DEPLOYMENT_NAMEの環境変数が設定されている必要があります。
    # または、deployment_nameを明示的に渡すこともできます： client =
    # AzureOpenAIResponsesClient(credential=AzureCliCredential(),
    # deployment_name="your-deployment-name")
    client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    image_uri = create_sample_image()
    message = ChatMessage(
        role=Role.USER,
        contents=[TextContent(text="What's in this image?"), DataContent(uri=image_uri, media_type="image/png")],
    )

    response = await client.get_response(message)
    print(f"Image Response: {response}")


async def test_pdf() -> None:
    """Azure OpenAI Responses APIを使ったPDFドキュメント解析のテスト。"""
    client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    pdf_bytes = load_sample_pdf()
    message = ChatMessage(
        role=Role.USER,
        contents=[
            TextContent(text="What information can you extract from this document?"),
            DataContent(
                data=pdf_bytes,
                media_type="application/pdf",
                additional_properties={"filename": "sample.pdf"},
            ),
        ],
    )

    response = await client.get_response(message)
    print(f"PDF Response: {response}")


async def main() -> None:
    print("=== Testing Azure OpenAI Responses API Multimodal ===")
    print("The Responses API supports both images AND PDFs")
    await test_image()
    await test_pdf()


if __name__ == "__main__":
    asyncio.run(main())
