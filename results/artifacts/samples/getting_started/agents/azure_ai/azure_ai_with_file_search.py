# Copyright (c) Microsoft. All rights reserved.

import asyncio
from pathlib import Path

from agent_framework import ChatAgent, HostedFileSearchTool, HostedVectorStoreContent
from agent_framework_azure_ai import AzureAIAgentClient
from azure.ai.agents.models import FileInfo, VectorStore
from azure.identity.aio import AzureCliCredential

"""
The following sample demonstrates how to create a simple, Azure AI agent that
uses a file search tool to answer user questions.
"""


# Agentとの会話をシミュレートします。
USER_INPUTS = [
    "Who is the youngest employee?",
    "Who works in sales?",
    "I have a customer request, who can help me?",
]


async def main() -> None:
    """ファイル検索機能を備えたAzure AI Agentを示すメイン関数です。"""
    client = AzureAIAgentClient(async_credential=AzureCliCredential())
    file: FileInfo | None = None
    vector_store: VectorStore | None = None

    try:
        # 1. ファイルをアップロードしてベクターストアを作成します。
        pdf_file_path = Path(__file__).parent.parent / "resources" / "employees.pdf"
        print(f"Uploading file from: {pdf_file_path}")

        file = await client.project_client.agents.files.upload_and_poll(
            file_path=str(pdf_file_path), purpose="assistants"
        )
        print(f"Uploaded file, file ID: {file.id}")

        vector_store = await client.project_client.agents.vector_stores.create_and_poll(
            file_ids=[file.id], name="my_vectorstore"
        )
        print(f"Created vector store, vector store ID: {vector_store.id}")

        # 2. アップロードしたリソースを使ってファイル検索ツールを作成します。
        file_search_tool = HostedFileSearchTool(inputs=[HostedVectorStoreContent(vector_store_id=vector_store.id)])

        # 3. ファイル検索機能を持つAgentを作成します。 tool_resourcesはHostedFileSearchToolから自動的に抽出されます。
        async with ChatAgent(
            chat_client=client,
            name="EmployeeSearchAgent",
            instructions=(
                "You are a helpful assistant that can search through uploaded employee files "
                "to answer questions about employees."
            ),
            tools=file_search_tool,
        ) as agent:
            # 4. Agentとの会話をシミュレートします。
            for user_input in USER_INPUTS:
                print(f"# User: '{user_input}'")
                response = await agent.run(user_input)
                print(f"# Agent: {response.text}")

            # 5. クリーンアップ：ベクターストアとファイルを削除します。
            try:
                if vector_store:
                    await client.project_client.agents.vector_stores.delete(vector_store.id)
                if file:
                    await client.project_client.agents.files.delete(file.id)
            except Exception:
                # 問題の隠蔽を避けるためにクリーンアップのエラーは無視します。
                pass
    finally:
        # 6. クリーンアップ：早期失敗時に孤立リソースを防ぐためにベクターストアとファイルを削除します。
        # チャットAgentがクライアントを閉じるため、クライアントのリフレッシュが必要です。
        client = AzureAIAgentClient(async_credential=AzureCliCredential())
        try:
            if vector_store:
                await client.project_client.agents.vector_stores.delete(vector_store.id)
            if file:
                await client.project_client.agents.files.delete(file.id)
        except Exception:
            # 問題の隠蔽を避けるためにクリーンアップのエラーは無視します。
            pass
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
