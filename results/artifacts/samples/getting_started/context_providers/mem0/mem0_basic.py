# Copyright (c) Microsoft. All rights reserved.

import asyncio
import uuid

from agent_framework.azure import AzureAIAgentClient
from agent_framework.mem0 import Mem0Provider
from azure.identity.aio import AzureCliCredential


def retrieve_company_report(company_code: str, detailed: bool) -> str:
    if company_code != "CNTS":
        raise ValueError("Company code not found")
    if not detailed:
        return "CNTS is a company that specializes in technology."
    return (
        "CNTS is a company that specializes in technology. "
        "It had a revenue of $10 million in 2022. It has 100 employees."
    )


async def main() -> None:
    """Mem0 context providerを使ったメモリ使用例。"""
    print("=== Mem0 Context Provider Example ===")

    # Mem0の各レコードはagent_id、user_id、application_id、またはthread_idに関連付けられている必要があります。
    # この例では、Mem0レコードをuser_idに関連付けています。
    user_id = str(uuid.uuid4())

    # Azure認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを希望の認証オプションに置き換えてください。
    # Mem0認証には、"api_key"パラメータまたはMEM0_API_KEY環境変数でMem0 APIキーを設定してください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="FriendlyAssistant",
            instructions="You are a friendly assistant.",
            tools=retrieve_company_report,
            context_providers=Mem0Provider(user_id=user_id),
        ) as agent,
    ):
        # 最初にエージェントに前のコンテキストなしで会社レポートの取得を依頼します。
        # エージェントは会社コードやレポート形式を知らないためツールを呼び出せず、明確化を求めるはずです。
        query = "Please retrieve my company report"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}\n")

        # 次に、エージェントに会社コードと使用したいレポート形式を伝えると ツールを呼び出してレポートを返すことができるはずです。
        query = "I always work with CNTS and I always want a detailed report format. Please remember and retrieve it."
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Agent: {result}\n")

        print("\nRequest within a new thread:")
        # エージェント用に新しいスレッドを作成します。 新しいスレッドは前の会話のコンテキストを持ちません。
        thread = agent.get_new_thread()

        # スレッドにmem0コンポーネントがあるため、エージェントは
        # Mem0コンポーネントからユーザーの好みを記憶できるので、明確化を求めずに会社レポートを取得できるはずです。
        query = "Please retrieve my company report"
        print(f"User: {query}")
        result = await agent.run(query, thread=thread)
        print(f"Agent: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
