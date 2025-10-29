# Copyright (c) Microsoft. All rights reserved.
"""SKプラグインとAgent FrameworkツールをチャットAgentで比較します。

実行前にOpenAIまたはAzure OpenAIの認証情報を設定してください。この例では両SDKが会話中に呼び出す「specials」ツールを公開しています。
"""

import asyncio


async def run_semantic_kernel() -> None:
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.functions import kernel_function

    class SpecialsPlugin:
        @kernel_function(name="specials", description="List daily specials")
        def specials(self) -> str:
            return "Clam chowder, Cobb salad, Chai tea"

    # SKは構築時にプラグインインスタンスを添付してツールを宣伝します。
    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(),
        name="Host",
        instructions="Answer menu questions accurately.",
        plugins=[SpecialsPlugin()],
    )
    thread = ChatHistoryAgentThread()
    response = await agent.get_response(
        messages="What soup can I order today?",
        thread=thread,
    )
    print("[SK]", response.message.content)


async def run_agent_framework() -> None:
    from agent_framework._tools import ai_function
    from agent_framework.openai import OpenAIChatClient

    @ai_function(name="specials", description="List daily specials")
    async def specials() -> str:
        return "Clam chowder, Cobb salad, Chai tea"

    # AFのツールは各Agentインスタンスの呼び出し可能オブジェクトとして提供されます。
    chat_agent = OpenAIChatClient().create_agent(
        name="Host",
        instructions="Answer menu questions accurately.",
        tools=[specials],
    )
    thread = chat_agent.get_new_thread()
    reply = await chat_agent.run(
        "What soup can I order today?",
        thread=thread,
        tool_choice="auto",
    )
    print("[AF]", reply.text)


async def main() -> None:
    await run_semantic_kernel()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
