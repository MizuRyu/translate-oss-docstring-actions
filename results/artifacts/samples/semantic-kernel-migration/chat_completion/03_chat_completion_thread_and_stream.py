# Copyright (c) Microsoft. All rights reserved.
"""チャットAgentの会話スレッドとストリーミングレスポンスを比較します。

両実装はターン間で会話スレッドを再利用し、2ターン目で出力をストリームします。
"""

import asyncio


async def run_semantic_kernel() -> None:
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    # SKのスレッドオブジェクトはAgent側で会話履歴を保持します。
    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(),
        name="Writer",
        instructions="Keep answers short and friendly.",
    )
    thread = ChatHistoryAgentThread()

    first = await agent.get_response(
        messages="Suggest a catchy headline for our product launch.",
        thread=thread,
    )
    print("[SK]", first.message.content)

    print("[SK][stream]", end=" ")
    async for update in agent.invoke_stream(
        messages="Draft a 2 sentence blurb.",
        thread=thread,
    ):
        if update.message:
            print(update.message.content, end="", flush=True)
    print()


async def run_agent_framework() -> None:
    from agent_framework.openai import OpenAIChatClient

    # AFのスレッドオブジェクトはAgentから明示的に要求されます。
    chat_agent = OpenAIChatClient().create_agent(
        name="Writer",
        instructions="Keep answers short and friendly.",
    )
    thread = chat_agent.get_new_thread()

    first = await chat_agent.run(
        "Suggest a catchy headline for our product launch.",
        thread=thread,
    )
    print("[AF]", first.text)

    print("[AF][stream]", end=" ")
    async for chunk in chat_agent.run_stream(
        "Draft a 2 sentence blurb.",
        thread=thread,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


async def main() -> None:
    await run_semantic_kernel()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
