# Copyright (c) Microsoft. All rights reserved.
"""AutoGen の SelectorGroupChat と Agent Framework の GroupChatBuilder の比較。

オーケストレーターが会話のコンテキストに基づいて次に話すエージェントを決定する
LLMベースのスピーカー選択を示します。
"""

import asyncio


async def run_autogen() -> None:
    """AutoGen の SelectorGroupChat による LLM ベースのスピーカー選択。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(model="gpt-4.1-mini")

    # 専門化されたエージェントを作成します
    python_expert = AssistantAgent(
        name="python_expert",
        model_client=client,
        system_message="You are a Python programming expert. Answer Python-related questions.",
        description="Expert in Python programming",
        model_client_stream=True,
    )

    javascript_expert = AssistantAgent(
        name="javascript_expert",
        model_client=client,
        system_message="You are a JavaScript programming expert. Answer JavaScript-related questions.",
        description="Expert in JavaScript programming",
        model_client_stream=True,
    )

    database_expert = AssistantAgent(
        name="database_expert",
        model_client=client,
        system_message="You are a database expert. Answer SQL and database-related questions.",
        description="Expert in databases and SQL",
        model_client_stream=True,
    )

    # セレクターグループチャットを作成します - LLM が適切な専門家を選択します
    team = SelectorGroupChat(
        participants=[python_expert, javascript_expert, database_expert],
        model_client=client,
        termination_condition=MaxMessageTermination(2),
        selector_prompt="Based on the conversation so far:\n{history}\n, "
        "select the most appropriate expert from {roles} to respond next.",
    )

    # 専門家選択が必要な質問で実行します
    print("[AutoGen] Selector group chat conversation:")
    await Console(team.run_stream(task="How do I connect to a PostgreSQL database using Python?"))


async def run_agent_framework() -> None:
    """Agent Framework の GroupChatBuilder による LLM ベースのスピーカー選択。"""
    from agent_framework import AgentRunUpdateEvent, GroupChatBuilder
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4.1-mini")

    # 専門化されたエージェントを作成します
    python_expert = client.create_agent(
        name="python_expert",
        instructions="You are a Python programming expert. Answer Python-related questions.",
        description="Expert in Python programming",
    )

    javascript_expert = client.create_agent(
        name="javascript_expert",
        instructions="You are a JavaScript programming expert. Answer JavaScript-related questions.",
        description="Expert in JavaScript programming",
    )

    database_expert = client.create_agent(
        name="database_expert",
        instructions="You are a database expert. Answer SQL and database-related questions.",
        description="Expert in databases and SQL",
    )

    workflow = (
        GroupChatBuilder()
        .participants([python_expert, javascript_expert, database_expert])
        .set_prompt_based_manager(
            chat_client=client,
            instructions="Based on the conversation, select the most appropriate expert to respond next.",
        )
        .with_max_rounds(1)
        .build()
    )

    # 専門家選択が必要な質問で実行します
    print("[Agent Framework] Group chat conversation:")
    current_executor = None
    async for event in workflow.run_stream("How do I connect to a PostgreSQL database using Python?"):
        if isinstance(event, AgentRunUpdateEvent):
            # 新しいエージェントに切り替わる際に executor 名のヘッダーを表示します
            if current_executor != event.executor_id:
                if current_executor is not None:
                    print()  # 前のエージェントのメッセージの後に改行を入れます
                print(f"---------- {event.executor_id} ----------")
                current_executor = event.executor_id
            if event.data:
                print(event.data.text, end="", flush=True)
    print()  # 会話の最後に改行を入れます


async def main() -> None:
    print("=" * 60)
    print("Selector Group Chat Comparison")
    print("=" * 60)
    print("AutoGen: SelectorGroupChat")
    print("Agent Framework: GroupChatBuilder with standard_manager\n")
    await run_autogen()
    print()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
