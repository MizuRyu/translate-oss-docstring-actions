# Copyright (c) Microsoft. All rights reserved.
"""AutoGen と Agent Framework: Agent-as-a-Tool パターン。

一つのエージェントが専門化されたサブエージェントをツールとしてラップし、作業を委任する階層的エージェントアーキテクチャを示します。
"""

import asyncio


async def run_autogen() -> None:
    """ストリーミングを持つ階層的エージェントのための AutoGen の AgentTool。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.tools import AgentTool
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # 専門化されたライターエージェントを作成する
    writer_client = OpenAIChatCompletionClient(model="gpt-4.1-mini")
    writer = AssistantAgent(
        name="writer",
        model_client=writer_client,
        system_message="You are a creative writer. Write short, engaging content.",
        model_client_stream=True,
    )

    # ライターエージェントをツールとしてラップする（説明は agent.description から取得）
    writer_tool = AgentTool(agent=writer)

    # ライターをツールとして持つコーディネーターエージェントを作成する 重要: AgentTool 使用時は parallel_tool_calls を無効にしてください
    coordinator_client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        parallel_tool_calls=False,
    )
    coordinator = AssistantAgent(
        name="coordinator",
        model_client=coordinator_client,
        tools=[writer_tool],
        system_message="You coordinate with specialized agents. Delegate writing tasks to the writer agent.",
        model_client_stream=True,
    )

    # ストリーミングでコーディネーターを実行する - ライターに委任します
    print("[AutoGen]")
    await Console(coordinator.run_stream(task="Create a tagline for a coffee shop"))


async def run_agent_framework() -> None:
    """ストリーミングを持つ階層的エージェントのための Agent Framework の as_tool()。"""
    from agent_framework import FunctionCallContent, FunctionResultContent
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4.1-mini")

    # 専門化されたライターエージェントを作成する
    writer = client.create_agent(
        name="writer",
        instructions="You are a creative writer. Write short, engaging content.",
    )

    # as_tool() を使ってライターをツールに変換する
    writer_tool = writer.as_tool(
        name="creative_writer",
        description="Generate creative content",
        arg_name="request",
        arg_description="What to write",
    )

    # ライターツールを持つコーディネーターエージェントを作成する
    coordinator = client.create_agent(
        name="coordinator",
        instructions="You coordinate with specialized agents. Delegate writing tasks to the writer agent.",
        tools=[writer_tool],
    )

    # ストリーミングでコーディネーターを実行する - ライターに委任します
    print("[Agent Framework]")

    # 蓄積された関数呼び出しを追跡する（インクリメンタルにストリームされます）
    accumulated_calls: dict[str, FunctionCallContent] = {}

    async for chunk in coordinator.run_stream("Create a tagline for a coffee shop"):
        # テキストトークンをストリームする
        if chunk.text:
            print(chunk.text, end="", flush=True)

        # ストリーミングされる関数呼び出しと結果を処理する
        if chunk.contents:
            for content in chunk.contents:
                if isinstance(content, FunctionCallContent):
                    # ストリームされる関数呼び出しの内容を蓄積する
                    call_id = content.call_id
                    if call_id in accumulated_calls:
                        # 既存の呼び出しに追加する（引数は徐々にストリームされる）
                        accumulated_calls[call_id] = accumulated_calls[call_id] + content
                    else:
                        # この関数呼び出しの最初のチャンク
                        accumulated_calls[call_id] = content
                        print("\n[Function Call - streaming]", flush=True)
                        print(f"  Call ID: {call_id}", flush=True)
                        print(f"  Name: {content.name}", flush=True)

                    # これまでの蓄積された引数を表示する
                    current_args = accumulated_calls[call_id].arguments
                    print(f"  Arguments: {current_args}", flush=True)

                elif isinstance(content, FunctionResultContent):
                    # ツールの結果 - ライターのレスポンスを表示する
                    result_text = content.result if isinstance(content.result, str) else str(content.result)
                    if result_text.strip():
                        print("\n[Function Result]", flush=True)
                        print(f"  Call ID: {content.call_id}", flush=True)
                        print(f"  Result: {result_text[:150]}{'...' if len(result_text) > 150 else ''}", flush=True)
    print()


async def main() -> None:
    print("=" * 60)
    print("Agent-as-Tool Pattern Comparison")
    print("=" * 60)
    print("Note: AutoGen requires parallel_tool_calls=False for AgentTool")
    print("      Agent Framework handles this automatically\n")
    await run_autogen()
    print()
    await run_agent_framework()


if __name__ == "__main__":
    asyncio.run(main())
