# Copyright (c) Microsoft. All rights reserved.
"""AutoGen の RoundRobinGroupChat と Agent Framework の GroupChatBuilder/SequentialBuilder の比較。

エージェントが順番にタスクを処理するラウンドロビン方式の逐次エージェントオーケストレーションを示します。
"""

import asyncio


async def run_autogen() -> None:
    """AutoGen の RoundRobinGroupChat による逐次エージェントオーケストレーション。"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(model="gpt-4.1-mini")

    # 専門化されたエージェントを作成します
    researcher = AssistantAgent(
        name="researcher",
        model_client=client,
        system_message="You are a researcher. Provide facts and data about the topic.",
        model_client_stream=True,
    )

    writer = AssistantAgent(
        name="writer",
        model_client=client,
        system_message="You are a writer. Turn research into engaging content.",
        model_client_stream=True,
    )

    editor = AssistantAgent(
        name="editor",
        model_client=client,
        system_message="You are an editor. Review and finalize the content. End with APPROVED if satisfied.",
        model_client_stream=True,
    )

    # ラウンドロビンチームを作成します
    team = RoundRobinGroupChat(
        participants=[researcher, writer, editor],
        termination_condition=TextMentionTermination("APPROVED"),
    )

    # チームを実行し、会話を表示します。
    print("[AutoGen] Round-robin conversation:")
    await Console(team.run_stream(task="Create a brief summary about electric vehicles"))


async def run_agent_framework() -> None:
    """Agent Framework の SequentialBuilder による逐次エージェントオーケストレーション。"""
    from agent_framework import AgentRunUpdateEvent, SequentialBuilder
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4.1-mini")

    # 専門化されたエージェントを作成します
    researcher = client.create_agent(
        name="researcher",
        instructions="You are a researcher. Provide facts and data about the topic.",
    )

    writer = client.create_agent(
        name="writer",
        instructions="You are a writer. Turn research into engaging content.",
    )

    editor = client.create_agent(
        name="editor",
        instructions="You are an editor. Review and finalize the content.",
    )

    # 逐次ワークフローを作成します
    workflow = SequentialBuilder().participants([researcher, writer, editor]).build()

    # ワークフローを実行します
    print("[Agent Framework] Sequential conversation:")
    current_executor = None
    async for event in workflow.run_stream("Create a brief summary about electric vehicles"):
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


async def run_agent_framework_with_cycle() -> None:
    """Agent Framework の WorkflowBuilder による循環エッジと条件付き終了を持つワークフロー。"""
    from agent_framework import (
        AgentExecutorRequest,
        AgentExecutorResponse,
        AgentRunUpdateEvent,
        WorkflowBuilder,
        WorkflowContext,
        WorkflowOutputEvent,
        executor,
    )
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4.1-mini")

    # 専門化されたエージェントを作成します
    researcher = client.create_agent(
        name="researcher",
        instructions="You are a researcher. Provide facts and data about the topic.",
    )

    writer = client.create_agent(
        name="writer",
        instructions="You are a writer. Turn research into engaging content.",
    )

    editor = client.create_agent(
        name="editor",
        instructions="You are an editor. Review and finalize the content. End with APPROVED if satisfied.",
    )

    # 承認チェック用のカスタム executor を作成します
    @executor
    async def check_approval(
        response: AgentExecutorResponse, context: WorkflowContext[AgentExecutorRequest, str]
    ) -> None:
        assert response.full_conversation is not None
        last_message = response.full_conversation[-1]
        if last_message and "APPROVED" in last_message.text:
            await context.yield_output("Content approved.")
        else:
            await context.send_message(AgentExecutorRequest(messages=response.full_conversation, should_respond=True))

    workflow = (
        WorkflowBuilder()
        .add_edge(researcher, writer)
        .add_edge(writer, editor)
        .add_edge(
            editor,
            check_approval,
        )
        .add_edge(check_approval, researcher)
        .set_start_executor(researcher)
        .build()
    )

    # ワークフローを実行します
    print("[Agent Framework with Cycle] Cyclic conversation:")
    current_executor = None
    async for event in workflow.run_stream("Create a brief summary about electric vehicles"):
        if isinstance(event, WorkflowOutputEvent):
            print("\n---------- Workflow Output ----------")
            print(event.data)
        elif isinstance(event, AgentRunUpdateEvent):
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
    print("Round-Robin / Sequential Orchestration Comparison")
    print("=" * 60)
    print("AutoGen: RoundRobinGroupChat")
    print("Agent Framework: SequentialBuilder + WorkflowBuilder with cycles\n")
    await run_autogen()
    print()
    await run_agent_framework()
    print()
    await run_agent_framework_with_cycle()


if __name__ == "__main__":
    asyncio.run(main())
