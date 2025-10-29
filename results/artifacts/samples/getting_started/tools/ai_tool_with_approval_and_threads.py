# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated

from agent_framework import ChatAgent, ChatMessage, ai_function
from agent_framework.azure import AzureOpenAIChatClient

"""
Tool Approvals with Threads

This sample demonstrates using tool approvals with threads.
With threads, you don't need to manually pass previous messages -
the thread stores and retrieves them automatically.
"""


@ai_function(approval_mode="always_require")
def add_to_calendar(
    event_name: Annotated[str, "Name of the event"], date: Annotated[str, "Date of the event"]
) -> str:
    """カレンダーにイベントを追加する（承認が必要）。"""
    print(f">>> EXECUTING: add_to_calendar(event_name='{event_name}', date='{date}')")
    return f"Added '{event_name}' to calendar on {date}"


async def approval_example() -> None:
    """スレッドでの承認を示す例。"""
    print("=== Tool Approval with Thread ===\n")

    agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(),
        name="CalendarAgent",
        instructions="You are a helpful calendar assistant.",
        tools=[add_to_calendar],
    )

    thread = agent.get_new_thread()

    # ステップ1: Agentがツール呼び出しを要求する
    query = "Add a dentist appointment on March 15th"
    print(f"User: {query}")
    result = await agent.run(query, thread=thread)

    # 承認リクエストをチェックする
    if result.user_input_requests:
        for request in result.user_input_requests:
            print(f"\nApproval needed:")
            print(f"  Function: {request.function_call.name}")
            print(f"  Arguments: {request.function_call.arguments}")

            # ユーザーが承認する（実際のアプリではユーザー入力）
            approved = True  # 拒否を見るにはFalseに変更する
            print(f"  Decision: {'Approved' if approved else 'Rejected'}")

            # ステップ2: 承認レスポンスを送信する
            approval_response = request.create_response(approved=approved)
            result = await agent.run(ChatMessage(role="user", contents=[approval_response]), thread=thread)

    print(f"Agent: {result}\n")


async def rejection_example() -> None:
    """スレッドでの拒否を示す例。"""
    print("=== Tool Rejection with Thread ===\n")

    agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(),
        name="CalendarAgent",
        instructions="You are a helpful calendar assistant.",
        tools=[add_to_calendar],
    )

    thread = agent.get_new_thread()

    query = "Add a team meeting on December 20th"
    print(f"User: {query}")
    result = await agent.run(query, thread=thread)

    if result.user_input_requests:
        for request in result.user_input_requests:
            print(f"\nApproval needed:")
            print(f"  Function: {request.function_call.name}")
            print(f"  Arguments: {request.function_call.arguments}")

            # ユーザーが拒否する
            print(f"  Decision: Rejected")

            # 拒否レスポンスを送信する
            rejection_response = request.create_response(approved=False)
            result = await agent.run(ChatMessage(role="user", contents=[rejection_response]), thread=thread)

    print(f"Agent: {result}\n")


async def main() -> None:
    await approval_example()
    await rejection_example()


if __name__ == "__main__":
    asyncio.run(main())
