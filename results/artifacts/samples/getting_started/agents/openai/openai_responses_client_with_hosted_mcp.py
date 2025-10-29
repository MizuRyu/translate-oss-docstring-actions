# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import TYPE_CHECKING, Any

from agent_framework import ChatAgent, HostedMCPTool
from agent_framework.openai import OpenAIResponsesClient

"""
OpenAI Responses Client with Hosted MCP Example

This sample demonstrates integrating hosted Model Context Protocol (MCP) tools with
OpenAI Responses Client, including user approval workflows for function call security.
"""

if TYPE_CHECKING:
    from agent_framework import AgentProtocol, AgentThread


async def handle_approvals_without_thread(query: str, agent: "AgentProtocol"):
    """スレッドがない場合、入力、承認リクエスト、および承認を返すことを保証する必要があります。"""
    from agent_framework import ChatMessage

    result = await agent.run(query)
    while len(result.user_input_requests) > 0:
        new_inputs: list[Any] = [query]
        for user_input_needed in result.user_input_requests:
            print(
                f"User Input Request for function from {agent.name}: {user_input_needed.function_call.name}"
                f" with arguments: {user_input_needed.function_call.arguments}"
            )
            new_inputs.append(ChatMessage(role="assistant", contents=[user_input_needed]))
            user_approval = input("Approve function call? (y/n): ")
            new_inputs.append(
                ChatMessage(role="user", contents=[user_input_needed.create_response(user_approval.lower() == "y")])
            )

        result = await agent.run(new_inputs)
    return result


async def handle_approvals_with_thread(query: str, agent: "AgentProtocol", thread: "AgentThread"):
    """ここではスレッドに以前のレスポンスを処理させ、承認で再実行します。"""
    from agent_framework import ChatMessage

    result = await agent.run(query, thread=thread, store=True)
    while len(result.user_input_requests) > 0:
        new_input: list[Any] = []
        for user_input_needed in result.user_input_requests:
            print(
                f"User Input Request for function from {agent.name}: {user_input_needed.function_call.name}"
                f" with arguments: {user_input_needed.function_call.arguments}"
            )
            user_approval = input("Approve function call? (y/n): ")
            new_input.append(
                ChatMessage(
                    role="user",
                    contents=[user_input_needed.create_response(user_approval.lower() == "y")],
                )
            )
        result = await agent.run(new_input, thread=thread, store=True)
    return result


async def handle_approvals_with_thread_streaming(query: str, agent: "AgentProtocol", thread: "AgentThread"):
    """ここではスレッドに以前のレスポンスを処理させ、承認で再実行します。"""
    from agent_framework import ChatMessage

    new_input: list[ChatMessage] = []
    new_input_added = True
    while new_input_added:
        new_input_added = False
        new_input.append(ChatMessage(role="user", text=query))
        async for update in agent.run_stream(new_input, thread=thread, store=True):
            if update.user_input_requests:
                for user_input_needed in update.user_input_requests:
                    print(
                        f"User Input Request for function from {agent.name}: {user_input_needed.function_call.name}"
                        f" with arguments: {user_input_needed.function_call.arguments}"
                    )
                    user_approval = input("Approve function call? (y/n): ")
                    new_input.append(
                        ChatMessage(
                            role="user", contents=[user_input_needed.create_response(user_approval.lower() == "y")]
                        )
                    )
                    new_input_added = True
            else:
                yield update


async def run_hosted_mcp_without_thread_and_specific_approval() -> None:
    """スレッドを使用せずに承認付きMcp Toolsを示す例。"""
    print("=== Mcp with approvals and without thread ===")

    # Agent作成時にToolsが提供されます Agentはその生涯の間に任意のクエリでこれらのToolsを使用できます
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # microsoft_docs_searchツール呼び出しには承認を必要としません しかし他のツールには必要です
            approval_mode={"never_require_approval": ["microsoft_docs_search"]},
        ),
    ) as agent:
        # 最初のクエリ
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await handle_approvals_without_thread(query1, agent)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # 2番目のクエリ
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await handle_approvals_without_thread(query2, agent)
        print(f"{agent.name}: {result2}\n")


async def run_hosted_mcp_without_approval() -> None:
    """承認なしのMcp Toolsを示す例。"""
    print("=== Mcp without approvals ===")

    # Agent作成時にToolsが提供されます Agentはその生涯の間に任意のクエリでこれらのToolsを使用できます
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # すべての関数呼び出しに承認を必要としません これは承認メッセージが表示されず、
            # サービスによって完全に処理され、最終レスポンスが返されることを意味します。
            approval_mode="never_require",
        ),
    ) as agent:
        # 最初のクエリ
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await handle_approvals_without_thread(query1, agent)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # 2番目のクエリ
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await handle_approvals_without_thread(query2, agent)
        print(f"{agent.name}: {result2}\n")


async def run_hosted_mcp_with_thread() -> None:
    """スレッドを使用した承認付きMcp Toolsを示す例。"""
    print("=== Mcp with approvals and with thread ===")

    # Agent作成時にToolsが提供されます Agentはその生涯の間に任意のクエリでこれらのToolsを使用できます
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # すべての関数呼び出しに承認を必要とします
            approval_mode="always_require",
        ),
    ) as agent:
        # 最初のクエリ
        thread = agent.get_new_thread()
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await handle_approvals_with_thread(query1, agent, thread)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # 2番目のクエリ
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await handle_approvals_with_thread(query2, agent, thread)
        print(f"{agent.name}: {result2}\n")


async def run_hosted_mcp_with_thread_streaming() -> None:
    """スレッドを使用した承認付きMcp Toolsを示す例。"""
    print("=== Mcp with approvals and with thread ===")

    # Agent作成時にToolsが提供されます Agentはその生涯の間に任意のクエリでこれらのToolsを使用できます
    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=HostedMCPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
            # すべての関数呼び出しに承認を必要とします
            approval_mode="always_require",
        ),
    ) as agent:
        # 最初のクエリ
        thread = agent.get_new_thread()
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        print(f"{agent.name}: ", end="")
        async for update in handle_approvals_with_thread_streaming(query1, agent, thread):
            print(update, end="")
        print("\n")
        print("\n=======================================\n")
        # 2番目のクエリ
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        print(f"{agent.name}: ", end="")
        async for update in handle_approvals_with_thread_streaming(query2, agent, thread):
            print(update, end="")
        print("\n")


async def main() -> None:
    print("=== OpenAI Responses Client Agent with Hosted Mcp Tools Examples ===\n")

    await run_hosted_mcp_without_approval()
    await run_hosted_mcp_without_thread_and_specific_approval()
    await run_hosted_mcp_with_thread()
    await run_hosted_mcp_with_thread_streaming()


if __name__ == "__main__":
    asyncio.run(main())
