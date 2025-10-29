# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import ChatAgent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

"""
OpenAI Chat Client with Local MCP Example

This sample demonstrates integrating Model Context Protocol (MCP) tools with
OpenAI Chat Client for extended functionality and external service access.
"""


async def mcp_tools_on_run_level() -> None:
    """Agent実行時に定義されたMCPツールの例。"""
    print("=== Tools Defined on Run Level ===")

    # Agentを実行する際にToolsが提供されます これは、Agentを実行する前にMCPサーバーに接続し、
    # Toolsをrunメソッドに渡す必要があることを意味します。
    async with (
        MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ) as mcp_server,
        ChatAgent(
            chat_client=OpenAIChatClient(),
            name="DocsAgent",
            instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        ) as agent,
    ):
        # 最初のクエリ
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await agent.run(query1, tools=mcp_server)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # 2番目のクエリ
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, tools=mcp_server)
        print(f"{agent.name}: {result2}\n")


async def mcp_tools_on_agent_level() -> None:
    """Agent作成時にToolsを定義する例。"""
    print("=== Tools Defined on Agent Level ===")

    # Agent作成時にToolsが提供されます Agentはその生涯の間に任意のクエリでこれらのToolsを使用できます
    # Agentはコンテキストマネージャーを通じてMCPサーバーに接続します。
    async with OpenAIChatClient().create_agent(
        name="DocsAgent",
        instructions="You are a helpful assistant that can help with microsoft documentation questions.",
        tools=MCPStreamableHTTPTool(  # Tools defined at agent creation
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ),
    ) as agent:
        # 最初のクエリ
        query1 = "How to create an Azure storage account using az cli?"
        print(f"User: {query1}")
        result1 = await agent.run(query1)
        print(f"{agent.name}: {result1}\n")
        print("\n=======================================\n")
        # 2番目のクエリ
        query2 = "What is Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await agent.run(query2)
        print(f"{agent.name}: {result2}\n")


async def main() -> None:
    print("=== OpenAI Chat Client Agent with MCP Tools Examples ===\n")

    await mcp_tools_on_agent_level()
    await mcp_tools_on_run_level()


if __name__ == "__main__":
    asyncio.run(main())
