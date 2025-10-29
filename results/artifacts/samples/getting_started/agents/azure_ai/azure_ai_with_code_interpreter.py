# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import AgentRunResponse, ChatResponseUpdate, HostedCodeInterpreterTool
from agent_framework.azure import AzureAIAgentClient
from azure.ai.agents.models import (
    RunStepDeltaCodeInterpreterDetailItemObject,
)
from azure.identity.aio import AzureCliCredential

"""
Azure AI Agent with Code Interpreter Example

This sample demonstrates using HostedCodeInterpreterTool with Azure AI Agents
for Python code execution and mathematical problem solving.
"""


def print_code_interpreter_inputs(response: AgentRunResponse) -> None:
    """コードインタープリターのデータにアクセスするためのヘルパーメソッド。"""

    print("\nCode Interpreter Inputs during the run:")
    if response.raw_representation is None:
        return
    for chunk in response.raw_representation:
        if isinstance(chunk, ChatResponseUpdate) and isinstance(
            chunk.raw_representation, RunStepDeltaCodeInterpreterDetailItemObject
        ):
            print(chunk.raw_representation.input, end="")
    print("\n")


async def main() -> None:
    """Azure AI で HostedCodeInterpreterTool を使う例。"""
    print("=== Azure AI Agent with Code Interpreter Example ===")

    # 認証には、ターミナルで `az login` コマンドを実行するか、AzureCliCredential を好みの認証オプションに置き換えてください。
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential) as chat_client,
    ):
        agent = chat_client.create_agent(
            name="CodingAgent",
            instructions=("You are a helpful assistant that can write and execute Python code to solve problems."),
            tools=HostedCodeInterpreterTool(),
        )
        query = "Generate the factorial of 100 using python code, show the code and execute it."
        print(f"User: {query}")
        response = await AgentRunResponse.from_agent_response_generator(agent.run_stream(query))
        print(f"Agent: {response}")
        # コードインタープリターの出力を確認するには、レスポンスの raw_representations からアクセスできます。次の行のコメントを外してください:
        # print_code_interpreter_inputs(response)


if __name__ == "__main__":
    asyncio.run(main())
