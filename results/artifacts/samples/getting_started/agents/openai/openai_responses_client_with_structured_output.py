# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework import AgentRunResponse
from agent_framework.openai import OpenAIResponsesClient
from pydantic import BaseModel

"""
OpenAI Responses Client with Structured Output Example

This sample demonstrates using structured output capabilities with OpenAI Responses Client,
showing Pydantic model integration for type-safe response parsing and data extraction.
"""


class OutputStruct(BaseModel):
    """テスト目的のための構造化された出力。"""

    city: str
    description: str


async def non_streaming_example() -> None:
    print("=== Non-streaming example ===")

    # OpenAI Responsesエージェントを作成する
    agent = OpenAIResponsesClient().create_agent(
        name="CityAgent",
        instructions="You are a helpful agent that describes cities in a structured format.",
    )

    # エージェントに都市について尋ねる
    query = "Tell me about Paris, France"
    print(f"User: {query}")

    # response_formatパラメータを使用してエージェントから構造化されたレスポンスを取得する
    result = await agent.run(query, response_format=OutputStruct)

    # レスポンス値から直接構造化された出力にアクセスする
    if result.value:
        structured_data: OutputStruct = result.value  # type: ignore
        print("Structured Output Agent (from result.value):")
        print(f"City: {structured_data.city}")
        print(f"Description: {structured_data.description}")
    else:
        print("Error: No structured data found in result.value")


async def streaming_example() -> None:
    print("=== Streaming example ===")

    # OpenAI Responsesエージェントを作成する
    agent = OpenAIResponsesClient().create_agent(
        name="CityAgent",
        instructions="You are a helpful agent that describes cities in a structured format.",
    )

    # エージェントに都市について尋ねる
    query = "Tell me about Tokyo, Japan"
    print(f"User: {query}")

    # AgentRunResponse.from_agent_response_generatorを使用してストリーミングエージェントから構造化されたレスポンスを取得する
    # このメソッドはすべてのストリーミング更新を収集し、それらを単一のAgentRunResponseに結合します
    result = await AgentRunResponse.from_agent_response_generator(
        agent.run_stream(query, response_format=OutputStruct),
        output_format_type=OutputStruct,
    )

    # レスポンス値から直接構造化された出力にアクセスする
    if result.value:
        structured_data: OutputStruct = result.value  # type: ignore
        print("Structured Output (from streaming with AgentRunResponse.from_agent_response_generator):")
        print(f"City: {structured_data.city}")
        print(f"Description: {structured_data.description}")
    else:
        print("Error: No structured data found in result.value")


async def main() -> None:
    print("=== OpenAI Responses Agent with Structured Output ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
