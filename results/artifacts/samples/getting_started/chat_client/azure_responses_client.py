# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import ChatResponse
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from pydantic import BaseModel, Field

"""
Azure Responses Client Direct Usage Example

Demonstrates direct AzureResponsesClient usage for structured response generation with Azure OpenAI models.
Shows function calling capabilities with custom business logic.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """指定された場所の天気を取得する。"""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


class OutputStruct(BaseModel):
    """天気情報の構造化された出力。"""

    location: str
    weather: str


async def main() -> None:
    # 認証には、ターミナルで`az login`コマンドを実行するか、AzureCliCredentialを希望の認証オプションに置き換えてください。
    client = AzureOpenAIResponsesClient(credential=AzureCliCredential())
    message = "What's the weather in Amsterdam and in Paris?"
    stream = True
    print(f"User: {message}")
    if stream:
        response = await ChatResponse.from_chat_response_generator(
            client.get_streaming_response(message, tools=get_weather, response_format=OutputStruct),
            output_format_type=OutputStruct,
        )
        print(f"Assistant: {response.value}")

    else:
        response = await client.get_response(message, tools=get_weather, response_format=OutputStruct)
        print(f"Assistant: {response.value}")


if __name__ == "__main__":
    asyncio.run(main())
