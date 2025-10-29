# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
from pathlib import Path
from typing import Any

from agent_framework import ChatAgent
from agent_framework_azure_ai import AzureAIAgentClient
from azure.ai.agents.models import OpenApiAnonymousAuthDetails, OpenApiTool
from azure.identity.aio import AzureCliCredential

"""
The following sample demonstrates how to create a simple, Azure AI agent that
uses OpenAPI tools to answer user questions.
"""

# Agentとの会話をシミュレートします。
USER_INPUTS = [
    "What is the name and population of the country that uses currency with abbreviation THB?",
    "What is the current weather in the capital city of that country?",
]


def load_openapi_specs() -> tuple[dict[str, Any], dict[str, Any]]:
    """OpenAPI仕様ファイルを読み込みます。"""
    resources_path = Path(__file__).parent.parent / "resources"

    with open(resources_path / "weather.json") as weather_file:
        weather_spec = json.load(weather_file)

    with open(resources_path / "countries.json") as countries_file:
        countries_spec = json.load(countries_file)

    return weather_spec, countries_spec


async def main() -> None:
    """OpenAPIツールを備えたAzure AI Agentを示すメイン関数です。"""
    # 1. OpenAPI仕様を読み込みます（同期操作）。
    weather_openapi_spec, countries_openapi_spec = load_openapi_specs()

    # 2. AzureAIAgentClientを非同期コンテキストマネージャーとして使用し自動クリーンアップを行います。
    async with AzureAIAgentClient(async_credential=AzureCliCredential()) as client:
        # 3. Azure AIのOpenApiToolを使ってOpenAPIツールを作成します。
        auth = OpenApiAnonymousAuthDetails()

        openapi_weather = OpenApiTool(
            name="get_weather",
            spec=weather_openapi_spec,
            description="Retrieve weather information for a location using wttr.in service",
            auth=auth,
        )

        openapi_countries = OpenApiTool(
            name="get_country_info",
            spec=countries_openapi_spec,
            description="Retrieve country information including population and capital city",
            auth=auth,
        )

        # 4. OpenAPIツールを持つAgentを作成します。
        # 注意：AgentフレームワークにはまだHostedOpenApiToolラッパーがないため、Azure
        # AIのOpenApiTool定義を直接渡す必要があります。
        async with ChatAgent(
            chat_client=client,
            name="OpenAPIAgent",
            instructions=(
                "You are a helpful assistant that can search for country information "
                "and weather data using APIs. When asked about countries, use the country "
                "API to find information. When asked about weather, use the weather API. "
                "Provide clear, informative answers based on the API results."
            ),
            # Azure AIのOpenApiToolからの生のツール定義を渡します。
            tools=[*openapi_countries.definitions, *openapi_weather.definitions],
        ) as agent:
            # 5. スレッドコンテキストを維持しながらAgentとの会話をシミュレートします。
            print("=== Azure AI Agent with OpenAPI Tools ===\n")

            # 複数の実行間で会話コンテキストを維持するためのスレッドを作成します。
            thread = agent.get_new_thread()

            for user_input in USER_INPUTS:
                print(f"User: {user_input}")
                # 複数のagent.run()呼び出し間でコンテキストを維持するためにスレッドを渡します。
                response = await agent.run(user_input, thread=thread)
                print(f"Agent: {response.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
