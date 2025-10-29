# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randrange
from typing import TYPE_CHECKING, Annotated, Any

from agent_framework import AgentRunResponse, ChatAgent, ChatMessage, ai_function
from agent_framework.openai import OpenAIResponsesClient

if TYPE_CHECKING:
    from agent_framework import AgentProtocol

"""
Demonstration of a tool with approvals.

This sample demonstrates using AI functions with user approval workflows.
It shows how to handle function call approvals without using threads.
"""

conditions = ["sunny", "cloudy", "raining", "snowing", "clear"]


@ai_function
def get_weather(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """指定された場所の現在の天気を取得する。"""
    # 天気データをシミュレートする
    return f"The weather in {location} is {conditions[randrange(0, len(conditions))]} and {randrange(-10, 30)}°C."


# 承認が必要なシンプルな天気ツールを定義する
@ai_function(approval_mode="always_require")
def get_weather_detail(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """指定された場所の現在の天気を取得する。"""
    # 天気データをシミュレートする
    return (
        f"The weather in {location} is {conditions[randrange(0, len(conditions))]} and {randrange(-10, 30)}°C, "
        "with a humidity of 88%. "
        f"Tomorrow will be {conditions[randrange(0, len(conditions))]} with a high of {randrange(-10, 30)}°C."
    )


async def handle_approvals(query: str, agent: "AgentProtocol") -> AgentRunResponse:
    """関数呼び出しの承認を処理する。

    スレッドがない場合、元のクエリ、承認リクエスト、および承認レスポンスを
    各イテレーションに含める必要がある。

    """
    result = await agent.run(query)
    while len(result.user_input_requests) > 0:
        # 元のクエリから開始する
        new_inputs: list[Any] = [query]

        for user_input_needed in result.user_input_requests:
            print(
                f"\nUser Input Request for function from {agent.name}:"
                f"\n  Function: {user_input_needed.function_call.name}"
                f"\n  Arguments: {user_input_needed.function_call.arguments}"
            )

            # 承認リクエストを含むアシスタントメッセージを追加する
            new_inputs.append(ChatMessage(role="assistant", contents=[user_input_needed]))

            # ユーザーの承認を得る
            user_approval = await asyncio.to_thread(input, "\nApprove function call? (y/n): ")

            # ユーザーの承認レスポンスを追加する
            new_inputs.append(
                ChatMessage(role="user", contents=[user_input_needed.create_response(user_approval.lower() == "y")])
            )

        # すべてのコンテキストで再度実行する
        result = await agent.run(new_inputs)

    return result


async def handle_approvals_streaming(query: str, agent: "AgentProtocol") -> None:
    """ストリーミングレスポンスで関数呼び出しの承認を処理する。

    スレッドがない場合、元のクエリ、承認リクエスト、および承認レスポンスを
    各イテレーションに含める必要がある。

    """
    current_input: str | list[Any] = query
    has_user_input_requests = True
    while has_user_input_requests:
        has_user_input_requests = False
        user_input_requests: list[Any] = []

        # レスポンスをストリームする
        async for chunk in agent.run_stream(current_input):
            if chunk.text:
                print(chunk.text, end="", flush=True)

            # ストリームからユーザー入力リクエストを収集する
            if chunk.user_input_requests:
                user_input_requests.extend(chunk.user_input_requests)

        if user_input_requests:
            has_user_input_requests = True
            # 元のクエリから開始する
            new_inputs: list[Any] = [query]

            for user_input_needed in user_input_requests:
                print(
                    f"\n\nUser Input Request for function from {agent.name}:"
                    f"\n  Function: {user_input_needed.function_call.name}"
                    f"\n  Arguments: {user_input_needed.function_call.arguments}"
                )

                # 承認リクエストを含むアシスタントメッセージを追加する
                new_inputs.append(ChatMessage(role="assistant", contents=[user_input_needed]))

                # ユーザーの承認を得る
                user_approval = await asyncio.to_thread(input, "\nApprove function call? (y/n): ")

                # ユーザーの承認レスポンスを追加する
                new_inputs.append(
                    ChatMessage(role="user", contents=[user_input_needed.create_response(user_approval.lower() == "y")])
                )

            # 次のイテレーションのためにすべてのコンテキストで入力を更新する
            current_input = new_inputs


async def run_weather_agent_with_approval(is_streaming: bool) -> None:
    """承認要件付きAI関数の例。"""
    print(f"\n=== Weather Agent with Approval Required ({'Streaming' if is_streaming else 'Non-Streaming'}) ===\n")

    async with ChatAgent(
        chat_client=OpenAIResponsesClient(),
        name="WeatherAgent",
        instructions=("You are a helpful weather assistant. Use the get_weather tool to provide weather information."),
        tools=[get_weather, get_weather_detail],
    ) as agent:
        query = "Can you give me an update of the weather in LA and Portland and detailed weather for Seattle?"
        print(f"User: {query}")

        if is_streaming:
            print(f"\n{agent.name}: ", end="", flush=True)
            await handle_approvals_streaming(query, agent)
            print()
        else:
            result = await handle_approvals(query, agent)
            print(f"\n{agent.name}: {result}\n")


async def main() -> None:
    print("=== Demonstration of a tool with approvals ===\n")

    await run_weather_agent_with_approval(is_streaming=False)
    await run_weather_agent_with_approval(is_streaming=True)


if __name__ == "__main__":
    asyncio.run(main())
