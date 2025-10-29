# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework.openai import OpenAIChatClient


async def suspend_resume_service_managed_thread() -> None:
    """サービス管理スレッドの一時停止と再開の方法を示す。"""
    print("=== Suspend-Resume Service-Managed Thread ===")

    # ここではOpenAI Chat Clientを例として使用しているが、他のチャットクライアントも使用可能。
    agent = OpenAIChatClient().create_agent(name="Joker", instructions="You are good at telling jokes.")

    # Agent会話のために新しいスレッドを開始する。
    thread = agent.get_new_thread()

    # ユーザー入力に応答する。
    query = "Tell me a joke about a pirate."
    print(f"User: {query}")
    print(f"Agent: {await agent.run(query, thread=thread)}\n")

    # 後で使用できるようにスレッド状態をシリアライズする。
    serialized_thread = await thread.serialize()

    # スレッドはデータベース、ファイル、または他の任意のストレージメカニズムに保存し、後で再度読み込むことができる。
    print(f"Serialized thread: {serialized_thread}\n")

    # ストレージから読み込んだ後にスレッド状態をデシリアライズする。
    resumed_thread = await agent.deserialize_thread(serialized_thread)

    # ユーザー入力に応答する。
    query = "Now tell the same joke in the voice of a pirate, and add some emojis to the joke."
    print(f"User: {query}")
    print(f"Agent: {await agent.run(query, thread=resumed_thread)}\n")


async def suspend_resume_in_memory_thread() -> None:
    """インメモリスレッドの一時停止と再開の方法を示す。"""
    print("=== Suspend-Resume In-Memory Thread ===")

    # ここではOpenAI Chat Clientを例として使用しているが、他のチャットクライアントも使用可能。
    agent = OpenAIChatClient().create_agent(name="Joker", instructions="You are good at telling jokes.")

    # Agent会話のために新しいスレッドを開始する。
    thread = agent.get_new_thread()

    # ユーザー入力に応答する。
    query = "Tell me a joke about a pirate."
    print(f"User: {query}")
    print(f"Agent: {await agent.run(query, thread=thread)}\n")

    # 後で使用できるようにスレッド状態をシリアライズする。
    serialized_thread = await thread.serialize()

    # スレッドはデータベース、ファイル、または他の任意のストレージメカニズムに保存し、後で再度読み込むことができる。
    print(f"Serialized thread: {serialized_thread}\n")

    # ストレージから読み込んだ後にスレッド状態をデシリアライズする。
    resumed_thread = await agent.deserialize_thread(serialized_thread)

    # ユーザー入力に応答する。
    query = "Now tell the same joke in the voice of a pirate, and add some emojis to the joke."
    print(f"User: {query}")
    print(f"Agent: {await agent.run(query, thread=resumed_thread)}\n")


async def main() -> None:
    print("=== Suspend-Resume Thread Examples ===")
    await suspend_resume_service_managed_thread()
    await suspend_resume_in_memory_thread()


if __name__ == "__main__":
    asyncio.run(main())
