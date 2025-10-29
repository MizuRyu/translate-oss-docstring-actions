# Copyright (c) Microsoft. All rights reserved.

"""Redis Context Provider: スレッドスコープの例

このサンプルでは、Redis context providerを使用した場合の会話メモリのスコープ方法を示します。3つのシナリオをカバーしています：

1) グローバルスレッドスコープ
   - 固定のthread_idを提供し、操作/スレッド間でメモリを共有します。

2) 操作ごとのスレッドスコープ
   - scope_to_per_operation_thread_idを有効にして、providerインスタンスのライフタイム中に単一のスレッドにバインドします。同じスレッドオブジェクトを使って読み書きします。

3) 複数Agentでメモリを分離
   - 異なるagent_id値を使い、同じuser_idでも異なるAgentペルソナのメモリを分離します。

要件：
  - RediSearchが有効なRedisインスタンス（例：Redis Stack）
  - Redisエクストラがインストールされたagent-framework：pip install "agent-framework-redis"
  - このデモのチャットクライアント用にOpenAI APIキー（任意）

実行方法：
  python redis_threads.py
"""

import asyncio
import os
import uuid

from agent_framework.openai import OpenAIChatClient
from agent_framework_redis._provider import RedisProvider
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import OpenAITextVectorizer

# OpenAIベクトライザーを使用するには、OPENAI_API_KEYとOPENAI_CHAT_MODEL_ID環境変数を設定してください
# OPENAI_CHAT_MODEL_IDの推奨デフォルトはgpt-4o-miniです


async def example_global_thread_scope() -> None:
    """例1: グローバルthread_idスコープ（すべての操作でメモリ共有）。"""
    print("1. Global Thread Scope Example:")
    print("-" * 40)

    global_thread_id = str(uuid.uuid4())

    client = OpenAIChatClient(
        model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_threads_global",
        # overwrite_redis_index=True, drop_redis_index=True,
        application_id="threads_demo_app",
        agent_id="threads_demo_agent",
        user_id="threads_demo_user",
        thread_id=global_thread_id,
        scope_to_per_operation_thread_id=False,  # Share memories across all threads
    )

    agent = client.create_agent(
        name="GlobalMemoryAssistant",
        instructions=(
            "You are a helpful assistant. Personalize replies using provided context. "
            "Before answering, always check for stored context containing information"
        ),
        tools=[],
        context_providers=provider,
    )

    # グローバルスコープに好みを保存します
    query = "Remember that I prefer technical responses with code examples when discussing programming."
    print(f"User: {query}")
    result = await agent.run(query)
    print(f"Agent: {result}\n")

    # 新しいスレッドを作成します - グローバルスコープのためメモリは引き続きアクセス可能なはずです
    new_thread = agent.get_new_thread()
    query = "What technical responses do I prefer?"
    print(f"User (new thread): {query}")
    result = await agent.run(query, thread=new_thread)
    print(f"Agent: {result}\n")

    # Redisインデックスをクリーンアップします
    await provider.redis_index.delete()


async def example_per_operation_thread_scope() -> None:
    """例2: 操作ごとのスレッドスコープ（スレッドごとにメモリが分離）。

    注意: scope_to_per_operation_thread_id=Trueの場合、providerはライフタイム中単一のスレッドにバインドされます。
    そのproviderでのすべての操作に同じスレッドオブジェクトを使用してください。

    """
    print("2. Per-Operation Thread Scope Example:")
    print("-" * 40)

    client = OpenAIChatClient(
        model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorizer = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
        cache=EmbeddingsCache(name="openai_embeddings_cache", redis_url="redis://localhost:6379"),
    )

    provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_threads_dynamic",
        # overwrite_redis_index=True, drop_redis_index=True,
        application_id="threads_demo_app",
        agent_id="threads_demo_agent",
        user_id="threads_demo_user",
        scope_to_per_operation_thread_id=True,  # Isolate memories per thread
        redis_vectorizer=vectorizer,
        vector_field_name="vector",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
    )

    agent = client.create_agent(
        name="ScopedMemoryAssistant",
        instructions="You are an assistant with thread-scoped memory.",
        context_providers=provider,
    )

    # このスコープ付きprovider用に特定のスレッドを作成します
    dedicated_thread = agent.get_new_thread()

    # 専用スレッドに情報を保存します
    query = "Remember that for this conversation, I'm working on a Python project about data analysis."
    print(f"User (dedicated thread): {query}")
    result = await agent.run(query, thread=dedicated_thread)
    print(f"Agent: {result}\n")

    # 同じ専用スレッドでメモリ取得をテストします
    query = "What project am I working on?"
    print(f"User (same dedicated thread): {query}")
    result = await agent.run(query, thread=dedicated_thread)
    print(f"Agent: {result}\n")

    # 同じスレッドにさらに情報を保存します
    query = "Also remember that I prefer using pandas and matplotlib for this project."
    print(f"User (same dedicated thread): {query}")
    result = await agent.run(query, thread=dedicated_thread)
    print(f"Agent: {result}\n")

    # 包括的なメモリ取得をテストします
    query = "What do you know about my current project and preferences?"
    print(f"User (same dedicated thread): {query}")
    result = await agent.run(query, thread=dedicated_thread)
    print(f"Agent: {result}\n")

    # Redisインデックスをクリーンアップします
    await provider.redis_index.delete()


async def example_multiple_agents() -> None:
    """例3: 複数Agentで異なるスレッド構成（agent_idで分離）だが1つのインデックス内。"""
    print("3. Multiple Agents with Different Thread Configurations:")
    print("-" * 40)

    client = OpenAIChatClient(
        model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorizer = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
        cache=EmbeddingsCache(name="openai_embeddings_cache", redis_url="redis://localhost:6379"),
    )

    personal_provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_threads_agents",
        application_id="threads_demo_app",
        agent_id="agent_personal",
        user_id="threads_demo_user",
        redis_vectorizer=vectorizer,
        vector_field_name="vector",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
    )

    personal_agent = client.create_agent(
        name="PersonalAssistant",
        instructions="You are a personal assistant that helps with personal tasks.",
        context_providers=personal_provider,
    )

    work_provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_threads_agents",
        application_id="threads_demo_app",
        agent_id="agent_work",
        user_id="threads_demo_user",
        redis_vectorizer=vectorizer,
        vector_field_name="vector",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
    )

    work_agent = client.create_agent(
        name="WorkAssistant",
        instructions="You are a work assistant that helps with professional tasks.",
        context_providers=work_provider,
    )

    # 個人情報を保存します
    query = "Remember that I like to exercise at 6 AM and prefer outdoor activities."
    print(f"User to Personal Agent: {query}")
    result = await personal_agent.run(query)
    print(f"Personal Agent: {result}\n")

    # 仕事情報を保存します
    query = "Remember that I have team meetings every Tuesday at 2 PM."
    print(f"User to Work Agent: {query}")
    result = await work_agent.run(query)
    print(f"Work Agent: {result}\n")

    # メモリの分離をテストします
    query = "What do you know about my schedule?"
    print(f"User to Personal Agent: {query}")
    result = await personal_agent.run(query)
    print(f"Personal Agent: {result}\n")

    print(f"User to Work Agent: {query}")
    result = await work_agent.run(query)
    print(f"Work Agent: {result}\n")

    # Redisインデックスをクリーンアップします（共有）
    await work_provider.redis_index.delete()


async def main() -> None:
    print("=== Redis Thread Scoping Examples ===\n")
    await example_global_thread_scope()
    await example_per_operation_thread_scope()
    await example_multiple_agents()


if __name__ == "__main__":
    asyncio.run(main())
