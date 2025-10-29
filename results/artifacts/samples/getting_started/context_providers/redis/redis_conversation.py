# Copyright (c) Microsoft. All rights reserved.

"""Redis Context Provider: 基本的な使い方とAgent統合

この例では、Redis ChatMessageStoreProtocolを使用して会話の詳細を永続化する方法を示します。create_agentのコンストラクタ引数として渡します。

要件：
  - RediSearchが有効なRedisインスタンス（例：Redis Stack）
  - Redisエクストラがインストールされたagent-framework：pip install "agent-framework-redis"
  - ハイブリッド検索のために埋め込みを有効にする場合はOpenAI APIキー（任意）

実行方法：
  python redis_conversation.py
"""

import asyncio
import os

from agent_framework.openai import OpenAIChatClient
from agent_framework_redis._chat_message_store import RedisChatMessageStore
from agent_framework_redis._provider import RedisProvider
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import OpenAITextVectorizer


async def main() -> None:
    """providerとチャットメッセージストアの使用例を説明します。

    デバッグに便利（繰り返し時にコメント解除してください）：
      - print(await provider.redis_index.info())
      - print(await provider.search_all())

    """
    vectorizer = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
        cache=EmbeddingsCache(name="openai_embeddings_cache", redis_url="redis://localhost:6379"),
    )

    thread_id = "test_thread"

    provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_conversation",
        prefix="redis_conversation",
        application_id="matrix_of_kermits",
        agent_id="agent_kermit",
        user_id="kermit",
        redis_vectorizer=vectorizer,
        vector_field_name="vector",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
        thread_id=thread_id,
    )
    chat_message_store_factory = lambda: RedisChatMessageStore(
        redis_url="redis://localhost:6379",
        thread_id=thread_id,
        key_prefix="chat_messages",
        max_messages=100,
    )

    # Agent用のチャットクライアントを作成します
    client = OpenAIChatClient(model_id=os.getenv("OPENAI_CHAT_MODEL_ID"), api_key=os.getenv("OPENAI_API_KEY"))
    # Redis context
    # providerに接続されたAgentを作成します。providerは会話の詳細を自動的に永続化し、各ターンで関連コンテキストを提供します。
    agent = client.create_agent(
        name="MemoryEnhancedAssistant",
        instructions=(
            "You are a helpful assistant. Personalize replies using provided context. "
            "Before answering, always check for stored context"
        ),
        tools=[],
        context_providers=provider,
        chat_message_store_factory=chat_message_store_factory,
    )

    # ユーザーの好みを教えます。Agentはこれをproviderのメモリに書き込みます。
    query = "Remember that I enjoy gumbo"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    # Agentに保存された好みを思い出させます。メモリから取得できるはずです。
    query = "What do I enjoy?"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    query = "What did I say to you just now?"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    query = "Remember that anyone who does not clean shrimp will be eaten by a shark"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    query = "Tulips are red"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    query = "What was the first thing I said to you this conversation?"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)
    # Redisのproviderインデックスを削除します
    await provider.redis_index.delete()


if __name__ == "__main__":
    asyncio.run(main())
