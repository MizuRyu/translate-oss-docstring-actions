# Copyright (c) Microsoft. All rights reserved.

"""Redis Context Provider: 基本的な使い方とAgent統合

この例では、Redis context providerを使用してAgentの会話メモリを永続化および取得する方法を示します。3つの段階的に現実的なシナリオをカバーしています：

1) 単独のprovider使用（"basic cache"）
   - メッセージをRedisに書き込み、全文検索またはハイブリッドベクター検索を使って関連コンテキストを取得します。

2) Agent + provider
   - providerをAgentに接続し、Agentがユーザーの好みを保存し、ターン間で呼び出せるようにします。

3) Agent + provider + tool memory
   - シンプルなツールをAgentに公開し、ツールの出力からの詳細がAgentのメモリの一部としてキャプチャされ、取得可能であることを検証します。

要件：
  - RediSearchが有効なRedisインスタンス（例：Redis Stack）
  - Redisエクストラがインストールされたagent-framework：pip install "agent-framework-redis"
  - ハイブリッド検索のために埋め込みを有効にする場合はOpenAI APIキー（任意）

実行方法：
  python redis_basics.py
"""

import asyncio
import os

from agent_framework import ChatMessage, Role
from agent_framework.openai import OpenAIChatClient
from agent_framework_redis._provider import RedisProvider
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import OpenAITextVectorizer


def search_flights(origin_airport_code: str, destination_airport_code: str, detailed: bool = False) -> str:
    """ツールメモリを示すための模擬フライト検索ツール。

    Agentはこの関数を呼び出すことができ、返された詳細はRedis context providerによって保存されます。
    後でAgentにこれらのツール結果からの事実を思い出させ、メモリが期待通りに動作していることを検証します。

    """
    # ツールの構造化出力をシミュレートするために使用される最小限の静的カタログ
    flights = {
        ("JFK", "LAX"): {
            "airline": "SkyJet",
            "duration": "6h 15m",
            "price": 325,
            "cabin": "Economy",
            "baggage": "1 checked bag",
        },
        ("SFO", "SEA"): {
            "airline": "Pacific Air",
            "duration": "2h 5m",
            "price": 129,
            "cabin": "Economy",
            "baggage": "Carry-on only",
        },
        ("LHR", "DXB"): {
            "airline": "EuroWings",
            "duration": "6h 50m",
            "price": 499,
            "cabin": "Business",
            "baggage": "2 bags included",
        },
    }

    route = (origin_airport_code.upper(), destination_airport_code.upper())
    if route not in flights:
        return f"No flights found between {origin_airport_code} and {destination_airport_code}"

    flight = flights[route]
    if not detailed:
        return f"Flights available from {origin_airport_code} to {destination_airport_code}."

    return (
        f"{flight['airline']} operates flights from {origin_airport_code} to {destination_airport_code}. "
        f"Duration: {flight['duration']}. "
        f"Price: ${flight['price']}. "
        f"Cabin: {flight['cabin']}. "
        f"Baggage policy: {flight['baggage']}."
    )


async def main() -> None:
    """providerのみ、Agent統合、およびツールメモリのシナリオを順に説明します。

    デバッグに便利（繰り返し時にコメント解除してください）：
      - print(await provider.redis_index.info())
      - print(await provider.search_all())

    """

    print("1. Standalone provider usage:")
    print("-" * 40)
    # パーティションスコープとOpenAI埋め込みを使ってproviderを作成します
    # OpenAIベクトライザーを使用するには、OPENAI_API_KEYとOPENAI_CHAT_MODEL_ID環境変数を設定してください
    # OPENAI_CHAT_MODEL_IDの推奨デフォルトはgpt-4o-miniです
    # providerがハイブリッド（テキスト＋ベクター）検索を実行できるように埋め込みベクトライザーを付加します。
    # テキストのみの検索を好む場合は、'vectorizer'およびvector_*パラメータなしでRedisProviderをインスタンス化してください。
    vectorizer = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
        cache=EmbeddingsCache(name="openai_embeddings_cache", redis_url="redis://localhost:6379"),
    )
    # providerは永続化と取得を管理します。application_id/agent_id/user_idはマルチテナント分離のためのスコープデータで、thread_id（後で設定）は特定の会話に絞り込みます。
    provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_basics",
        application_id="matrix_of_kermits",
        agent_id="agent_kermit",
        user_id="kermit",
        redis_vectorizer=vectorizer,
        vector_field_name="vector",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
    )

    # Redisに永続化するためのサンプルチャットメッセージを構築します
    messages = [
        ChatMessage(role=Role.USER, text="runA CONVO: User Message"),
        ChatMessage(role=Role.ASSISTANT, text="runA CONVO: Assistant Message"),
        ChatMessage(role=Role.SYSTEM, text="runA CONVO: System Message"),
    ]

    # 'runA'の下で会話/スレッドを宣言/開始し、メッセージを書き込みます。
    # スレッドはproviderが会話固有のコンテキストをグループ化および取得するための論理的境界です。
    await provider.thread_created(thread_id="runA")
    await provider.invoked(request_messages=messages)

    # 仮想的なモデル呼び出しのために関連するメモリを取得します。providerは現在のRequestメッセージを検索クエリとして使用し、モデルの指示に注入するコンテキストを返します。
    ctx = await provider.invoking([ChatMessage(role=Role.SYSTEM, text="B: Assistant Message")])

    # 指示に注入される取得済みメモリを検査します （デバッグ用の出力で、取得が期待通りに動作していることを確認できます。）
    print("Model Invoking Result:")
    print(ctx)

    # Redisのproviderインデックスを削除します
    await provider.redis_index.delete()

    # --- Agent + provider: 好みを教えて思い出す ---

    print("\n2. Agent + provider: teach and recall a preference")
    print("-" * 40)
    # Agentデモ用の新しいprovider（インデックスを再作成）
    vectorizer = OpenAITextVectorizer(
        model="text-embedding-ada-002",
        api_config={"api_key": os.getenv("OPENAI_API_KEY")},
        cache=EmbeddingsCache(name="openai_embeddings_cache", redis_url="redis://localhost:6379"),
    )
    # 次のシナリオを新鮮に開始するためにクリーンなインデックスを再作成します
    provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_basics_2",
        prefix="context_2",
        application_id="matrix_of_kermits",
        agent_id="agent_kermit",
        user_id="kermit",
        redis_vectorizer=vectorizer,
        vector_field_name="vector",
        vector_algorithm="hnsw",
        vector_distance_metric="cosine",
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
    )

    # ユーザーの好みを教えます。Agentはこれをproviderのメモリに書き込みます。
    query = "Remember that I enjoy glugenflorgle"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    # Agentに保存された好みを思い出させます。メモリから取得できるはずです。
    query = "What do I enjoy?"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    # Redisのproviderインデックスを削除します
    await provider.redis_index.delete()

    # --- Agent + provider + tool: ツール由来のコンテキストを保存して思い出す ---

    print("\n3. Agent + provider + tool: store and recall tool-derived context")
    print("-" * 40)
    # テキストのみのprovider（全文検索のみ）。ベクトライザーおよび関連パラメータは省略しています。
    provider = RedisProvider(
        redis_url="redis://localhost:6379",
        index_name="redis_basics_3",
        prefix="context_3",
        application_id="matrix_of_kermits",
        agent_id="agent_kermit",
        user_id="kermit",
    )

    # フライト検索ツールを公開するAgentを作成します。ツールの出力はproviderによってキャプチャされ、後のターンで取得可能なコンテキストになります。
    client = OpenAIChatClient(model_id=os.getenv("OPENAI_CHAT_MODEL_ID"), api_key=os.getenv("OPENAI_API_KEY"))
    agent = client.create_agent(
        name="MemoryEnhancedAssistant",
        instructions=(
            "You are a helpful assistant. Personalize replies using provided context. "
            "Before answering, always check for stored context"
        ),
        tools=search_flights,
        context_providers=provider,
    )
    # ツールを呼び出します。出力はメモリ/コンテキストの一部になります。
    query = "Are there any flights from new york city (jfk) to la? Give me details"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)
    # Agentがツール由来のコンテキストを思い出せることを検証します。
    query = "Which flight did I ask about?"
    result = await agent.run(query)
    print("User: ", query)
    print("Agent: ", result)

    # Redisのproviderインデックスを削除します
    await provider.redis_index.delete()


if __name__ == "__main__":
    asyncio.run(main())
