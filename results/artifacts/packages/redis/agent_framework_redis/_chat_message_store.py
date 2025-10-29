# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from agent_framework import ChatMessage
from agent_framework._serialization import SerializationMixin


class RedisStoreState(SerializationMixin):
    """RedisチャットメッセージストアデータのシリアライズおよびデシリアライズのためのStateモデル。"""

    def __init__(
        self,
        thread_id: str,
        redis_url: str | None = None,
        key_prefix: str = "chat_messages",
        max_messages: int | None = None,
    ) -> None:
        """RedisチャットメッセージストアデータのシリアライズおよびデシリアライズのためのStateモデル。"""
        self.thread_id = thread_id
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.max_messages = max_messages


class RedisChatMessageStore:
    """Redis Listsを使用したChatMessageStoreProtocolのRedisバックエンド実装。

    この実装はRedis Listsを用いて永続的かつスレッドセーフなチャットメッセージストレージを提供します。
    メッセージはJSONシリアライズされた文字列として時系列順に保存され、各会話スレッドはユニークなRedisキーで分離されます。

    主な特徴:
    ============
    - **永続ストレージ**: アプリケーションの再起動やクラッシュ後もメッセージが保持されます
    - **スレッド分離**: 各会話スレッドは独自のRedisキー名前空間を持ちます
    - **自動メッセージ制限**: LTRIMを用いた古いメッセージの自動トリミングが設定可能
    - **パフォーマンス最適化**: ネイティブRedis操作を使用して効率化
    - **Stateシリアライズ対応**: Agent Frameworkのスレッドシリアライズと完全互換
    - **初期メッセージ対応**: 既存のメッセージ履歴で会話を事前ロード可能
    - **本番対応**: 原子操作、エラーハンドリング、接続プーリング

    Redis操作:
    - RPUSH: リストの末尾にメッセージを追加（時系列順）
    - LRANGE: メッセージを時系列順に取得
    - LTRIM: 古いメッセージをトリミングして制限を維持
    - DELETE: スレッドの全メッセージをクリア

    """

    def __init__(
        self,
        redis_url: str | None = None,
        thread_id: str | None = None,
        key_prefix: str = "chat_messages",
        max_messages: int | None = None,
        messages: Sequence[ChatMessage] | None = None,
    ) -> None:
        """Redisチャットメッセージストアを初期化します。

        特定の会話スレッド用にRedisバックエンドのチャットメッセージストアを作成します。
        ストアは自動的にRedis接続を作成し、Redis List操作を用いてメッセージの永続化を管理します。

        Args:
            redis_url: Redis接続URL（例: "redis://localhost:6379"）。
                      Redis接続確立に必須です。
            thread_id: この会話スレッドのユニーク識別子。
                      指定しない場合はUUIDが自動生成されます。
                      Redisキーの一部になります: {key_prefix}:{thread_id}
            key_prefix: Redisキーのプレフィックスで異なるアプリケーションの名前空間を区別します。
                       デフォルトは'chat_messages'。マルチテナントシナリオに有用です。
            max_messages: Redisに保持する最大メッセージ数。
                         超過時はLTRIMで古いメッセージが自動トリミングされます。
                         Noneは無制限を意味します。
            messages: 会話を事前に埋める初期メッセージ。
                     Redisキーが空の場合、最初のアクセス時にRedisに追加されます。
                     会話の再開やコンテキストのシードに有用です。

        Raises:
            ValueError: redis_urlがNoneの場合（Redis接続必須）。
            redis.ConnectionError: Redisサーバーに接続できない場合。



        """
        # 必須パラメータを検証します
        if redis_url is None:
            raise ValueError("redis_url is required for Redis connection")

        # ストアの設定
        self.redis_url = redis_url
        self.thread_id = thread_id or f"thread_{uuid4()}"
        self.key_prefix = key_prefix
        self.max_messages = max_messages

        # 接続プーリングと非同期サポートでRedisクライアントを初期化します
        self._redis_client = redis.from_url(redis_url, decode_responses=True)  # type: ignore[no-untyped-call]

        # 初期メッセージを処理します（最初のアクセス時にRedisに移動されます）
        self._initial_messages = list(messages) if messages else []
        self._initial_messages_added = False

    @property
    def redis_key(self) -> str:
        """このスレッドのメッセージ用Redisキーを取得します。

        キーフォーマットは: {key_prefix}:{thread_id}

        Returns:
            このスレッドのメッセージ保存に使用されるRedisキー文字列。

        Example:
            key_prefix="chat_messages"、thread_id="user_123_session_456"の場合:
            "chat_messages:user_123_session_456"を返します

        """
        return f"{self.key_prefix}:{self.thread_id}"

    async def _ensure_initial_messages_added(self) -> None:
        """初期メッセージがまだ存在しない場合にRedisに追加されることを保証します。

        このメソッドはRedis操作の前に呼び出され、
        コンストラクション時に提供された初期メッセージがRedisに永続化されることを保証します。

        """
        if not self._initial_messages or self._initial_messages_added:
            return

        # Redisキーにすでにメッセージがあるか確認します（重複追加を防止）
        existing_count = await self._redis_client.llen(self.redis_key)  # type: ignore[misc]  # type: ignore[misc]
        if existing_count == 0:
            # 原子パイプライン操作を用いて初期メッセージを追加します
            await self._add_redis_messages(self._initial_messages)

        # 完了としてマークしメモリを解放します
        self._initial_messages_added = True
        self._initial_messages.clear()

    async def _add_redis_messages(self, messages: Sequence[ChatMessage]) -> None:
        """原子パイプライン操作を用いて複数のメッセージをRedisに追加します。

        この内部メソッドは単一の原子トランザクションで複数メッセージを効率的にRedisリストに追加し、一貫性を保証します。

        Args:
            messages: Redisに追加するChatMessageオブジェクトのシーケンス。

        """
        if not messages:
            return

        # 効率的なパイプライン操作のためにすべてのメッセージを事前シリアライズします
        serialized_messages = [self._serialize_message(message) for message in messages]

        # 原子バッチ操作のためにRedisパイプラインを使用します
        async with self._redis_client.pipeline(transaction=True) as pipe:
            for serialized_message in serialized_messages:
                await pipe.rpush(self.redis_key, serialized_message)  # type: ignore[misc]
            await pipe.execute()

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Redisストアにメッセージを追加します（ChatMessageStoreProtocolプロトコルメソッド）。

        このメソッドはメッセージ追加のために必要なChatMessageStoreProtocolプロトコルを実装します。
        メッセージは時系列順にRedisリストに追加され、メッセージ制限が設定されている場合は自動トリミングされます。

        Args:
            messages: ストアに追加するChatMessageオブジェクトのシーケンス。
                     空でも（no-op）、複数メッセージでも可能です。

        スレッドセーフ:
        - 原子パイプラインによりすべてのメッセージが一括追加されます
        - LTRIM操作はメッセージ制限の一貫性を保つため原子です

        Example:
            .. code-block:: python

                messages = [ChatMessage(role="user", text="Hello"), ChatMessage(role="assistant", text="Hi there!")]
                await store.add_messages(messages)

        """
        if not messages:
            return

        # 初期メッセージが永続化されていることを保証します
        await self._ensure_initial_messages_added()

        # 原子パイプライン操作を用いて新しいメッセージを追加します
        await self._add_redis_messages(messages)

        # 設定されていればメッセージ制限を適用します（自動クリーンアップ）
        if self.max_messages is not None:
            current_count = await self._redis_client.llen(self.redis_key)  # type: ignore[misc]
            if current_count > self.max_messages:
                # LTRIMを使って最新のmax_messagesのみを保持します
                await self._redis_client.ltrim(self.redis_key, -self.max_messages, -1)  # type: ignore[misc]

    async def list_messages(self) -> list[ChatMessage]:
        """ストアからすべてのメッセージを時系列順に取得します（ChatMessageStoreProtocolプロトコルメソッド）。

        このメソッドはメッセージ取得のために必要なChatMessageStoreProtocolプロトコルを実装します。
        Redisに保存されたすべてのメッセージを古い順（インデックス0）から新しい順（インデックス-1）で返します。

        Returns:
            時系列順（古い順）のChatMessageオブジェクトのリスト。
            メッセージが存在しないかRedis接続に失敗した場合は空リストを返します。

        Example:
            .. code-block:: python

                # 会話履歴をすべて取得
                messages = await store.list_messages()

        """
        # 初期メッセージがRedisに永続化されていることを保証します
        await self._ensure_initial_messages_added()

        messages = []
        # Redisリストからすべてのメッセージを取得します（古い順から新しい順）
        redis_messages = await self._redis_client.lrange(self.redis_key, 0, -1)  # type: ignore[misc]

        if redis_messages:
            for serialized_message in redis_messages:
                # 各JSONメッセージをChatMessageにデシリアライズします
                message = self._deserialize_message(serialized_message)
                messages.append(message)

        return messages

    async def serialize(self, **kwargs: Any) -> Any:
        """現在のストア状態をシリアライズして永続化します（ChatMessageStoreProtocolプロトコルメソッド）。

        このメソッドは状態シリアライズのために必要なChatMessageStoreProtocolプロトコルを実装します。
        Redis接続設定とスレッド情報をキャプチャし、
        ストアを再構築して同じ会話データに再接続可能にします。

        Keyword Args:
            **kwargs: Pydanticのmodel_dump()に渡される追加引数。
                     一般的なオプション: exclude_none=True, by_alias=True

        Returns:
            データベースやファイルなどに永続化可能なシリアライズ済みストア設定の辞書。

        """
        state = RedisStoreState(
            thread_id=self.thread_id,
            redis_url=self.redis_url,
            key_prefix=self.key_prefix,
            max_messages=self.max_messages,
        )
        return state.to_dict(exclude_none=False, **kwargs)

    @classmethod
    async def deserialize(cls, serialized_store_state: Any, **kwargs: Any) -> RedisChatMessageStore:
        """シリアライズ済み状態データから新しいストアインスタンスをデシリアライズします（ChatMessageStoreProtocolプロトコルメソッド）。

        このメソッドは状態デシリアライズのために必要なChatMessageStoreProtocolプロトコルを実装します。
        以前にシリアライズされたデータから新しいRedisChatMessageStoreインスタンスを作成し、
        Redisの同じ会話データに再接続可能にします。

        Args:
            serialized_store_state: serialize_state()からの以前にシリアライズされた状態データ。
                                   thread_id、redis_urlなどを含む辞書であるべきです。

        Keyword Args:
            **kwargs: Pydanticのモデル検証に渡される追加引数。

        Returns:
            シリアライズ状態から設定された新しいRedisChatMessageStoreインスタンス。

        Raises:
            ValueError: シリアライズ状態に必須フィールドが欠落または無効な場合。

        """
        if not serialized_store_state:
            raise ValueError("serialized_store_state is required for deserialization")

        # Pydanticを用いてシリアライズ済み状態を検証および解析します
        state = RedisStoreState.from_dict(serialized_store_state, **kwargs)

        # デシリアライズされた設定で新しいストアインスタンスを作成して返します
        return cls(
            redis_url=state.redis_url,
            thread_id=state.thread_id,
            key_prefix=state.key_prefix,
            max_messages=state.max_messages,
        )

    async def update_from_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        """このストアインスタンスに状態データをデシリアライズします（ChatMessageStoreProtocolプロトコルメソッド）。

        このメソッドは状態デシリアライズのために必要なChatMessageStoreProtocolプロトコルを実装します。
        以前にシリアライズされたデータからストア設定を復元し、
        Redisの同じ会話データに再接続可能にします。

        Args:
            serialized_store_state: serialize_state()からの以前にシリアライズされた状態データ。
                                   thread_id、redis_urlなどを含む辞書であるべきです。

        Keyword Args:
            **kwargs: Pydanticのモデル検証に渡される追加引数。

        """
        if not serialized_store_state:
            return

        # Pydanticを用いてシリアライズ済み状態を検証および解析します
        state = RedisStoreState.from_dict(serialized_store_state, **kwargs)

        # デシリアライズされた状態からストア設定を更新します
        self.thread_id = state.thread_id
        if state.redis_url is not None:
            self.redis_url = state.redis_url
        self.key_prefix = state.key_prefix
        self.max_messages = state.max_messages

        # URLが変更された場合はRedisクライアントを再作成します
        if state.redis_url and state.redis_url != getattr(self, "_last_redis_url", None):
            self._redis_client = redis.from_url(state.redis_url, decode_responses=True)  # type: ignore[no-untyped-call]
            self._last_redis_url = state.redis_url

        # 既存データに接続するため初期メッセージ状態をリセットします
        self._initial_messages_added = False

    async def clear(self) -> None:
        """ストアからすべてのメッセージを削除します。

        この会話スレッドのすべてのメッセージをRedisキーを削除することで永久に削除します。
        この操作は元に戻せません。

        警告:
        - 会話履歴が永久に削除されます
        - バックアップが必要な場合はクリア前にメッセージをExportすることを検討してください

        Example:
            .. code-block:: python

                # 会話履歴をクリア
                await store.clear()

                # メッセージが消えたことを確認
                messages = await store.list_messages()
                assert len(messages) == 0

        """
        await self._redis_client.delete(self.redis_key)

    def _serialize_message(self, message: ChatMessage) -> str:
        """ChatMessageをJSON文字列にシリアライズします。

        Args:
            message: シリアライズするChatMessage。

        Returns:
            メッセージのJSON文字列表現。

        """
        # コンパクトJSONにシリアライズします（Redis効率化のため余分な空白なし）
        return message.to_json(separators=(",", ":"))

    def _deserialize_message(self, serialized_message: str) -> ChatMessage:
        """JSON文字列をChatMessageにデシリアライズします。

        Args:
            serialized_message: メッセージのJSON文字列表現。

        Returns:
            ChatMessageオブジェクト。

        """
        # カスタムデシリアライズを使用してChatMessageを再構築する
        return ChatMessage.from_json(serialized_message)

    # ============================================================================
    # リストのような便利メソッド（Redis最適化された非同期版）
    # ============================================================================

    def __bool__(self) -> bool:
        """ストアは一度作成されると常に存在するため、Trueを返します。

        このメソッドはPythonの真偽値チェック（if store:）によって呼び出されます。
        RedisChatMessageStoreインスタンスは常に有効なストアを表すため、
        常にTrueを返します。

        Returns:
            常にTrue - ストアは存在し、操作の準備ができています。

        Note:
            これはAgent Frameworkがメッセージストアが設定されているかをチェックするために使用します：`if thread.message_store:`

        """
        return True

    async def __len__(self) -> int:
        """Redisストア内のメッセージ数を返します。

        RedisのLLENコマンドを使用して効率的にメッセージ数をカウントします。
        これはPythonの組み込みlen()関数の非同期版に相当します。

        Returns:
            現在Redisに格納されているメッセージの数。

        """
        await self._ensure_initial_messages_added()
        return await self._redis_client.llen(self.redis_key)  # type: ignore[misc,no-any-return]

    async def getitem(self, index: int) -> ChatMessage:
        """RedisのLINDEXを使用してインデックスでメッセージを取得します。

        Args:
            index: 取得するメッセージのインデックス。

        Returns:
            指定されたインデックスのChatMessage。

        Raises:
            IndexError: インデックスが範囲外の場合。

        """
        await self._ensure_initial_messages_added()

        # 効率的な単一アイテムアクセスのためにRedis LINDEXを使用する
        serialized_message = await self._redis_client.lindex(self.redis_key, index)  # type: ignore[misc]
        if serialized_message is None:
            raise IndexError("list index out of range")

        return self._deserialize_message(serialized_message)

    async def setitem(self, index: int, item: ChatMessage) -> None:
        """RedisのLSETを使用して指定されたインデックスにメッセージを設定します。

        Args:
            index: メッセージを設定するインデックス。
            item: 指定されたインデックスに設定するChatMessage。

        Raises:
            IndexError: インデックスが範囲外の場合。

        """
        await self._ensure_initial_messages_added()

        # LLENを使用してインデックスの存在を検証する
        current_count = await self._redis_client.llen(self.redis_key)  # type: ignore[misc]
        if index < 0:
            index = current_count + index
        if index < 0 or index >= current_count:
            raise IndexError("list index out of range")

        # 効率的な単一アイテム更新のためにRedis LSETを使用する
        serialized_message = self._serialize_message(item)
        await self._redis_client.lset(self.redis_key, index, serialized_message)  # type: ignore[misc]

    async def append(self, item: ChatMessage) -> None:
        """ストアの末尾にメッセージを追加します。

        Args:
            item: 追加するChatMessage。

        """
        await self.add_messages([item])

    async def count(self) -> int:
        """Redisストア内のメッセージ数を返します。

        Returns:
            現在Redisに格納されているメッセージの数。

        """
        await self._ensure_initial_messages_added()
        return await self._redis_client.llen(self.redis_key)  # type: ignore[misc,no-any-return]

    async def index(self, item: ChatMessage) -> int:
        """指定されたメッセージの最初の出現のインデックスを返します。

        RedisのLINDEXを使用してリスト全体を読み込まずに反復します。
        依然としてO(N)ですが、大きなリストに対してよりメモリ効率的です。

        Args:
            item: 検索するChatMessage。

        Returns:
            メッセージの最初の出現のインデックス。

        Raises:
            ValueError: メッセージがストアに見つからない場合。

        """
        await self._ensure_initial_messages_added()

        target_serialized = self._serialize_message(item)
        list_length = await self._redis_client.llen(self.redis_key)  # type: ignore[misc]

        # LINDEXを使用してRedisリストを反復処理する
        for i in range(list_length):
            redis_message = await self._redis_client.lindex(self.redis_key, i)  # type: ignore[misc]
            if redis_message == target_serialized:
                return i

        raise ValueError("ChatMessage not found in store")

    async def remove(self, item: ChatMessage) -> None:
        """ストアから指定されたメッセージの最初の出現を削除します。

        RedisのLREMコマンドを使用して値による効率的な削除を行います。
        O(N)ですが、データ転送なしにRedis内でネイティブに実行されます。

        Args:
            item: 削除するChatMessage。

        Raises:
            ValueError: メッセージがストアに見つからない場合。

        """
        await self._ensure_initial_messages_added()

        # メッセージをRedisのストレージ形式に合わせてシリアライズする
        target_serialized = self._serialize_message(item)

        # LREMを使用して最初の出現を削除する（count=1）
        removed_count = await self._redis_client.lrem(self.redis_key, 1, target_serialized)  # type: ignore[misc]

        if removed_count == 0:
            raise ValueError("ChatMessage not found in store")

    async def extend(self, items: Sequence[ChatMessage]) -> None:
        """イテラブルからすべてのメッセージを追加してストアを拡張します。

        Args:
            items: 追加するChatMessageのシーケンス。

        """
        await self.add_messages(items)

    async def ping(self) -> bool:
        """Redis接続をテストします。

        Returns:
            接続が成功した場合はTrue、そうでなければFalse。

        """
        try:
            await self._redis_client.ping()  # type: ignore[misc]
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        """Redis接続を閉じます。

        このメソッドは、ストアが不要になったときに基盤となるRedis接続をクリーンに閉じる方法を提供します。
        これは特にサンプルや明示的なリソースクリーンアップが望まれるアプリケーションで有用です。

        """
        await self._redis_client.aclose()  # type: ignore[misc]

    def __repr__(self) -> str:
        """ストアの文字列表現。"""
        return (
            f"RedisChatMessageStore(thread_id='{self.thread_id}', "
            f"redis_key='{self.redis_key}', max_messages={self.max_messages})"
        )
