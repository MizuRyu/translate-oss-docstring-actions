# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role, TextContent

from agent_framework_redis import RedisChatMessageStore


class TestRedisChatMessageStore:
    """モックされたRedisクライアントを使用したRedisChatMessageStoreの単体テスト。

    これらのテストはモックされたRedis操作を使用して、実際のRedisサーバーを必要とせずに
    RedisChatMessageStoreのロジックと動作を検証します。

    """

    @pytest.fixture
    def sample_messages(self):
        """テスト用のサンプルチャットメッセージ。"""
        return [
            ChatMessage(role=Role.USER, text="Hello", message_id="msg1"),
            ChatMessage(role=Role.ASSISTANT, text="Hi there!", message_id="msg2"),
            ChatMessage(role=Role.USER, text="How are you?", message_id="msg3"),
        ]

    @pytest.fixture
    def mock_redis_client(self):
        """必要なすべてのメソッドを持つモックRedisクライアント。"""
        client = MagicMock()
        # コアのリスト操作
        client.lrange = AsyncMock(return_value=[])
        client.llen = AsyncMock(return_value=0)
        client.lindex = AsyncMock(return_value=None)
        client.lset = AsyncMock(return_value=True)
        client.lrem = AsyncMock(return_value=0)
        client.lpop = AsyncMock(return_value=None)
        client.rpop = AsyncMock(return_value=None)
        client.ltrim = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=1)

        # パイプライン操作
        mock_pipeline = AsyncMock()
        mock_pipeline.rpush = AsyncMock()
        mock_pipeline.execute = AsyncMock()
        client.pipeline.return_value.__aenter__.return_value = mock_pipeline

        return client

    @pytest.fixture
    def redis_store(self, mock_redis_client):
        """モッククライアントを使用したRedisチャットメッセージストア。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis_client
            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id="test_thread_123")
            store._redis_client = mock_redis_client
            return store

    def test_init_with_thread_id(self):
        """明示的なスレッドIDでの初期化テスト。"""
        thread_id = "user123_session456"
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id=thread_id)

        assert store.thread_id == thread_id
        assert store.redis_url == "redis://localhost:6379"
        assert store.key_prefix == "chat_messages"
        assert store.redis_key == f"chat_messages:{thread_id}"

    def test_init_auto_generate_thread_id(self):
        """自動生成されたスレッドIDでの初期化テスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(redis_url="redis://localhost:6379")

        assert store.thread_id is not None
        assert store.thread_id.startswith("thread_")
        assert len(store.thread_id) > 10  # UUIDであるべきです

    def test_init_with_custom_prefix(self):
        """カスタムキー接頭辞での初期化テスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(
                redis_url="redis://localhost:6379", thread_id="test123", key_prefix="custom_messages"
            )

        assert store.redis_key == "custom_messages:test123"

    def test_init_with_max_messages(self):
        """メッセージ制限付きの初期化テスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id="test123", max_messages=100)

        assert store.max_messages == 100

    def test_init_with_redis_url_required(self):
        """初期化にredis_urlが必要であることのテスト。"""
        with pytest.raises(ValueError, match="redis_url is required for Redis connection"):
            # redis_urlが必要なため例外を発生させるべきです
            RedisChatMessageStore(thread_id="test123")

    def test_init_with_initial_messages(self, sample_messages):
        """初期メッセージを使った初期化テスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(
                redis_url="redis://localhost:6379", thread_id="test123", messages=sample_messages
            )

        assert store._initial_messages == sample_messages

    async def test_add_messages_single(self, redis_store, mock_redis_client, sample_messages):
        """パイプライン操作を使った単一メッセージ追加のテスト。"""
        message = sample_messages[0]

        await redis_store.add_messages([message])

        # パイプライン操作が呼び出されたことを検証する
        mock_redis_client.pipeline.assert_called_with(transaction=True)

        # パイプラインのモックを取得し、正しく使用されたことを検証する
        pipeline_mock = mock_redis_client.pipeline.return_value.__aenter__.return_value
        pipeline_mock.rpush.assert_called()
        pipeline_mock.execute.assert_called()

    async def test_add_messages_multiple(self, redis_store, mock_redis_client, sample_messages):
        """パイプライン操作を使った複数メッセージ追加のテスト。"""
        await redis_store.add_messages(sample_messages)

        # パイプライン操作を検証する
        mock_redis_client.pipeline.assert_called_with(transaction=True)

        # 各メッセージに対してrpushが呼び出されたことを検証する
        pipeline_mock = mock_redis_client.pipeline.return_value.__aenter__.return_value
        assert pipeline_mock.rpush.call_count == len(sample_messages)

    async def test_add_messages_with_max_limit(self, mock_redis_client):
        """最大制限付きメッセージ追加でトリミングが発生することのテスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url") as mock_from_url:
            mock_from_url.return_value = mock_redis_client

            # 追加後に制限を超えるカウントを返すようにllenをモックする
            mock_redis_client.llen.return_value = 5

            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id="test123", max_messages=3)
            store._redis_client = mock_redis_client

            message = ChatMessage(role=Role.USER, text="Test")
            await store.add_messages([message])

            # 追加後に最後の3メッセージのみを保持するためにトリミングすべきです
            mock_redis_client.ltrim.assert_called_once_with("chat_messages:test123", -3, -1)

    async def test_list_messages_empty(self, redis_store, mock_redis_client):
        """ストアが空の場合のメッセージ一覧取得テスト。"""
        mock_redis_client.lrange.return_value = []

        messages = await redis_store.list_messages()

        assert messages == []
        mock_redis_client.lrange.assert_called_once_with("chat_messages:test_thread_123", 0, -1)

    async def test_list_messages_with_data(self, redis_store, mock_redis_client, sample_messages):
        """Redisにデータがある場合のメッセージ一覧取得テスト。"""
        # 実際のシリアル化メソッドを使って適切なシリアル化メッセージを作成する
        test_messages = [
            ChatMessage(role=Role.USER, text="Hello", message_id="msg1"),
            ChatMessage(role=Role.ASSISTANT, text="Hi there!", message_id="msg2"),
        ]
        serialized_messages = [redis_store._serialize_message(msg) for msg in test_messages]
        mock_redis_client.lrange.return_value = serialized_messages

        messages = await redis_store.list_messages()

        assert len(messages) == 2
        assert messages[0].role == Role.USER
        assert messages[0].text == "Hello"
        assert messages[1].role == Role.ASSISTANT
        assert messages[1].text == "Hi there!"

    async def test_list_messages_with_initial_messages(self, sample_messages):
        """初期メッセージがRedisに追加され正しく取得されることのテスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url") as mock_from_url:
            mock_redis_client = MagicMock()
            mock_redis_client.llen = AsyncMock(return_value=0)  # Redisキーが空です
            mock_redis_client.lrange = AsyncMock(return_value=[])

            # 初期メッセージ追加用のパイプラインをモックする
            mock_pipeline = AsyncMock()
            mock_pipeline.rpush = AsyncMock()
            mock_pipeline.execute = AsyncMock()
            mock_redis_client.pipeline.return_value.__aenter__.return_value = mock_pipeline

            mock_from_url.return_value = mock_redis_client

            store = RedisChatMessageStore(
                redis_url="redis://localhost:6379",
                thread_id="test123",
                messages=sample_messages[:1],  # One initial message
            )
            store._redis_client = mock_redis_client

            # 追加後に初期メッセージを返すようにRedisをモックする
            initial_message_json = store._serialize_message(sample_messages[0])
            mock_redis_client.lrange.return_value = [initial_message_json]

            messages = await store.list_messages()

            assert len(messages) == 1
            assert messages[0].text == "Hello"
            # パイプライン経由で初期メッセージがRedisに追加されたことを検証する
            mock_pipeline.rpush.assert_called()

    async def test_initial_messages_not_added_if_key_exists(self, sample_messages):
        """Redisキーに既にデータがある場合、初期メッセージが追加されないことのテスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url") as mock_from_url:
            mock_redis_client = MagicMock()
            mock_redis_client.llen = AsyncMock(return_value=5)  # キーに既にメッセージがあります
            mock_redis_client.lrange = AsyncMock(return_value=[])

            # キーが既に存在するためパイプラインは呼び出されるべきではありません
            mock_pipeline = AsyncMock()
            mock_pipeline.rpush = AsyncMock()
            mock_pipeline.execute = AsyncMock()
            mock_redis_client.pipeline.return_value.__aenter__.return_value = mock_pipeline

            mock_from_url.return_value = mock_redis_client

            store = RedisChatMessageStore(
                redis_url="redis://localhost:6379",
                thread_id="test123",
                messages=sample_messages[:1],  # One initial message
            )
            store._redis_client = mock_redis_client

            await store.list_messages()

            # キーが存在するため長さをチェックするがメッセージは追加しないべきです
            mock_redis_client.llen.assert_called()
            mock_pipeline.rpush.assert_not_called()

    async def test_serialize_state(self, redis_store):
        """状態のシリアル化テスト。"""
        state = await redis_store.serialize()

        expected_state = {
            "type": "redis_store_state",
            "thread_id": "test_thread_123",
            "redis_url": "redis://localhost:6379",
            "key_prefix": "chat_messages",
            "max_messages": None,
        }

        assert state == expected_state

    async def test_deserialize_state(self, redis_store):
        """状態のデシリアル化テスト。"""
        serialized_state = {
            "thread_id": "restored_thread_456",
            "redis_url": "redis://localhost:6380",
            "key_prefix": "restored_messages",
            "max_messages": 50,
        }

        await redis_store.update_from_state(serialized_state)

        assert redis_store.thread_id == "restored_thread_456"
        assert redis_store.redis_url == "redis://localhost:6380"
        assert redis_store.key_prefix == "restored_messages"
        assert redis_store.max_messages == 50

    async def test_deserialize_state_empty(self, redis_store):
        """空の状態をデシリアライズしても何も変わらないことのテスト。"""
        original_thread_id = redis_store.thread_id

        await redis_store.update_from_state(None)

        assert redis_store.thread_id == original_thread_id

    async def test_clear_messages(self, redis_store, mock_redis_client):
        """すべてのメッセージをクリアするテスト。"""
        await redis_store.clear()

        mock_redis_client.delete.assert_called_once_with("chat_messages:test_thread_123")

    async def test_message_serialization_roundtrip(self, sample_messages):
        """メッセージのシリアル化とデシリアル化の往復テスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id="test123")

        message = sample_messages[0]

        # シリアル化のテスト。
        serialized = store._serialize_message(message)
        assert isinstance(serialized, str)

        # デシリアル化のテスト。
        deserialized = store._deserialize_message(serialized)
        assert deserialized.role == message.role
        assert deserialized.text == message.text
        assert deserialized.message_id == message.message_id

    async def test_message_serialization_with_complex_content(self):
        """複雑なコンテンツを持つメッセージのシリアル化テスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url"):
            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id="test123")

        # 複数のコンテンツタイプを持つメッセージ
        message = ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="Hello"), TextContent(text="World")],
            author_name="TestBot",
            message_id="complex_msg",
            additional_properties={"metadata": "test"},
        )

        serialized = store._serialize_message(message)
        deserialized = store._deserialize_message(serialized)

        assert deserialized.role == Role.ASSISTANT
        assert deserialized.text == "Hello World"
        assert deserialized.author_name == "TestBot"
        assert deserialized.message_id == "complex_msg"
        assert deserialized.additional_properties == {"metadata": "test"}

    async def test_redis_connection_error_handling(self):
        """add_messagesでのRedis接続エラー処理のテスト。"""
        with patch("agent_framework_redis._chat_message_store.redis.from_url") as mock_from_url:
            mock_client = MagicMock()

            # 実行中に例外を発生させるパイプラインをモックする
            mock_pipeline = AsyncMock()
            mock_pipeline.rpush = AsyncMock()
            mock_pipeline.execute = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.pipeline.return_value.__aenter__.return_value = mock_pipeline

            mock_from_url.return_value = mock_client

            store = RedisChatMessageStore(redis_url="redis://localhost:6379", thread_id="test123")
            store._redis_client = mock_client

            message = ChatMessage(role=Role.USER, text="Test")

            # Redis接続エラーを伝播すべきです
            with pytest.raises(Exception, match="Connection failed"):
                await store.add_messages([message])

    async def test_getitem(self, redis_store, mock_redis_client, sample_messages):
        """Redis LINDEXを使ったgetitemメソッドのテスト。"""
        # 特定のメッセージを返すようにLINDEXをモックする
        serialized_msg0 = redis_store._serialize_message(sample_messages[0])
        serialized_msg1 = redis_store._serialize_message(sample_messages[1])

        def mock_lindex(key, index):
            if index == 0:
                return serialized_msg0
            if index == -1 or index == 1:
                return serialized_msg1
            return None

        mock_redis_client.lindex = AsyncMock(side_effect=mock_lindex)

        # 正のインデックスのテスト
        message = await redis_store.getitem(0)
        assert message.text == "Hello"

        # 負のインデックスのテスト
        message = await redis_store.getitem(-1)
        assert message.text == "Hi there!"

    async def test_getitem_index_error(self, redis_store, mock_redis_client):
        """無効なインデックスでgetitemがIndexErrorを発生させるテスト。"""
        mock_redis_client.lindex = AsyncMock(return_value=None)

        with pytest.raises(IndexError):
            await redis_store.getitem(0)

    async def test_setitem(self, redis_store, mock_redis_client, sample_messages):
        """Redis LSETを使ったsetitemメソッドのテスト。"""
        mock_redis_client.llen.return_value = 2
        mock_redis_client.lset = AsyncMock()

        new_message = ChatMessage(role=Role.USER, text="Updated message")
        await redis_store.setitem(0, new_message)

        mock_redis_client.lset.assert_called_once()
        call_args = mock_redis_client.lset.call_args
        assert call_args[0][0] == "chat_messages:test_thread_123"
        assert call_args[0][1] == 0

    async def test_setitem_index_error(self, redis_store, mock_redis_client):
        """無効なインデックスでsetitemがIndexErrorを発生させるテスト。"""
        mock_redis_client.llen.return_value = 0

        new_message = ChatMessage(role=Role.USER, text="Test")
        with pytest.raises(IndexError):
            await redis_store.setitem(0, new_message)

    async def test_append(self, redis_store, mock_redis_client):
        """appendメソッドがadd_messagesに委譲するテスト。"""
        message = ChatMessage(role=Role.USER, text="Appended message")
        await redis_store.append(message)

        # add_messages経由でパイプライン操作を呼び出すべきです
        mock_redis_client.pipeline.assert_called_with(transaction=True)

        # パイプライン経由でメッセージが追加されたことを検証する
        pipeline_mock = mock_redis_client.pipeline.return_value.__aenter__.return_value
        pipeline_mock.rpush.assert_called()
        pipeline_mock.execute.assert_called()

    async def test_count(self, redis_store, mock_redis_client):
        """countメソッドのテスト。"""
        mock_redis_client.llen.return_value = 5

        count = await redis_store.count()

        assert count == 5
        mock_redis_client.llen.assert_called_with("chat_messages:test_thread_123")

    async def test_len_method(self, redis_store, mock_redis_client):
        """非同期の__len__メソッドのテスト。"""
        mock_redis_client.llen.return_value = 3

        length = await redis_store.__len__()

        assert length == 3
        mock_redis_client.llen.assert_called_with("chat_messages:test_thread_123")

    def test_bool_method(self, redis_store):
        """__bool__メソッドは常にTrueを返すテスト。"""
        # ストアは常に真とみなされるべきです
        assert bool(redis_store) is True
        assert redis_store.__bool__() is True

        # if文で動作すべきです（Agent Frameworkが使用する方法）
        if redis_store:
            assert True  # ここに到達すべきです
        else:
            raise AssertionError("Store should be truthy")

    async def test_index_found(self, redis_store, mock_redis_client, sample_messages):
        """Redis LINDEXを使ってメッセージが見つかった場合のindexメソッドのテスト。"""
        mock_redis_client.llen.return_value = 2

        # 各位置でメッセージを返すようにLINDEXをモックする
        serialized_msg0 = redis_store._serialize_message(sample_messages[0])
        serialized_msg1 = redis_store._serialize_message(sample_messages[1])

        def mock_lindex(key, index):
            if index == 0:
                return serialized_msg0
            if index == 1:
                return serialized_msg1
            return None

        mock_redis_client.lindex = AsyncMock(side_effect=mock_lindex)

        index = await redis_store.index(sample_messages[1])
        assert index == 1

        # lindexが2回呼び出されたはずです（インデックス0、次に1）
        assert mock_redis_client.lindex.call_count == 2

    async def test_index_not_found(self, redis_store, mock_redis_client, sample_messages):
        """メッセージが見つからなかった場合のindexメソッドのテスト。"""
        mock_redis_client.llen.return_value = 1
        mock_redis_client.lindex = AsyncMock(return_value="different_message")

        with pytest.raises(ValueError, match="ChatMessage not found in store"):
            await redis_store.index(sample_messages[0])

    async def test_remove(self, redis_store, mock_redis_client, sample_messages):
        """Redis LREMを使ったremoveメソッドのテスト。"""
        mock_redis_client.lrem = AsyncMock(return_value=1)  # 1つの要素が削除されました

        await redis_store.remove(sample_messages[0])

        # LREMを使ってメッセージを削除すべきです
        expected_serialized = redis_store._serialize_message(sample_messages[0])
        mock_redis_client.lrem.assert_called_once_with("chat_messages:test_thread_123", 1, expected_serialized)

    async def test_remove_not_found(self, redis_store, mock_redis_client, sample_messages):
        """メッセージが見つからなかった場合のremoveメソッドのテスト。"""
        mock_redis_client.lrem = AsyncMock(return_value=0)  # 0個の要素が削除されました

        with pytest.raises(ValueError, match="ChatMessage not found in store"):
            await redis_store.remove(sample_messages[0])

    async def test_extend(self, redis_store, mock_redis_client, sample_messages):
        """extendメソッドがadd_messagesに委譲するテスト。"""
        await redis_store.extend(sample_messages[:2])

        # add_messages経由でパイプライン操作を呼び出すべきです
        mock_redis_client.pipeline.assert_called_with(transaction=True)

        # 各メッセージに対してrpushが呼び出されたことを検証する
        pipeline_mock = mock_redis_client.pipeline.return_value.__aenter__.return_value
        assert pipeline_mock.rpush.call_count >= 2
