# Copyright (c) Microsoft. All rights reserved.

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from agent_framework import ChatMessage, Role
from agent_framework.exceptions import AgentException, ServiceInitializationError
from redisvl.utils.vectorize import CustomTextVectorizer

from agent_framework_redis import RedisProvider

CUSTOM_VECTORIZER = CustomTextVectorizer(embed=lambda x: [1.0, 2.0, 3.0], dtype="float32")


@pytest.fixture
def mock_index() -> AsyncMock:
    idx = AsyncMock()
    idx.create = AsyncMock()
    idx.load = AsyncMock()
    idx.query = AsyncMock()
    idx.exists = AsyncMock(return_value=False)

    async def _paginate_generator(*_args: Any, **_kwargs: Any):
        # デフォルトの空ジェネレーター；必要に応じてテストごとにオーバーライド
        if False:  # pragma: no cover
            yield []
        return

    idx.paginate = _paginate_generator
    return idx


@pytest.fixture
def patch_index_from_dict(mock_index: AsyncMock):
    with patch("agent_framework_redis._provider.AsyncSearchIndex") as mock_cls:
        mock_cls.from_dict = MagicMock(return_value=mock_index)

        # from_existingをモックしてデフォルトで一致するスキーマのモックを返す これはスキーマ検証を特にテストしないテストでのスキーマ検証エラーを防ぎます
        async def mock_from_existing(index_name, redis_url):
            mock_existing = AsyncMock()
            # プロバイダーが生成するものと一致するスキーマを返す これは少しハックですが、既存のテストを継続して動作させます
            mock_existing.schema.to_dict = MagicMock(
                side_effect=lambda: mock_cls.from_dict.call_args[0][0] if mock_cls.from_dict.call_args else {}
            )
            return mock_existing

        mock_cls.from_existing = AsyncMock(side_effect=mock_from_existing)

        yield mock_cls


@pytest.fixture
def patch_queries():
    calls: dict[str, Any] = {"TextQuery": [], "HybridQuery": [], "FilterExpression": []}

    def _mk_query(kind: str):
        class _Q:  # simple marker object with captured kwargs
            def __init__(self, **kwargs):
                self.kind = kind
                self.kwargs = kwargs

        return _Q

    with (
        patch(
            "agent_framework_redis._provider.TextQuery",
            side_effect=lambda **k: calls["TextQuery"].append(k) or _mk_query("text")(**k),
        ) as text_q,
        patch(
            "agent_framework_redis._provider.HybridQuery",
            side_effect=lambda **k: calls["HybridQuery"].append(k) or _mk_query("hybrid")(**k),
        ) as hybrid_q,
        patch(
            "agent_framework_redis._provider.FilterExpression",
            side_effect=lambda s: calls["FilterExpression"].append(s) or ("FE", s),
        ) as filt,
    ):
        yield {"calls": calls, "TextQuery": text_q, "HybridQuery": hybrid_q, "FilterExpression": filt}


class TestRedisProviderInitialization:
    # パッケージからプロバイダーをインポートできることを検証します
    def test_import(self):
        from agent_framework_redis._provider import RedisProvider

        assert RedisProvider is not None

    # フィルターなしでの構築は例外を発生させないはずです；フィルターは呼び出し時に強制されます
    def test_init_without_filters_ok(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider()
        assert provider.user_id is None
        assert provider.agent_id is None
        assert provider.application_id is None
        assert provider.thread_id is None

    # ベクター設定が提供されていない場合、スキーマはベクターフィールドを省略すべきです
    def test_schema_without_vector_field(self, patch_index_from_dict):
        RedisProvider(user_id="u1")
        # from_dictに渡されたスキーマを検査する
        args, kwargs = patch_index_from_dict.from_dict.call_args
        schema = args[0]
        assert isinstance(schema, dict)
        names = [f["name"] for f in schema["fields"]]
        types = [f["type"] for f in schema["fields"]]
        assert "content" in names
        assert "text" in types
        assert "vector" not in types


class TestRedisProviderMessages:
    @pytest.fixture
    def sample_messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role=Role.USER, text="Hello, how are you?"),
            ChatMessage(role=Role.ASSISTANT, text="I'm doing well, thank you!"),
            ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant"),
        ]

    # 書き込みは無制限の操作を避けるために少なくとも1つのスコーピングフィルターを必要とします
    async def test_messages_adding_requires_filters(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider()
        with pytest.raises(ServiceInitializationError):
            await provider.invoked("thread123", ChatMessage(role=Role.USER, text="Hello"))

    # 提供された場合に操作ごとのスレッドIDをキャプチャします
    async def test_thread_created_sets_per_operation_id(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider(user_id="u1")
        await provider.thread_created("t1")
        assert provider._per_operation_thread_id == "t1"

    # scope_to_per_operation_thread_idがTrueの場合に単一スレッド使用を強制します
    async def test_thread_created_conflict_when_scoped(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider(user_id="u1", scope_to_per_operation_thread_id=True)
        provider._per_operation_thread_id = "t1"
        with pytest.raises(ValueError) as exc:
            await provider.thread_created("t2")
        assert "only be used with one thread" in str(exc.value)

    # 非同期ページネーターからのすべての結果を集約してフラットなリストにします
    async def test_search_all_paginates(self, mock_index: AsyncMock, patch_index_from_dict):  # noqa: ARG002
        async def gen(_q, page_size: int = 200):  # noqa: ARG001, ANN001
            yield [{"id": 1}]
            yield [{"id": 2}, {"id": 3}]

        mock_index.paginate = gen
        provider = RedisProvider(user_id="u1")
        res = await provider.search_all(page_size=2)
        assert res == [{"id": 1}, {"id": 2}, {"id": 3}]


class TestRedisProviderModelInvoking:
    # 読み取りは無制限の操作を避けるために少なくとも1つのスコーピングフィルターを必要とします
    async def test_model_invoking_requires_filters(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider()
        with pytest.raises(ServiceInitializationError):
            await provider.invoking(ChatMessage(role=Role.USER, text="Hi"))

    # テキストのみの検索パスを使用し、ヒットからコンテキストを構成することを保証します
    async def test_textquery_path_and_context_contents(
        self, mock_index: AsyncMock, patch_index_from_dict, patch_queries
    ):  # noqa: ARG002
        # 準備：テキストのみの検索
        mock_index.query = AsyncMock(return_value=[{"content": "A"}, {"content": "B"}])
        provider = RedisProvider(user_id="u1")

        # 実行
        ctx = await provider.invoking([ChatMessage(role=Role.USER, text="q1")])

        # 検証：TextQueryが使用されている（HybridQueryではない）、filter_expressionが含まれている
        assert patch_queries["TextQuery"].call_count == 1
        assert patch_queries["HybridQuery"].call_count == 0
        kwargs = patch_queries["calls"]["TextQuery"][0]
        assert kwargs["text"] == "q1"
        assert kwargs["text_field_name"] == "content"
        assert kwargs["num_results"] == 10
        assert "filter_expression" in kwargs

        # ContextはデフォルトのPromptの後に結合されたメモリを含みます
        assert ctx.messages is not None and len(ctx.messages) == 1
        text = ctx.messages[0].text
        assert text.endswith("A\nB")

    # 結果が返されない場合、Contextは内容を持ちません
    async def test_model_invoking_empty_results_returns_empty_context(
        self, mock_index: AsyncMock, patch_index_from_dict, patch_queries
    ):  # noqa: ARG002
        mock_index.query = AsyncMock(return_value=[])
        provider = RedisProvider(user_id="u1")
        ctx = await provider.invoking([ChatMessage(role=Role.USER, text="any")])
        assert ctx.messages == []

    # ベクタイザーとベクターフィールドが設定されている場合にハイブリッドベクター・テキスト検索が使用されることを保証します
    async def test_hybridquery_path_with_vectorizer(self, mock_index: AsyncMock, patch_index_from_dict, patch_queries):  # noqa: ARG002
        mock_index.query = AsyncMock(return_value=[{"content": "Hit"}])
        provider = RedisProvider(user_id="u1", redis_vectorizer=CUSTOM_VECTORIZER, vector_field_name="vec")

        ctx = await provider.invoking([ChatMessage(role=Role.USER, text="hello")])

        # 検証：ベクターとベクターフィールドを使ったHybridQueryが使用されている
        assert patch_queries["HybridQuery"].call_count == 1
        k = patch_queries["calls"]["HybridQuery"][0]
        assert k["text"] == "hello"
        assert k["vector_field_name"] == "vec"
        assert k["vector"] == [1.0, 2.0, 3.0]
        assert k["dtype"] == "float32"
        assert k["num_results"] == 10
        assert "filter_expression" in k

        # 返されたメモリから組み立てられたContext
        assert ctx.messages and "Hit" in ctx.messages[0].text


class TestRedisProviderContextManager:
    # 非同期コンテキストマネージャがチェーンのためにselfを返すことを検証します
    async def test_async_context_manager_returns_self(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider(user_id="u1")
        async with provider as ctx:
            assert ctx is provider

    # Exitは何もしない操作で例外を発生させません
    async def test_aexit_noop(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider(user_id="u1")
        assert await provider.__aexit__(None, None, None) is None


class TestMessagesAddingBehavior:
    # パーティションのデフォルトを注入し、許可されたロールを保持しながらメッセージを追加します
    async def test_messages_adding_adds_partition_defaults_and_roles(
        self, mock_index: AsyncMock, patch_index_from_dict
    ):  # noqa: ARG002
        provider = RedisProvider(
            application_id="app",
            agent_id="agent",
            user_id="u1",
            scope_to_per_operation_thread_id=True,
        )

        msgs = [
            ChatMessage(role=Role.USER, text="u"),
            ChatMessage(role=Role.ASSISTANT, text="a"),
            ChatMessage(role=Role.SYSTEM, text="s"),
        ]

        await provider.invoked(msgs)

        # デフォルトを含む整形されたドキュメントでloadが呼び出されることを保証します
        assert mock_index.load.await_count == 1
        (loaded_args, _kwargs) = mock_index.load.call_args
        docs = loaded_args[0]
        assert isinstance(docs, list) and len(docs) == 3
        for d in docs:
            assert d["role"] in {"user", "assistant", "system"}
            assert d["content"] in {"u", "a", "s"}
            assert d["application_id"] == "app"
            assert d["agent_id"] == "agent"
            assert d["user_id"] == "u1"

    # メッセージを追加する際に空白のテキストや許可されていないロール（例：TOOL）をスキップします
    async def test_messages_adding_ignores_blank_and_disallowed_roles(
        self, mock_index: AsyncMock, patch_index_from_dict
    ):  # noqa: ARG002
        provider = RedisProvider(user_id="u1", scope_to_per_operation_thread_id=True)
        msgs = [
            ChatMessage(role=Role.USER, text="   "),
            ChatMessage(role=Role.TOOL, text="tool output"),
        ]
        await provider.invoked(msgs)
        # 有効なメッセージがない場合 -> ロードしません
        assert mock_index.load.await_count == 0


class TestIndexCreationPublicCalls:
    # drop=True の場合、最初のパブリック書き込み呼び出し時にインデックスが一度だけ作成されることを保証します
    async def test_messages_adding_triggers_index_create_once_when_drop_true(
        self, mock_index: AsyncMock, patch_index_from_dict
    ):  # noqa: ARG002
        provider = RedisProvider(user_id="u1")
        await provider.invoked(ChatMessage(role=Role.USER, text="m1"))
        await provider.invoked(ChatMessage(role=Role.USER, text="m2"))
        # 最初の呼び出し時のみ作成します
        assert mock_index.create.await_count == 1

    # drop=False でインデックスが存在しない場合、最初の読み込み時にインデックスが作成されることを保証します
    async def test_model_invoking_triggers_create_when_drop_false_and_not_exists(
        self, mock_index: AsyncMock, patch_index_from_dict
    ):  # noqa: ARG002
        mock_index.exists = AsyncMock(return_value=False)
        provider = RedisProvider(user_id="u1")
        mock_index.query = AsyncMock(return_value=[{"content": "C"}])
        await provider.invoking([ChatMessage(role=Role.USER, text="q")])
        assert mock_index.create.await_count == 1


class TestThreadCreatedAdditional:
    # None または同じスレッドIDの繰り返しを許可します。スコープ内で異なるIDは例外を発生させます
    async def test_thread_created_allows_none_and_same_id(self, patch_index_from_dict):  # noqa: ARG002
        provider = RedisProvider(user_id="u1", scope_to_per_operation_thread_id=True)
        # None は許可されます
        await provider.thread_created(None)
        # 同じIDの繰り返しは許可されます
        await provider.thread_created("t1")
        await provider.thread_created("t1")
        # 異なるIDは例外を発生させるべきです
        with pytest.raises(ValueError):
            await provider.thread_created("t2")


class TestVectorPopulation:
    # vectorizer が設定されている場合、呼び出されるとコンテンツを埋め込み、ベクトルフィールドを埋めます
    async def test_messages_adding_populates_vector_field_when_vectorizer_present(
        self, mock_index: AsyncMock, patch_index_from_dict
    ):  # noqa: ARG002
        provider = RedisProvider(
            user_id="u1",
            scope_to_per_operation_thread_id=True,
            redis_vectorizer=CUSTOM_VECTORIZER,
            vector_field_name="vec",
        )

        await provider.invoked(ChatMessage(role=Role.USER, text="hello"))
        assert mock_index.load.await_count == 1
        (loaded_args, _kwargs) = mock_index.load.call_args
        docs = loaded_args[0]
        assert isinstance(docs, list) and len(docs) == 1
        vec = docs[0].get("vec")
        assert isinstance(vec, (bytes, bytearray))
        assert len(vec) == 3 * np.dtype(np.float32).itemsize


class TestRedisProviderSchemaVectors:
    # vectorizer が暗黙的に次元を提供する場合、ベクトルフィールドを追加します
    def test_schema_with_vector_field_and_dims_inferred(self, patch_index_from_dict):  # noqa: ARG002
        RedisProvider(user_id="u1", redis_vectorizer=CUSTOM_VECTORIZER, vector_field_name="vec")
        args, _ = patch_index_from_dict.from_dict.call_args
        schema = args[0]
        names = [f["name"] for f in schema["fields"]]
        types = {f["name"]: f["type"] for f in schema["fields"]}
        assert "vec" in names
        assert types["vec"] == "vector"

    # redis_vectorizer が正しい型でない場合に例外を発生させます
    def test_init_invalid_vectorizer(self, patch_index_from_dict):  # noqa: ARG002
        class DummyVectorizer:
            pass

        with pytest.raises(AgentException):
            RedisProvider(user_id="u1", redis_vectorizer=DummyVectorizer(), vector_field_name="vec")


class TestEnsureIndex:
    # インデックスを一度だけ作成し、重複呼び出しを防ぐために _index_initialized をマークします
    async def test_ensure_index_creates_once(self, mock_index: AsyncMock, patch_index_from_dict):  # noqa: ARG002
        # モックインデックスは存在しないため、作成されます
        mock_index.exists = AsyncMock(return_value=False)
        provider = RedisProvider(user_id="u1", overwrite_index=False)

        assert provider._index_initialized is False
        await provider._ensure_index()
        assert mock_index.create.await_count == 1
        assert provider._index_initialized is True

        # _index_initialized フラグにより、2回目の呼び出しで再作成されるべきではありません
        await provider._ensure_index()
        assert mock_index.create.await_count == 1

    # overwrite_index=True の場合、overwrite=True でインデックスを作成します
    async def test_ensure_index_with_overwrite_true(self, mock_index: AsyncMock, patch_index_from_dict):  # noqa: ARG002
        mock_index.exists = AsyncMock(return_value=True)
        provider = RedisProvider(user_id="u1", overwrite_index=True)

        await provider._ensure_index()

        # overwrite=True, drop=False で create を呼び出すべきです
        mock_index.create.assert_called_once_with(overwrite=True, drop=False)

    # インデックスが存在しない場合、overwrite=False でインデックスを作成します
    async def test_ensure_index_create_if_missing(self, mock_index: AsyncMock, patch_index_from_dict):  # noqa: ARG002
        mock_index.exists = AsyncMock(return_value=False)
        provider = RedisProvider(user_id="u1", overwrite_index=False)

        await provider._ensure_index()

        # overwrite=False, drop=False で create を呼び出すべきです
        mock_index.create.assert_called_once_with(overwrite=False, drop=False)

    # インデックスが存在し overwrite=False の場合、スキーマの互換性を検証します
    async def test_ensure_index_schema_validation_success(self, mock_index: AsyncMock, patch_index_from_dict):  # noqa: ARG002
        mock_index.exists = AsyncMock(return_value=True)
        provider = RedisProvider(user_id="u1", overwrite_index=False)

        # スキーマが一致する既存のモックインデックス
        expected_schema = provider.schema_dict
        patch_index_from_dict.from_existing.return_value.schema.to_dict.return_value = expected_schema

        await provider._ensure_index()

        # スキーマを検証し、作成を続行すべきです
        patch_index_from_dict.from_existing.assert_called_once_with("context", redis_url="redis://localhost:6379")
        mock_index.create.assert_called_once_with(overwrite=False, drop=False)

    # スキーマが一致しない場合、ServiceInitializationError を発生させます
    async def test_ensure_index_schema_validation_failure(self, mock_index: AsyncMock, patch_index_from_dict):  # noqa: ARG002
        mock_index.exists = AsyncMock(return_value=True)
        provider = RedisProvider(user_id="u1", overwrite_index=False)

        # プロバイダー作成後に異なるスキーマを返すようにモックをオーバーライドします
        async def mock_from_existing_different(index_name, redis_url):
            mock_existing = AsyncMock()
            mock_existing.schema.to_dict = MagicMock(return_value={"different": "schema"})
            return mock_existing

        patch_index_from_dict.from_existing = AsyncMock(side_effect=mock_from_existing_different)

        with pytest.raises(ServiceInitializationError) as exc:
            await provider._ensure_index()

        assert "incompatible with the current configuration" in str(exc.value)
        assert "overwrite_index=True" in str(exc.value)

        # スキーマ検証に失敗した場合、create を呼び出すべきではありません
        mock_index.create.assert_not_called()
