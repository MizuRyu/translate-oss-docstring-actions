# Copyright (c) Microsoft. All rights reserved.

import json
import sys
from collections.abc import MutableSequence, Sequence
from functools import reduce
from operator import and_
from typing import Any, Literal, cast

import numpy as np
from agent_framework import ChatMessage, Context, ContextProvider, Role
from agent_framework.exceptions import (
    AgentException,
    ServiceInitializationError,
    ServiceInvalidRequestError,
)
from redisvl.index import AsyncSearchIndex
from redisvl.query import FilterQuery, HybridQuery, TextQuery
from redisvl.query.filter import FilterExpression, Tag
from redisvl.utils.token_escaper import TokenEscaper
from redisvl.utils.vectorize import BaseVectorizer

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover

if sys.version_info >= (3, 12):
    from typing import override  # type: ignore # pragma: no cover
else:
    from typing_extensions import override  # type: ignore[import] # pragma: no cover


class RedisProvider(ContextProvider):
    """動的でフィルタ可能なスキーマを持つRedisコンテキストプロバイダー。

    Redisにコンテキストを保存し、スコープされたコンテキストを取得します。
    フルテキストまたはオプションのハイブリッドベクター検索を使用してモデルの応答を基盤化します。

    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "context",
        prefix: str = "context",
        # Redisベクタイザの設定（オプション、クライアントによって注入される）
        redis_vectorizer: BaseVectorizer | None = None,
        vector_field_name: str | None = None,
        vector_algorithm: Literal["flat", "hnsw"] | None = None,
        vector_distance_metric: Literal["cosine", "ip", "l2"] | None = None,
        # パーティションフィールド（フィルタリング用にインデックス化）
        application_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        scope_to_per_operation_thread_id: bool = False,
        # Promptとランタイム
        context_prompt: str = ContextProvider.DEFAULT_CONTEXT_PROMPT,
        redis_index: Any = None,
        overwrite_index: bool = False,
    ):
        """Redis Context Providerを作成します。

        Args:
            redis_url: RedisサーバーのURL。
            index_name: Redisインデックスの名前。
            prefix: Redisデータベース内のすべてのキーのプレフィックス。
            redis_vectorizer: Redisで使用するベクタイザ。
            vector_field_name: Redis内のベクターフィールドの名前。
            vector_algorithm: ベクター検索に使用するアルゴリズム。
            vector_distance_metric: ベクター検索に使用する距離メトリック。
            application_id: コンテキストをスコープするためのアプリケーションID。
            agent_id: コンテキストをスコープするためのエージェントID。
            user_id: コンテキストをスコープするためのユーザーID。
            thread_id: コンテキストをスコープするためのスレッドID。
            scope_to_per_operation_thread_id: 操作ごとのスレッドIDにスコープするかどうか。
            context_prompt: プロバイダーで使用するコンテキストプロンプト。
            redis_index: プロバイダーで使用するRedisインデックス。
            overwrite_index: 既存のRedisインデックスを上書きするかどうか。


        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.prefix = prefix
        if redis_vectorizer is not None and not isinstance(redis_vectorizer, BaseVectorizer):
            raise AgentException(
                f"The redis vectorizer is not a valid type, got: {type(redis_vectorizer)}, expected: BaseVectorizer."
            )
        self.redis_vectorizer = redis_vectorizer
        self.vector_field_name = vector_field_name
        self.vector_algorithm: Literal["flat", "hnsw"] | None = vector_algorithm
        self.vector_distance_metric: Literal["cosine", "ip", "l2"] | None = vector_distance_metric
        self.application_id = application_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.thread_id = thread_id
        self.scope_to_per_operation_thread_id = scope_to_per_operation_thread_id
        self.context_prompt = context_prompt
        self.overwrite_index = overwrite_index
        self._per_operation_thread_id: str | None = None
        self._token_escaper: TokenEscaper = TokenEscaper()
        self._conversation_id: str | None = None
        self._index_initialized: bool = False
        self._schema_dict: dict[str, Any] | None = None
        self.redis_index = redis_index or AsyncSearchIndex.from_dict(
            self.schema_dict, redis_url=self.redis_url, validate_on_load=True
        )

    @property
    def schema_dict(self) -> dict[str, Any]:
        """Redisスキーマ辞書を取得し、初回アクセス時に計算してキャッシュします。"""
        if self._schema_dict is None:
            # 利用可能な場合はベクタイザからベクター設定を取得する
            vector_dims = self.redis_vectorizer.dims if self.redis_vectorizer is not None else None
            vector_datatype = self.redis_vectorizer.dtype if self.redis_vectorizer is not None else None

            self._schema_dict = self._build_schema_dict(
                index_name=self.index_name,
                prefix=self.prefix,
                vector_field_name=self.vector_field_name,
                vector_dims=vector_dims,
                vector_datatype=vector_datatype,
                vector_algorithm=self.vector_algorithm,
                vector_distance_metric=self.vector_distance_metric,
            )
        return self._schema_dict

    def _build_filter_from_dict(self, filters: dict[str, str | None]) -> Any | None:
        """単純な等価タグから結合フィルター式を構築します。

        これは空でないタグフィルターをAND結合し、すべての操作をapp/agent/user/threadパーティションにスコープするために使用されます。

        Args:
            filters: フィールド名から値へのマッピング。偽値は無視されます。

        Returns:
            結合されたフィルター式、またはフィルターが提供されていない場合はNone。

        """
        parts = [Tag(k) == v for k, v in filters.items() if v]
        return reduce(and_, parts) if parts else None

    def _build_schema_dict(
        self,
        *,
        index_name: str,
        prefix: str,
        vector_field_name: str | None,
        vector_dims: int | None,
        vector_datatype: str | None,
        vector_algorithm: Literal["flat", "hnsw"] | None,
        vector_distance_metric: Literal["cosine", "ip", "l2"] | None,
    ) -> dict[str, Any]:
        """RediSearchスキーマ設定辞書を構築します。

        メッセージ用のテキストおよびタグフィールドと、KNN/ハイブリッド検索を可能にするオプションのベクターフィールドを定義します。

        Keyword Args:
            index_name: インデックス名。
            prefix: キープレフィックス。
            vector_field_name: ベクターフィールド名またはNone。
            vector_dims: ベクター次元数またはNone。
            vector_datatype: ベクターデータ型またはNone。
            vector_algorithm: ベクターインデックスアルゴリズムまたはNone。
            vector_distance_metric: ベクター距離メトリックまたはNone。

        Returns:
            インデックスとフィールド設定を表す辞書。

        """
        fields: list[dict[str, Any]] = [
            {"name": "role", "type": "tag"},
            {"name": "mime_type", "type": "tag"},
            {"name": "content", "type": "text"},
            # 会話の追跡
            {"name": "conversation_id", "type": "tag"},
            {"name": "message_id", "type": "tag"},
            {"name": "author_name", "type": "tag"},
            # パーティションフィールド（高速フィルタリング用のTAG）
            {"name": "application_id", "type": "tag"},
            {"name": "agent_id", "type": "tag"},
            {"name": "user_id", "type": "tag"},
            {"name": "thread_id", "type": "tag"},
        ]

        # 設定されている場合のみベクターフィールドを追加（パラメータなしでもプロバイダーを実行可能に保つ）
        if vector_field_name is not None and vector_dims is not None:
            fields.append({
                "name": vector_field_name,
                "type": "vector",
                "attrs": {
                    "algorithm": (vector_algorithm or "hnsw"),
                    "dims": int(vector_dims),
                    "distance_metric": (vector_distance_metric or "cosine"),
                    "datatype": (vector_datatype or "float32"),
                },
            })

        return {
            "index": {
                "name": index_name,
                "prefix": prefix,
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": fields,
        }

    async def _ensure_index(self) -> None:
        """検索インデックスを初期化します。

        - 既存のインデックスが存在しスキーマが一致する場合は接続
        - 存在しない場合は新しいインデックスを作成
        - overwrite_index=Trueで上書き
        - 偶発的なデータ損失を防ぐためスキーマ互換性を検証

        """
        if self._index_initialized:
            return

        # インデックスが既に存在するかをチェックする
        index_exists = await self.redis_index.exists()

        if not self.overwrite_index and index_exists:
            # 接続前にスキーマ互換性を検証する
            await self._validate_schema_compatibility()

        # インデックスを作成する（既存に接続するか新規作成）
        await self.redis_index.create(overwrite=self.overwrite_index, drop=False)

        self._index_initialized = True

    async def _validate_schema_compatibility(self) -> None:
        """既存のインデックススキーマが現在の設定と一致することを検証します。

        スキーマが一致しない場合はServiceInitializationErrorを発生させ、有用なガイダンスを提供します。

        self._build_schema_dictは最小限のスキーマを返しますが、Redisはすべてのデフォルトを埋めた拡張スキーマを返します。
        非互換性を比較するために、正規化されたデフォルト値で署名を作成し、スキーマの重要部分を比較します。

        """
        # attr正規化のデフォルト値
        TAG_DEFAULTS = {"separator": ",", "case_sensitive": False, "withsuffixtrie": False}
        TEXT_DEFAULTS = {"weight": 1.0, "no_stem": False}

        def _significant_index(i: dict[str, Any]) -> dict[str, Any]:
            return {k: i.get(k) for k in ("name", "prefix", "key_separator", "storage_type")}

        def _sig_tag(attrs: dict[str, Any] | None) -> dict[str, Any]:
            a = {**TAG_DEFAULTS, **(attrs or {})}
            return {k: a[k] for k in ("separator", "case_sensitive", "withsuffixtrie")}

        def _sig_text(attrs: dict[str, Any] | None) -> dict[str, Any]:
            a = {**TEXT_DEFAULTS, **(attrs or {})}
            return {k: a[k] for k in ("weight", "no_stem")}

        def _sig_vector(attrs: dict[str, Any] | None) -> dict[str, Any]:
            a = {**(attrs or {})}
            # ベクターフィールドが存在する場合はこれらの存在を要求する
            return {k: a.get(k) for k in ("algorithm", "dims", "distance_metric", "datatype")}

        def _schema_signature(schema: dict[str, Any]) -> dict[str, Any]:
            # 順序に依存しない最小限の署名
            sig: dict[str, Any] = {"index": _significant_index(schema.get("index", {})), "fields": {}}
            for f in schema.get("fields", []):
                name, ftype = f.get("name"), f.get("type")
                if not name:
                    continue
                if ftype == "tag":
                    sig["fields"][name] = {"type": "tag", "attrs": _sig_tag(f.get("attrs"))}
                elif ftype == "text":
                    sig["fields"][name] = {"type": "text", "attrs": _sig_text(f.get("attrs"))}
                elif ftype == "vector":
                    sig["fields"][name] = {"type": "vector", "attrs": _sig_vector(f.get("attrs"))}
                else:
                    # 不明なフィールドタイプ：タイプのみで比較する
                    sig["fields"][name] = {"type": ftype}
            return sig

        existing_index = await AsyncSearchIndex.from_existing(self.index_name, redis_url=self.redis_url)
        existing_schema = existing_index.schema.to_dict()
        current_schema = self.schema_dict

        existing_sig = _schema_signature(existing_schema)
        current_sig = _schema_signature(current_schema)

        if existing_sig != current_sig:
            # エラーメッセージに署名を追加する
            raise ServiceInitializationError(
                "Existing Redis index schema is incompatible with the current configuration.\n"
                f"Existing (significant): {json.dumps(existing_sig, indent=2, sort_keys=True)}\n"
                f"Current  (significant): {json.dumps(current_sig, indent=2, sort_keys=True)}\n"
                "Set overwrite_index=True to rebuild if this change is intentional."
            )

    async def _add(
        self,
        *,
        data: dict[str, Any] | list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """パーティションフィールドを埋めて1つまたは複数のドキュメントを挿入します。

        デフォルトのパーティションフィールドを埋め、設定されていればコンテンツを埋め込み、バッチでドキュメントをロードします。

        Keyword Args:
            data: 挿入する単一ドキュメントまたはドキュメントのリスト。
            metadata: オプションのメタデータ辞書（未使用のプレースホルダー）。

        Raises:
            ServiceInvalidRequestError: 必須フィールドが欠落または無効な場合。

        """
        # プロバイダーに少なくとも1つのスコープが設定されていることを保証する（Mem0Providerとの対称性）
        self._validate_filters()
        await self._ensure_index()
        docs = data if isinstance(data, list) else [data]

        prepared: list[dict[str, Any]] = []
        for doc in docs:
            d = dict(doc)  # 浅いコピー

            # パーティションのデフォルト値
            d.setdefault("application_id", self.application_id)
            d.setdefault("agent_id", self.agent_id)
            d.setdefault("user_id", self.user_id)
            d.setdefault("thread_id", self._effective_thread_id)
            # 会話のデフォルト値
            d.setdefault("conversation_id", self._conversation_id)

            # 論理的要件
            if "content" not in d:
                raise ServiceInvalidRequestError("add() requires a 'content' field in data")

            # ベクターフィールドの要件（スキーマに存在する場合のみ）
            if self.vector_field_name:
                d.setdefault(self.vector_field_name, None)

            prepared.append(d)

        # すべてのメッセージのコンテンツをバッチで埋め込む
        if self.redis_vectorizer and self.vector_field_name:
            text_list = [d["content"] for d in prepared]
            embeddings = await self.redis_vectorizer.aembed_many(text_list, batch_size=len(text_list))
            for i, d in enumerate(prepared):
                vec = np.asarray(embeddings[i], dtype=np.float32).tobytes()
                field_name: str = self.vector_field_name
                d[field_name] = vec

        # サポートされていれば一括でロードする
        await self.redis_index.load(prepared)
        return

    async def _redis_search(
        self,
        text: str,
        *,
        text_scorer: str = "BM25STD",
        filter_expression: Any | None = None,
        return_fields: list[str] | None = None,
        num_results: int = 10,
        alpha: float = 0.7,
    ) -> list[dict[str, Any]]:
        """オプションのフィルター付きでテキストまたはハイブリッドベクターテキスト検索を実行します。

        TextQueryまたはHybridQueryを構築し、パーティションフィルターを自動的にAND結合して結果をスコープし安全に保ちます。

        Args:
            text: クエリテキスト。

        Keyword Args:
            text_scorer: テキストランキングに使用するスコアラー。
            filter_expression: パーティションフィルターとAND結合する追加のフィルター式。
            return_fields: 結果に返すフィールド。
            num_results: 最大結果数。
            alpha: ベクターが有効な場合のハイブリッドバランスパラメータ。

        Returns:
            結果辞書のリスト。

        Raises:
            ServiceInvalidRequestError: 入力が無効またはクエリが失敗した場合。

        """
        # 少なくとも1つのプロバイダーレベルのフィルターが存在することを強制する（Mem0Providerとの対称性）
        await self._ensure_index()
        self._validate_filters()

        q = (text or "").strip()
        if not q:
            raise ServiceInvalidRequestError("text_search() requires non-empty text")
        num_results = max(int(num_results or 10), 1)

        combined_filter = self._build_filter_from_dict({
            "application_id": self.application_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "thread_id": self._effective_thread_id,
            "conversation_id": self._conversation_id,
        })

        if filter_expression is not None:
            combined_filter = (combined_filter & filter_expression) if combined_filter else filter_expression

        # 返却フィールドを選択する
        return_fields = (
            return_fields
            if return_fields is not None
            else ["content", "role", "application_id", "agent_id", "user_id", "thread_id"]
        )

        try:
            if self.redis_vectorizer and self.vector_field_name:
                # ハイブリッドクエリを構築する：フルテキストとベクター類似度を組み合わせる
                vector = await self.redis_vectorizer.aembed(q)
                query = HybridQuery(
                    text=q,
                    text_field_name="content",
                    vector=vector,
                    vector_field_name=self.vector_field_name,
                    text_scorer=text_scorer,
                    filter_expression=combined_filter,
                    alpha=alpha,
                    dtype=self.redis_vectorizer.dtype,
                    num_results=num_results,
                    return_fields=return_fields,
                    stopwords=None,
                )
                hybrid_results = await self.redis_index.query(query)
                return cast(list[dict[str, Any]], hybrid_results)
            # テキストのみの検索
            query = TextQuery(
                text=q,
                text_field_name="content",
                text_scorer=text_scorer,
                filter_expression=combined_filter,
                num_results=num_results,
                return_fields=return_fields,
                stopwords=None,
            )
            text_results = await self.redis_index.query(query)
            return cast(list[dict[str, Any]], text_results)
        except Exception as exc:  # pragma: no cover - surface as framework error
            raise ServiceInvalidRequestError(f"Redis text search failed: {exc}") from exc

    async def search_all(self, page_size: int = 200) -> list[dict[str, Any]]:
        """インデックス内のすべてのドキュメントを返します。

        ページネーションによるストリーミングで過剰なメモリ使用やレスポンスサイズを回避します。

        Args:
            page_size: 内部でページネーションに使用されるページサイズ。

        Returns:
            すべてのドキュメントのリスト。

        """
        out: list[dict[str, Any]] = []
        async for batch in self.redis_index.paginate(
            FilterQuery(FilterExpression("*"), return_fields=[], num_results=page_size),
            page_size=page_size,
        ):
            out.extend(batch)
        return out

    @property
    def _effective_thread_id(self) -> str | None:
        """アクティブなスレッドIDを解決します。

        スコープが有効な場合は操作ごとのスレッドIDを返し、そうでなければプロバイダーのスレッドIDを返します。

        """
        return self._per_operation_thread_id if self.scope_to_per_operation_thread_id else self.thread_id

    @override
    async def thread_created(self, thread_id: str | None) -> None:
        """新しいスレッドが作成されたときに呼び出されます。

        スコープが有効な場合に、単一スレッド使用を強制するために操作ごとのスレッドIDをキャプチャします。

        Args:
            thread_id: スレッドのIDまたはNone。

        """
        self._validate_per_operation_thread_id(thread_id)
        self._per_operation_thread_id = self._per_operation_thread_id or thread_id
        # 現在の会話IDを追跡します（Agentがここにconversation_idを渡します）
        self._conversation_id = thread_id or self._conversation_id

    @override
    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        self._validate_filters()

        request_messages_list = (
            [request_messages] if isinstance(request_messages, ChatMessage) else list(request_messages)
        )
        response_messages_list = (
            [response_messages]
            if isinstance(response_messages, ChatMessage)
            else list(response_messages)
            if response_messages
            else []
        )
        messages_list = [*request_messages_list, *response_messages_list]

        messages: list[dict[str, Any]] = []
        for message in messages_list:
            if (
                message.role.value in {Role.USER.value, Role.ASSISTANT.value, Role.SYSTEM.value}
                and message.text
                and message.text.strip()
            ):
                shaped: dict[str, Any] = {
                    "role": message.role.value,
                    "content": message.text,
                    "conversation_id": self._conversation_id,
                    "message_id": message.message_id,
                    "author_name": message.author_name,
                }
                messages.append(shaped)
        if messages:
            await self._add(data=messages)

    @override
    async def invoking(self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any) -> Context:
        """モデルを呼び出す前にスコープ付きコンテキストを提供するために呼び出されます。

        最近のメッセージをクエリに連結し、Redisから一致するメモリを取得します。
        それらを指示として先頭に追加します。

        Args:
            messages: スレッド内の新しいメッセージのリスト。

        Keyword Args:
            **kwargs: 現時点では使用されていません。

        Returns:
            Context: メモリを含む指示を持つContextオブジェクト。

        """
        self._validate_filters()
        messages_list = [messages] if isinstance(messages, ChatMessage) else list(messages)
        input_text = "\n".join(msg.text for msg in messages_list if msg and msg.text and msg.text.strip())

        memories = await self._redis_search(text=input_text)
        line_separated_memories = "\n".join(
            str(memory.get("content", "")) for memory in memories if memory.get("content")
        )

        return Context(
            messages=[ChatMessage(role="user", text=f"{self.context_prompt}\n{line_separated_memories}")]
            if line_separated_memories
            else None
        )

    async def __aenter__(self) -> Self:
        """非同期コンテキストマネージャのエントリー。

        特別なセットアップは不要です。Mem0プロバイダーとの対称性のために提供されています。

        """
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """非同期コンテキストマネージャの終了。

        クリーンアップは不要です。インデックスやキーは明示的にクリアされない限り残ります。

        """
        return

    def _validate_filters(self) -> None:
        """少なくとも1つのフィルターが提供されていることを検証します。

        読み書きの前にパーティションフィルターを要求することで無制限の操作を防ぎます。

        Raises:
            ServiceInitializationError: フィルターが提供されていない場合。

        """
        if not self.agent_id and not self.user_id and not self.application_id and not self.thread_id:
            raise ServiceInitializationError(
                "At least one of the filters: agent_id, user_id, application_id, or thread_id is required."
            )

    def _validate_per_operation_thread_id(self, thread_id: str | None) -> None:
        """スコープが有効な場合に新しいスレッドIDが競合しないことを検証します。

        操作ごとのスコープが有効な場合に単一スレッド使用を強制し、スレッド間のデータ漏洩を防ぎます。

        Args:
            thread_id: 新しいスレッドIDまたはNone。

        Raises:
            ValueError: 新しいスレッドIDが既存のものと競合する場合。

        """
        if (
            self.scope_to_per_operation_thread_id
            and thread_id
            and self._per_operation_thread_id
            and thread_id != self._per_operation_thread_id
        ):
            raise ValueError(
                "RedisProvider can only be used with one thread, when scope_to_per_operation_thread_id is True."
            )
