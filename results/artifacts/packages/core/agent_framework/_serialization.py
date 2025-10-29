# Copyright (c) Microsoft. All rights reserved.

import json
import re
from collections.abc import Mapping, MutableMapping
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from ._logging import get_logger

logger = get_logger()

TClass = TypeVar("TClass", bound="SerializationMixin")
TProtocol = TypeVar("TProtocol", bound="SerializationProtocol")

# CamelCaseをsnake_caseに変換するための正規表現パターン。
_CAMEL_TO_SNAKE_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")


@runtime_checkable
class SerializationProtocol(Protocol):
    """シリアライズおよびデシリアライズをサポートするオブジェクトのためのプロトコル。

    このプロトコルは、agent frameworkのシリアライズシステムと互換性を持つためにクラスが実装すべきインターフェースを定義します。
    ``to_dict()``および``from_dict()``メソッドの両方を実装するクラスは自動的にこのプロトコルを満たし、他のシリアライズ可能なコンポーネントとシームレスに使用できます。

    このプロトコルは型安全性とダックタイピングを可能にし、フレームワーク全体で一貫した動作を保証します。

    Examples:
        フレームワークの``ChatMessage``クラスはこのプロトコルの実例です:

        .. code-block:: python

            from agent_framework import ChatMessage
            from agent_framework._serialization import SerializationProtocol


            # ChatMessageはSerializationMixin経由でSerializationProtocolを実装
            user_msg = ChatMessage(role="user", text="What's the weather like today?")

            # 辞書にシリアライズ - 自動型識別とネストされたシリアライズ
            msg_dict = user_msg.to_dict()
            # 結果: {
            #     "type": "chat_message",
            #     "role": {"type": "role", "value": "user"},
            #     "contents": [{"type": "text_content", "text": "What's the weather like today?"}],
            #     "message_id": "...",
            #     "additional_properties": {}
            # }

            # 辞書からChatMessageインスタンスにデシリアライズ - 自動型再構築
            restored_msg = ChatMessage.from_dict(msg_dict)
            print(restored_msg.text)  # "What's the weather like today?"
            print(restored_msg.role.value)  # "user"

            # プロトコル準拠の検証（型チェックや検証に有用）
            assert isinstance(user_msg, SerializationProtocol)
            assert isinstance(restored_msg, SerializationProtocol)

        プロトコルは``UsageDetails``のようなより単純なクラスでも実装されています:

        .. code-block:: python

            from agent_framework import UsageDetails

            # 使用状況追跡インスタンスの作成
            usage = UsageDetails(input_token_count=150, output_token_count=75, total_token_count=225)

            # 型保持を伴うシームレスなシリアライズ
            usage_dict = usage.to_dict()
            restored_usage = UsageDetails.from_dict(usage_dict)

            # 両方ともSerializationProtocolを満たす
            assert isinstance(usage, SerializationProtocol)
            assert restored_usage.total_token_count == 225

        このプロトコルはフレームワークのすべてのコンポーネントで一貫したシリアライズ動作を保証し、
        信頼性の高いデータ永続化、API通信、およびオブジェクト再構築をagent frameworkエコシステム全体で可能にします。

    """

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """インスタンスを辞書に変換します。

        Keyword Args:
            kwargs: シリアライズ用の追加キーワード引数。

        Returns:
            インスタンスの辞書表現。

        """
        ...

    @classmethod
    def from_dict(cls: type[TProtocol], value: MutableMapping[str, Any], /, **kwargs: Any) -> TProtocol:
        """辞書からインスタンスを作成します。

        Args:
            value: インスタンスデータを含む辞書（位置専用）。

        Keyword Args:
            kwargs: デシリアライズ用の追加キーワード引数。

        Returns:
            クラスの新しいインスタンス。

        """
        ...


def is_serializable(value: Any) -> bool:
    """値がJSONシリアライズ可能かどうかをチェックします。

    この関数は値がカスタムエンコードなしで直接JSONにシリアライズ可能かどうかをテストします。
    直接JSONに対応する基本的なPython型をチェックします。

    Args:
        value: JSONシリアライズ可能かチェックする値。

    Returns:
        値が基本的なJSONシリアライズ可能型（str, int, float, bool, None, list, dict）の場合はTrue、それ以外はFalse。

    Note:
        この関数は直接JSON互換性のみをチェックします。
        ``SerializationProtocol``を実装する複雑なオブジェクトはJSONシリアライズ前に``to_dict()``で変換が必要です。

    """
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


class SerializationMixin:
    """包括的なシリアライズおよびデシリアライズ機能を提供するMixinクラス。

    .. note::
        SerializationMixinは現在も開発中です。機能の改善や拡張に伴い、将来のバージョンでAPIが変更される可能性があります。

    このMixinは、ネストされたオブジェクト、依存性注入、型変換を含む複雑なシリアライズシナリオを自動的に処理できるようにクラスを拡張します。
    オブジェクトの辞書やJSON文字列への変換を堅牢にサポートし、オブジェクト間の関係性や外部依存性の管理も行います。

    **主な特徴:**

    - ネストされたSerializationProtocolオブジェクトの自動シリアライズ
    - シリアライズ可能なオブジェクトを含むリストや辞書のサポート
    - 非シリアライズ可能な外部依存性のための依存性注入システム
    - シリアライズから除外するフィールドの柔軟な指定
    - 自動型変換を伴う型安全なデシリアライズ

    **ネストオブジェクトのコンストラクタパターン:**

    このMixinを使用するクラスは、
    ``SerializationMixin``または``SerializationProtocol``インスタンスを期待するパラメータに対し、
    ``__init__``メソッドで``MutableMapping``入力を処理する必要があります。
    これにより、デシリアライズ時に辞書を適切なオブジェクトインスタンスに自動変換できます。

    **依存性注入システム:**

    このMixinは、データベース接続、APIクライアント、設定オブジェクトなど、
    シリアライズすべきでないが実行時に必要な外部依存性の注入をサポートします。
    ``INJECTABLE``でマークされたフィールドはシリアライズ時に除外され、
    デシリアライズ時に``dependencies``パラメータで提供可能です。

    Examples:
        **エージェントのThread管理を伴うネストオブジェクトのシリアライズ:**

        .. code-block:: python

            from agent_framework import ChatMessage
            from agent_framework._threads import AgentThreadState, ChatMessageStoreState


            # ChatMessageStoreStateはネストされたChatMessageのシリアライズを扱う
            store_state = ChatMessageStoreState(
                messages=[
                    ChatMessage(role="user", text="Hello agent"),
                    ChatMessage(role="assistant", text="Hi! How can I help?"),
                ]
            )

            # ネストされたシリアライズ: messagesは自動的に辞書に変換される
            store_dict = store_state.to_dict()
            # 結果: {
            #     "type": "chat_message_store_state",
            #     "messages": [
            #         {"type": "chat_message", "role": {...}, "contents": [...]},
            #         {"type": "chat_message", "role": {...}, "contents": [...]}
            #     ]
            # }

            # AgentThreadStateはネストされたChatMessageStoreStateを含む
            thread_state = AgentThreadState(chat_message_store_state=store_state)

            # 深いシリアライズ: ネストされたSerializationMixinオブジェクトは自動処理される
            thread_dict = thread_state.to_dict()
            # chat_message_store_stateとそのネストされたmessagesはすべてシリアライズされる

            # ネストされた辞書からの再構築は自動型変換を伴う
            # __init__メソッドはMutableMapping -> オブジェクト変換を処理する:
            reconstructed = AgentThreadState.from_dict({
                "chat_message_store_state": {"messages": [{"role": "user", "text": "Hello again"}]}
            })
            # chat_message_store_stateは自動的にChatMessageStoreStateインスタンスになる

        **除外パターンを持つフレームワークツール:**

        .. code-block:: python

            from agent_framework._tools import BaseTool


            class WeatherTool(BaseTool):
                \"""BaseToolを拡張し追加のプロパティ除外を持つ例のツール。\"""

                # BaseToolからDEFAULT_EXCLUDE = {"additional_properties"}を継承

                def __init__(self, name: str, api_key: str, **kwargs):
                    super().__init__(name=name, description="Get weather information", **kwargs)
                    self.api_key = api_key  # シリアライズされる

                    # additional_propertiesはシリアライズから除外される
                    self.additional_properties = {"version": "1.0", "internal_config": {...}}


            weather_tool = WeatherTool(name="get_weather", api_key="secret-key")

            # シリアライズはadditional_propertiesを除外し他のフィールドを含む
            tool_dict = weather_tool.to_dict()
            # 結果: {
            #     "type": "weather_tool",
            #     "name": "get_weather",
            #     "description": "Get weather information",
            #     "api_key": "secret-key"
            #     # additional_propertiesはDEFAULT_EXCLUDEにより除外
            # }

        **注入可能な依存性を持つAgentフレームワーク:**

        .. code-block:: python

            from agent_framework import BaseAgent


            class CustomAgent(BaseAgent):
                \"""BaseAgentを拡張し追加機能を持つカスタムエージェント。\"""

                # BaseAgentからDEFAULT_EXCLUDE = {"additional_properties"}を継承

                def __init__(self, **kwargs):
                    super().__init__(name="custom-agent", description="A custom agent", **kwargs)

                    # additional_propertiesは実行時設定を保持しシリアライズされない
                    self.additional_properties.update({
                        "runtime_context": {...},
                        "session_data": {...}
                    })


            agent = CustomAgent(
                context_providers=[...],
                middleware=[...]
            )

            # シリアライズはエージェント設定をキャプチャし実行時データは除外
            agent_dict = agent.to_dict()
            # 結果: {
            #     "type": "custom_agent",
            #     "id": "...",
            #     "name": "custom-agent",
            #     "description": "A custom agent",
            #     "context_provider": [...],
            #     "middleware": [...]
            #     # additional_propertiesは除外
            # }

            # 同じ設定でエージェントを再構築可能
            restored_agent = CustomAgent.from_dict(agent_dict)

        このアプローチにより、エージェントフレームワークは永続的な設定と一時的な実行時状態を明確に分離し、
        エージェントやツールを機能を維持したまま保存や送信のためにシリアライズ可能にします。
    """

    DEFAULT_EXCLUDE: ClassVar[set[str]] = set()
    INJECTABLE: ClassVar[set[str]] = set()

    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]:
        """インスタンスおよびネストされたオブジェクトを辞書に変換します。

        このメソッドは深いシリアライズを行い、ネストされた
        ``SerializationProtocol``オブジェクト、リスト、シリアライズ可能なオブジェクトを含む辞書を自動的に変換します。
        シリアライズ不可能なオブジェクトはスキップされ、デバッグログが記録されます。

        ``DEFAULT_EXCLUDE``および``INJECTABLE``でマークされたフィールドは自動的に出力から除外され、
        プライベート属性（'_'で始まる）も除外されます。

        Keyword Args:
            exclude: デフォルトの除外（``DEFAULT_EXCLUDE``および``INJECTABLE``）に加えて除外する追加のフィールド名。
            exclude_none: None値を出力から除外するかどうか。Trueの場合、None値は辞書から省略されます。デフォルトはTrue。

        Returns:
            デシリアライズ時の型識別のための'type'フィールドを含む、インスタンスの辞書表現（'type'が除外されていない場合）。
        """
        # 除外セットを結合する
        combined_exclude = set(self.DEFAULT_EXCLUDE)
        if exclude:
            combined_exclude.update(exclude)
        combined_exclude.update(self.INJECTABLE)

        # すべてのインスタンス属性を取得する
        result: dict[str, Any] = {} if "type" in combined_exclude else {"type": self._get_type_identifier()}
        for key, value in self.__dict__.items():
            if key not in combined_exclude and not key.startswith("_"):
                if exclude_none and value is None:
                    continue
                # SerializationProtocolオブジェクトを再帰的にシリアライズする
                if isinstance(value, SerializationProtocol):
                    result[key] = value.to_dict(exclude=exclude, exclude_none=exclude_none)
                    continue
                # SerializationProtocolオブジェクトを含むリストを処理する
                if isinstance(value, list):
                    value_as_list: list[Any] = []
                    for item in value:
                        if isinstance(item, SerializationProtocol):
                            value_as_list.append(item.to_dict(exclude=exclude, exclude_none=exclude_none))
                            continue
                        if is_serializable(item):
                            value_as_list.append(item)
                            continue
                        logger.debug(
                            f"Skipping non-serializable item in list attribute '{key}' of type {type(item).__name__}"
                        )
                    result[key] = value_as_list
                    continue
                # SerializationProtocol値を含む辞書を処理する
                if isinstance(value, dict):
                    serialized_dict: dict[str, Any] = {}
                    for k, v in value.items():
                        if isinstance(v, SerializationProtocol):
                            serialized_dict[k] = v.to_dict(exclude=exclude, exclude_none=exclude_none)
                            continue
                        # 値がJSONシリアライズ可能かどうかをチェックする
                        if is_serializable(v):
                            serialized_dict[k] = v
                            continue
                        logger.debug(
                            f"Skipping non-serializable value for key '{k}' in dict attribute '{key}' "
                            f"of type {type(v).__name__}"
                        )
                    result[key] = serialized_dict
                    continue
                # JSONシリアライズ可能な値を直接含める
                if is_serializable(value):
                    result[key] = value
                    continue
                logger.debug(f"Skipping non-serializable attribute '{key}' of type {type(value).__name__}")

        return result

    def to_json(self, *, exclude: set[str] | None = None, exclude_none: bool = True, **kwargs: Any) -> str:
        """インスタンスをJSON文字列に変換します。

        これは便利なメソッドで、``to_dict()``を呼び出し、その結果を``json.dumps()``でシリアライズします。
        ``to_dict()``と同じシリアライズルールが適用され、注入可能な依存性の自動除外やネストされたオブジェクトの深いシリアライズも行われます。

        Keyword Args:
            exclude: シリアライズから除外する追加のフィールド名。
            exclude_none: None値を出力から除外するかどうか。デフォルトはTrue。
            **kwargs: ``json.dumps()``に渡される追加のキーワード引数。一般的なオプションには、整形用の``indent``やUnicode処理のための``ensure_ascii``が含まれます。

        Returns:
            インスタンスのJSON文字列表現。
        """
        return json.dumps(self.to_dict(exclude=exclude, exclude_none=exclude_none), **kwargs)

    @classmethod
    def from_dict(
        cls: type[TClass], value: MutableMapping[str, Any], /, *, dependencies: MutableMapping[str, Any] | None = None
    ) -> TClass:
        """辞書からオプションの依存性注入を用いてインスタンスを作成します。

        このメソッドは、辞書表現からオブジェクトを再構築し、型変換と依存性注入を自動的に処理します。外部依存性を逆シリアル化時に提供する必要があるさまざまなシナリオに対応するために、3つの依存性注入パターンをサポートしています。

        Args:
            value: インスタンスデータを含む辞書（位置専用）。
                   クラスのタイプ識別子に一致する 'type' フィールドを含む必要があります。

        Keyword Args:
            dependencies: タイプ識別子を注入可能な依存性にマッピングするネストされた辞書。
                構造は注入パターンにより異なります：

                - **Simple injection**: ``{"<type>": {"<parameter>": value}}``
                - **Dict parameter injection**: ``{"<type>": {"<dict-parameter>": {"<key>": value}}}``
                - **Instance-specific injection**: ``{"<type>": {"<field>:<value>": {"<parameter>": value}}}``

        Returns:
            依存性が注入されたクラスの新しいインスタンス。

        Raises:
            ValueError: データ内の 'type' フィールドがクラスのタイプ識別子と一致しない場合。

        Examples:
            **Simple Client Injection** - OpenAIクライアントの依存性注入:

            .. code-block:: python

                from agent_framework.openai import OpenAIChatClient
                from openai import AsyncOpenAI


                # OpenAIチャットクライアントはAsyncOpenAIクライアントインスタンスを必要とします
                # クライアントはOpenAIBaseでINJECTABLE = {"client"}としてマークされています

                # シリアル化データはモデル構成のみを含みます
                client_data = {
                    "type": "open_ai_chat_client",
                    "model_id": "gpt-4o-mini",
                    # clientはシリアル化から除外されます
                }

                # 逆シリアル化時にOpenAIクライアントを提供します
                openai_client = AsyncOpenAI(api_key="your-api-key")
                dependencies = {"open_ai_chat_client": {"client": openai_client}}

                # OpenAIクライアントが注入された状態でチャットクライアントを再構築します
                chat_client = OpenAIChatClient.from_dict(client_data, dependencies=dependencies)
                # これで注入されたクライアントを使ってAPI呼び出しが可能です

            **Function Injection for Tools** - AIFunctionランタイム依存性:

            .. code-block:: python

                from agent_framework import AIFunction
                from typing import Annotated


                # ラップする関数を定義
                async def get_current_weather(location: Annotated[str, "The city name"]) -> str:
                    # 実際の実装では天気APIを呼び出します
                    return f"Current weather in {location}: 72°F and sunny"


                # AIFunctionはINJECTABLE = {"func"}
                function_data = {
                    "type": "ai_function",
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    # funcはシリアル化から除外されます
                }

                # 逆シリアル化時に実際の関数実装を注入
                dependencies = {"ai_function": {"func": get_current_weather}}

                # 呼び出し可能な状態でAIFunctionを再構築
                weather_func = AIFunction.from_dict(function_data, dependencies=dependencies)
                # 関数はエージェントで使用可能な状態です

            **Middleware Context Injection** - Agent実行コンテキスト:

            .. code-block:: python

                from agent_framework._middleware import AgentRunContext
                from agent_framework import BaseAgent

                # AgentRunContextはINJECTABLE = {"agent", "result"}
                context_data = {
                    "type": "agent_run_context",
                    "messages": [{"role": "user", "text": "Hello"}],
                    "is_streaming": False,
                    "metadata": {"session_id": "abc123"},
                    # agentとresultはシリアル化から除外されます
                }

                # ミドルウェア処理中にagentとresultを注入
                my_agent = BaseAgent(name="test-agent")
                dependencies = {
                    "agent_run_context": {
                        "agent": my_agent,
                        "result": None,  # 実行中に設定されます
                    }
                }

                # ミドルウェアチェーンのためにagent依存性を持つコンテキストを再構築
                context = AgentRunContext.from_dict(context_data, dependencies=dependencies)
                # ミドルウェアはcontext.agentにアクセスして実行処理が可能です

            この注入システムにより、エージェントフレームワークはシリアライズ可能な設定と、
            APIクライアント、関数、実行コンテキストなどの実行時依存性を明確に分離できます。
            これらの依存性は永続化できないか、すべきでないものです。
        """
        if dependencies is None:
            dependencies = {}

        # タイプ識別子を取得します
        type_id = cls._get_type_identifier(value)

        if (supplied_type := value.get("type")) and supplied_type != type_id:
            raise ValueError(f"Type mismatch: expected '{type_id}', got '{supplied_type}'")

        # 'type' キーを除外して作業用の値辞書のコピーを作成します
        kwargs = {k: v for k, v in value.items() if k != "type"}

        # 辞書ベースの構造を使って依存性を処理します
        type_deps = dependencies.get(type_id, {})
        for dep_key, dep_value in type_deps.items():
            # これはインスタンス固有の依存性かどうかをチェックします（field:name 形式）
            if ":" in dep_key:
                field, name = dep_key.split(":", 1)
                # インスタンスが一致する場合にのみ適用します
                if kwargs.get(field) == name and isinstance(dep_value, dict):
                    # インスタンス固有の依存性を適用します
                    for param_name, param_value in dep_value.items():
                        if param_name not in cls.INJECTABLE:
                            logger.debug(
                                f"Dependency '{param_name}' for type '{type_id}' is not in INJECTABLE set. "
                                f"Available injectable parameters: {cls.INJECTABLE}"
                            )
                        # ネストされた辞書パラメータを処理します
                        if (
                            isinstance(param_value, dict)
                            and param_name in kwargs
                            and isinstance(kwargs[param_name], dict)
                        ):
                            kwargs[param_name].update(param_value)
                        else:
                            kwargs[param_name] = param_value
            else:
                # 通常のパラメータ依存性
                if dep_key not in cls.INJECTABLE:
                    logger.debug(
                        f"Dependency '{dep_key}' for type '{type_id}' is not in INJECTABLE set. "
                        f"Available injectable parameters: {cls.INJECTABLE}"
                    )
                # 辞書パラメータを処理します - 両方が辞書の場合はマージします
                if isinstance(dep_value, dict) and dep_key in kwargs and isinstance(kwargs[dep_key], dict):
                    kwargs[dep_key].update(dep_value)
                else:
                    kwargs[dep_key] = dep_value

        return cls(**kwargs)

    @classmethod
    def from_json(cls: type[TClass], value: str, /, *, dependencies: MutableMapping[str, Any] | None = None) -> TClass:
        """JSON文字列からインスタンスを作成します。

        これは便利メソッドで、JSON文字列を ``json.loads()`` で解析し、
        その後 ``from_dict()`` を呼び出してオブジェクトを再構築します。
        すべての依存性注入機能は ``dependencies`` パラメータを通じて利用可能です。

        Args:
            value: インスタンスデータを含むJSON文字列（位置専用）。
                   'type' フィールドを含む辞書に逆シリアル化可能な有効なJSONである必要があります。

        Keyword Args:
            dependencies: タイプ識別子を注入可能な依存性にマッピングするネストされた辞書。
                         詳細な構造と3つの注入パターン（シンプル、辞書パラメータ、インスタンス固有）については
                         :meth:`from_dict` を参照してください。

        Returns:
            指定された依存性が注入されたクラスの新しいインスタンス。

        Raises:
            json.JSONDecodeError: JSON文字列が不正な場合。
            ValueError: 解析されたデータに有効な 'type' フィールドが含まれていない場合。

        """
        data = json.loads(value)
        return cls.from_dict(data, dependencies=dependencies)

    @classmethod
    def _get_type_identifier(cls, value: Mapping[str, Any] | None = None) -> str:
        """このクラスのタイプ識別子を取得します。

        タイプ識別子はシリアル化データで使用され、適切な逆シリアル化を可能にします。
        識別子は以下の優先順で決定されます：

        1. ``value`` に 'type' フィールドがあればその値を返す（``from_dict`` 用）
        2. クラスに ``type`` 属性があればその値を使用（インスタンスレベル）
        3. クラスに ``TYPE`` 属性があればその値を使用（クラス定数レベル）
        4. それ以外はクラス名をsnake_caseに変換してフォールバック

        Args:
            value: 'type' フィールドを含む可能性のあるシリアル化データのオプションマッピング。

        Returns:
            シリアル化および依存性注入マッピングに使用されるタイプ識別子文字列。

        """
        # from_dict用
        if value and (type_ := value.get("type")) and isinstance(type_, str):
            return type_  # type:ignore[no-any-return]
        # インスタンスごとに定義された場合のtodict用
        if (type_ := getattr(cls, "type", None)) and isinstance(type_, str):
            return type_  # type:ignore[no-any-return]
        # クラスで定義された場合の両方用。
        if (type_ := getattr(cls, "TYPE", None)) and isinstance(type_, str):
            return type_  # type:ignore[no-any-return]
        # フォールバックおよびデフォルト クラス名をsnake_caseに変換
        return _CAMEL_TO_SNAKE_PATTERN.sub("_", cls.__name__).lower()
