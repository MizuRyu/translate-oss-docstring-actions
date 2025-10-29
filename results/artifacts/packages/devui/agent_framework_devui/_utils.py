# Copyright (c) Microsoft. All rights reserved.

"""DevUIのユーティリティ関数。"""

import inspect
import json
import logging
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

from agent_framework import ChatMessage

logger = logging.getLogger(__name__)

# ============================================================================ Agent
# Metadata Extraction
# ============================================================================


def extract_agent_metadata(entity_object: Any) -> dict[str, Any]:
    """エンティティオブジェクトからエージェント固有のメタデータを抽出します。

    Args:
        entity_object: Agent Frameworkのエージェントオブジェクト

    Returns:
        エージェントメタデータを含む辞書：instructions、model、chat_client_type、
        context_providers、およびmiddleware

    """
    metadata = {
        "instructions": None,
        "model": None,
        "chat_client_type": None,
        "context_providers": None,
        "middleware": None,
    }

    # instructionsを取得しようとします
    if hasattr(entity_object, "chat_options") and hasattr(entity_object.chat_options, "instructions"):
        metadata["instructions"] = entity_object.chat_options.instructions

    # modelを取得しようとします - chat_optionsとchat_clientの両方をチェックします
    if (
        hasattr(entity_object, "chat_options")
        and hasattr(entity_object.chat_options, "model_id")
        and entity_object.chat_options.model_id
    ):
        metadata["model"] = entity_object.chat_options.model_id
    elif hasattr(entity_object, "chat_client") and hasattr(entity_object.chat_client, "model_id"):
        metadata["model"] = entity_object.chat_client.model_id

    # chat client typeを取得しようとします
    if hasattr(entity_object, "chat_client"):
        metadata["chat_client_type"] = entity_object.chat_client.__class__.__name__

    # context providersを取得しようとします
    if (
        hasattr(entity_object, "context_provider")
        and entity_object.context_provider
        and hasattr(entity_object.context_provider, "__class__")
    ):
        metadata["context_providers"] = [entity_object.context_provider.__class__.__name__]  # type: ignore

    # middlewareを取得しようとします
    if hasattr(entity_object, "middleware") and entity_object.middleware:
        middleware_list: list[str] = []
        for m in entity_object.middleware:
            # middlewareの良い名前を得るために複数の方法を試します
            if hasattr(m, "__name__"):  # Function or callable
                middleware_list.append(m.__name__)
            elif hasattr(m, "__class__"):  # Class instance
                middleware_list.append(m.__class__.__name__)
            else:
                middleware_list.append(str(m))
        metadata["middleware"] = middleware_list  # type: ignore

    return metadata


# ============================================================================ Workflow
# Input Type Utilities
# ============================================================================


def extract_executor_message_types(executor: Any) -> list[Any]:
    """指定されたexecutorの宣言された入力タイプを抽出します。

    Args:
        executor: Workflow executorオブジェクト

    Returns:
        executorが受け入れるメッセージタイプのリスト

    """
    message_types: list[Any] = []

    try:
        input_types = getattr(executor, "input_types", None)
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.debug(f"Failed to access executor input_types: {exc}")
    else:
        if input_types:
            message_types = list(input_types)

    if not message_types and hasattr(executor, "_handlers"):
        try:
            handlers = executor._handlers
            if isinstance(handlers, dict):
                message_types = list(handlers.keys())
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug(f"Failed to read executor handlers: {exc}")

    return message_types


def _contains_chat_message(type_hint: Any) -> bool:
    """提供された型ヒントが直接的または間接的にChatMessageを参照しているかどうかをチェックします。"""
    if type_hint is ChatMessage:
        return True

    origin = get_origin(type_hint)
    if origin in (list, tuple):
        return any(_contains_chat_message(arg) for arg in get_args(type_hint))

    if origin in (Union, UnionType):
        return any(_contains_chat_message(arg) for arg in get_args(type_hint))

    return False


def select_primary_input_type(message_types: list[Any]) -> Any | None:
    """workflowの入力に対して最もユーザーフレンドリーな入力タイプを選択します。

    ChatMessage（またはそのコンテナ）を優先し、その後プリミティブにフォールバックします。

    Args:
        message_types: 可能なメッセージタイプのリスト

    Returns:
        選択された主要な入力タイプ、リストが空の場合はNone

    """
    if not message_types:
        return None

    for message_type in message_types:
        if _contains_chat_message(message_type):
            return ChatMessage

    preferred = (str, dict)

    for candidate in preferred:
        for message_type in message_types:
            if message_type is candidate:
                return candidate
            origin = get_origin(message_type)
            if origin is candidate:
                return candidate

    return message_types[0]


# ============================================================================ Type
# System Utilities
# ============================================================================


def is_serialization_mixin(cls: type) -> bool:
    """クラスがSerializationMixinのサブクラスかどうかをチェックします。

    Args:
        cls: チェックするクラス

    Returns:
        クラスがSerializationMixinのサブクラスであればTrue

    """
    try:
        from agent_framework._serialization import SerializationMixin

        return isinstance(cls, type) and issubclass(cls, SerializationMixin)
    except ImportError:
        return False


def _type_to_schema(type_hint: Any, field_name: str) -> dict[str, Any]:
    """型ヒントをJSONスキーマに変換します。

    Args:
        type_hint: 変換する型ヒント
        field_name: フィールド名（ドキュメント用）

    Returns:
        JSONスキーマの辞書

    """
    type_str = str(type_hint)

    # None/Optionalを処理します
    if type_hint is type(None):
        return {"type": "null"}

    # 基本型を処理します
    if type_hint is str or "str" in type_str:
        return {"type": "string"}
    if type_hint is int or "int" in type_str:
        return {"type": "integer"}
    if type_hint is float or "float" in type_str:
        return {"type": "number"}
    if type_hint is bool or "bool" in type_str:
        return {"type": "boolean"}

    # Literal型を処理します（enumのような値用）
    if "Literal" in type_str:
        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            if args:
                return {"type": "string", "enum": list(args)}

    # Union/Optionalを処理します
    if "Union" in type_str or "Optional" in type_str:
        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            # None型を除外します
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return _type_to_schema(non_none_args[0], field_name)
            # 複数の型 - 最初のNoneでないものを選びます
            if non_none_args:
                return _type_to_schema(non_none_args[0], field_name)

    # コレクションを処理します
    if "list" in type_str or "List" in type_str or "Sequence" in type_str:
        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            if args:
                items_schema = _type_to_schema(args[0], field_name)
                return {"type": "array", "items": items_schema}
        return {"type": "array"}

    if "dict" in type_str or "Dict" in type_str or "Mapping" in type_str:
        return {"type": "object"}

    # デフォルトのフォールバック
    return {"type": "string", "description": f"Type: {type_hint}"}


def generate_schema_from_serialization_mixin(cls: type[Any]) -> dict[str, Any]:
    """SerializationMixinクラスからJSONスキーマを生成します。

    __init__のシグネチャを内省してパラメータの型とデフォルトを抽出します。

    Args:
        cls: SerializationMixinのサブクラス

    Returns:
        JSONスキーマの辞書

    """
    sig = inspect.signature(cls)

    # 型ヒントを取得します
    try:
        from typing import get_type_hints

        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "kwargs"):
            continue

        # 型注釈を取得します
        param_type = type_hints.get(param_name, str)

        # このパラメータのスキーマを生成します
        param_schema = _type_to_schema(param_type, param_name)
        properties[param_name] = param_schema

        # 必須かどうかをチェックします（デフォルト値なし、VAR_KEYWORDでない）
        if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}

    if required:
        schema["required"] = required

    return schema


def generate_schema_from_dataclass(cls: type[Any]) -> dict[str, Any]:
    """dataclassからJSONスキーマを生成します。

    Args:
        cls: Dataclassの型

    Returns:
        JSONスキーマの辞書

    """
    if not is_dataclass(cls):
        return {"type": "object"}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in fields(cls):
        # フィールドの型のスキーマを生成します
        field_schema = _type_to_schema(field.type, field.name)
        properties[field.name] = field_schema

        # 必須かどうかをチェックします（デフォルト値なし）
        if field.default == field.default_factory:  # No default
            required.append(field.name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}

    if required:
        schema["required"] = required

    return schema


def generate_input_schema(input_type: type) -> dict[str, Any]:
    """workflow入力タイプのJSONスキーマを生成します。

    優先順に複数の入力タイプをサポートします：
    1. 組み込み型（str、dict、intなど）
    2. Pydanticモデル（model_json_schema経由）
    3. SerializationMixinクラス（__init__内省経由）
    4. Dataclasses（fields内省経由）
    5. 文字列へのフォールバック

    Args:
        input_type: スキーマを生成する入力タイプ

    Returns:
        JSONスキーマの辞書

    """
    # 1. 組み込み型
    if input_type is str:
        return {"type": "string"}
    if input_type is dict:
        return {"type": "object"}
    if input_type is int:
        return {"type": "integer"}
    if input_type is float:
        return {"type": "number"}
    if input_type is bool:
        return {"type": "boolean"}

    # 2. Pydanticモデル（レガシーサポート）
    if hasattr(input_type, "model_json_schema"):
        return input_type.model_json_schema()  # type: ignore

    # 3. SerializationMixinクラス（ChatMessageなど）
    if is_serialization_mixin(input_type):
        return generate_schema_from_serialization_mixin(input_type)

    # 4. Dataclasses
    if is_dataclass(input_type):
        return generate_schema_from_dataclass(input_type)

    # 5. 文字列へのフォールバック
    type_name = getattr(input_type, "__name__", str(input_type))
    return {"type": "string", "description": f"Input type: {type_name}"}


# ============================================================================ Input
# Parsing Utilities
# ============================================================================


def parse_input_for_type(input_data: Any, target_type: type) -> Any:
    """入力データをターゲットタイプに合わせて解析します。

    生の入力（文字列、辞書）から期待される型への変換を処理します：
    - 組み込み型：直接変換
    - Pydanticモデル：model_validateまたはmodel_validate_jsonを使用
    - SerializationMixin：from_dictまたは文字列からの構築を使用
    - Dataclasses：辞書から構築

    Args:
        input_data: 生の入力データ（文字列、辞書、または既に正しい型）
        target_type: 入力の期待される型

    Returns:
        target_typeに一致する解析済み入力、解析に失敗した場合は元の入力

    """
    # 既に正しい型であればそのまま返します
    if isinstance(input_data, target_type):
        return input_data

    # 文字列入力を処理します
    if isinstance(input_data, str):
        return _parse_string_input(input_data, target_type)

    # 辞書入力を処理します
    if isinstance(input_data, dict):
        return _parse_dict_input(input_data, target_type)

    # フォールバック：元のまま返します
    return input_data


def _parse_string_input(input_str: str, target_type: type) -> Any:
    """文字列入力をターゲットタイプに解析します。

    Args:
        input_str: 入力文字列
        target_type: ターゲットタイプ

    Returns:
        解析済み入力または元の文字列

    """
    # 組み込み型
    if target_type is str:
        return input_str
    if target_type is int:
        try:
            return int(input_str)
        except ValueError:
            return input_str
    elif target_type is float:
        try:
            return float(input_str)
        except ValueError:
            return input_str
    elif target_type is bool:
        return input_str.lower() in ("true", "1", "yes")

    # Pydanticモデル
    if hasattr(target_type, "model_validate_json"):
        try:
            # まずJSONとして解析を試みます
            if input_str.strip().startswith("{"):
                return target_type.model_validate_json(input_str)  # type: ignore

            # 文字列値を持つ一般的なフィールド名を試します
            common_fields = ["text", "message", "content", "input", "data"]
            for field in common_fields:
                try:
                    return target_type(**{field: input_str})  # type: ignore
                except Exception as e:
                    logger.debug(f"Failed to parse string input with field '{field}': {e}")
                    continue
        except Exception as e:
            logger.debug(f"Failed to parse string as Pydantic model: {e}")

    # SerializationMixin（ChatMessageのような）
    if is_serialization_mixin(target_type):
        try:
            # まずJSON辞書として解析を試みます
            if input_str.strip().startswith("{"):
                data = json.loads(input_str)
                if hasattr(target_type, "from_dict"):
                    return target_type.from_dict(data)  # type: ignore
                return target_type(**data)  # type: ignore

            # 特にChatMessageの場合：テキストから作成 一般的なフィールドパターンを試します
            common_fields = ["text", "message", "content"]
            sig = inspect.signature(target_type)
            params = list(sig.parameters.keys())

            # 'text'パラメータがあればそれを使用します
            if "text" in params:
                try:
                    return target_type(role="user", text=input_str)  # type: ignore
                except Exception as e:
                    logger.debug(f"Failed to create SerializationMixin with text field: {e}")

            # 他の一般的なフィールドを試します
            for field in common_fields:
                if field in params:
                    try:
                        return target_type(**{field: input_str})  # type: ignore
                    except Exception as e:
                        logger.debug(f"Failed to create SerializationMixin with field '{field}': {e}")
                        continue
        except Exception as e:
            logger.debug(f"Failed to parse string as SerializationMixin: {e}")

    # Dataclasses
    if is_dataclass(target_type):
        try:
            # JSONとして解析を試みます
            if input_str.strip().startswith("{"):
                data = json.loads(input_str)
                return target_type(**data)  # type: ignore

            # 一般的なフィールド名を試します
            common_fields = ["text", "message", "content", "input", "data"]
            for field in common_fields:
                try:
                    return target_type(**{field: input_str})  # type: ignore
                except Exception as e:
                    logger.debug(f"Failed to create dataclass with field '{field}': {e}")
                    continue
        except Exception as e:
            logger.debug(f"Failed to parse string as dataclass: {e}")

    # フォールバック：元の文字列を返します
    return input_str


def _parse_dict_input(input_dict: dict[str, Any], target_type: type) -> Any:
    """辞書入力をターゲットタイプに解析します。

    Args:
        input_dict: 入力辞書
        target_type: ターゲットタイプ

    Returns:
        解析済み入力または元の辞書

    """
    # プリミティブ型を処理します - 一般的なフィールド名から抽出します
    if target_type in (str, int, float, bool):
        try:
            # 既に正しい型であればそのまま返します
            if isinstance(input_dict, target_type):
                return input_dict

            # まず"input"フィールドを試します（workflow入力で一般的）
            if "input" in input_dict:
                return target_type(input_dict["input"])  # type: ignore

            # 単一キーの辞書なら値を抽出します
            if len(input_dict) == 1:
                value = next(iter(input_dict.values()))
                return target_type(value)  # type: ignore

            # それ以外はそのまま返します
            return input_dict
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert dict to {target_type}: {e}")
            return input_dict

    # ターゲットが辞書ならそのまま返します
    if target_type is dict:
        return input_dict

    # Pydanticモデル
    if hasattr(target_type, "model_validate"):
        try:
            return target_type.model_validate(input_dict)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to validate dict as Pydantic model: {e}")

    # SerializationMixin
    if is_serialization_mixin(target_type):
        try:
            if hasattr(target_type, "from_dict"):
                return target_type.from_dict(input_dict)  # type: ignore
            return target_type(**input_dict)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to parse dict as SerializationMixin: {e}")

    # Dataclasses
    if is_dataclass(target_type):
        try:
            return target_type(**input_dict)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to parse dict as dataclass: {e}")

    # フォールバック：元の辞書を返します
    return input_dict
