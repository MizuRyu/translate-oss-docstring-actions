# Copyright (c) Microsoft. All rights reserved.

import contextlib
import importlib
import logging
import sys
from dataclasses import fields, is_dataclass
from typing import Any, cast

# チェックポイントのシリアライズヘルパー
MODEL_MARKER = "__af_model__"
DATACLASS_MARKER = "__af_dataclass__"

# 任意のユーザーデータのエンコード時に無限再帰を防ぐガード
_MAX_ENCODE_DEPTH = 100
_CYCLE_SENTINEL = "<cycle>"


logger = logging.getLogger(__name__)


def encode_checkpoint_value(value: Any) -> Any:
    """値を再帰的にJSONシリアライズ可能な構造にエンコードします。

    - to_dict/to_jsonを公開するオブジェクト -> { MODEL_MARKER: "module:Class", value: encoded }
    - dataclassインスタンス -> { DATACLASS_MARKER: "module:Class", value: {field: encoded} }
    - dict -> キーをstrに変換し値を再帰的にエンコード
    - list/tuple/set -> エンコードされたアイテムのリスト
    - その他 -> 既にJSONシリアライズ可能ならそのまま返す

    無限再帰を避けるためのサイクルおよび深さの保護を含みます。
    """

    def _enc(v: Any, stack: set[int], depth: int) -> Any:
        # 深さガード
        if depth > _MAX_ENCODE_DEPTH:
            logger.debug(f"Max encode depth reached at depth={depth} for type={type(v)}")
            return "<max_depth>"

        # 構造化モデルの処理（to_dict/to_jsonを公開するオブジェクト）
        if _supports_model_protocol(v):
            cls = cast(type[Any], type(v))  # type: ignore
            try:
                if hasattr(v, "to_dict") and callable(getattr(v, "to_dict", None)):
                    raw = v.to_dict()  # type: ignore[attr-defined]
                    strategy = "to_dict"
                elif hasattr(v, "to_json") and callable(getattr(v, "to_json", None)):
                    serialized = v.to_json()  # type: ignore[attr-defined]
                    if isinstance(serialized, (bytes, bytearray)):
                        try:
                            serialized = serialized.decode()
                        except Exception:
                            serialized = serialized.decode(errors="replace")
                    raw = serialized
                    strategy = "to_json"
                else:
                    raise AttributeError("Structured model lacks serialization hooks")
                return {
                    MODEL_MARKER: f"{cls.__module__}:{cls.__name__}",
                    "strategy": strategy,
                    "value": _enc(raw, stack, depth + 1),
                }
            except Exception as exc:  # best-effort fallback
                logger.debug(f"Structured model serialization failed for {cls}: {exc}")
                return str(v)

        # Dataclass（インスタンスのみ）
        if is_dataclass(v) and not isinstance(v, type):
            oid = id(v)
            if oid in stack:
                logger.debug("Cycle detected while encoding dataclass instance")
                return _CYCLE_SENTINEL
            stack.add(oid)
            try:
                # type(v)は十分に絞り込まれているため、castは冗長でした
                dc_cls: type[Any] = type(v)
                field_values: dict[str, Any] = {}
                for f in fields(v):  # type: ignore[arg-type]
                    field_values[f.name] = _enc(getattr(v, f.name), stack, depth + 1)
                return {
                    DATACLASS_MARKER: f"{dc_cls.__module__}:{dc_cls.__name__}",
                    "value": field_values,
                }
            finally:
                stack.remove(oid)

        # コレクション
        if isinstance(v, dict):
            v_dict = cast("dict[object, object]", v)
            oid = id(v_dict)
            if oid in stack:
                logger.debug("Cycle detected while encoding dict")
                return _CYCLE_SENTINEL
            stack.add(oid)
            try:
                json_dict: dict[str, Any] = {}
                for k_any, val_any in v_dict.items():  # type: ignore[assignment]
                    k_str: str = str(k_any)
                    json_dict[k_str] = _enc(val_any, stack, depth + 1)
                return json_dict
            finally:
                stack.remove(oid)

        if isinstance(v, (list, tuple, set)):
            iterable_v = cast("list[object] | tuple[object, ...] | set[object]", v)
            oid = id(iterable_v)
            if oid in stack:
                logger.debug("Cycle detected while encoding iterable")
                return _CYCLE_SENTINEL
            stack.add(oid)
            try:
                seq: list[object] = list(iterable_v)
                encoded_list: list[Any] = []
                for item in seq:
                    encoded_list.append(_enc(item, stack, depth + 1))
                return encoded_list
            finally:
                stack.remove(oid)

        # プリミティブ（または不明なオブジェクト）：JSONシリアライズ可能であることを保証
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        # フォールバック：JSONシリアライズエラーを避けるため不明なオブジェクトを文字列化
        try:
            return str(v)
        except Exception:
            return f"<{type(v).__name__}>"

    return _enc(value, set(), 0)


def decode_checkpoint_value(value: Any) -> Any:
    """encode_checkpoint_valueでエンコードされた値を再帰的にデコードします。"""
    if isinstance(value, dict):
        value_dict = cast(dict[str, Any], value)  # エンコード形式は常に文字列キーを使用します
        # 構造化モデルマーカーの処理
        if MODEL_MARKER in value_dict and "value" in value_dict:
            type_key: str | None = value_dict.get(MODEL_MARKER)  # type: ignore[assignment]
            strategy: str | None = value_dict.get("strategy")  # type: ignore[assignment]
            raw_encoded: Any = value_dict.get("value")
            decoded_payload = decode_checkpoint_value(raw_encoded)
            if isinstance(type_key, str):
                try:
                    cls = _import_qualified_name(type_key)
                except Exception as exc:
                    logger.debug(f"Failed to import structured model {type_key}: {exc}")
                    cls = None

                if cls is not None:
                    if strategy == "to_dict" and hasattr(cls, "from_dict"):
                        with contextlib.suppress(Exception):
                            return cls.from_dict(decoded_payload)
                    if strategy == "to_json" and hasattr(cls, "from_json"):
                        if isinstance(decoded_payload, (str, bytes, bytearray)):
                            with contextlib.suppress(Exception):
                                return cls.from_json(decoded_payload)
                        if isinstance(decoded_payload, dict) and hasattr(cls, "from_dict"):
                            with contextlib.suppress(Exception):
                                return cls.from_dict(decoded_payload)
            return decoded_payload
        # Dataclassマーカーの処理
        if DATACLASS_MARKER in value_dict and "value" in value_dict:
            type_key_dc: str | None = value_dict.get(DATACLASS_MARKER)  # type: ignore[assignment]
            raw_dc: Any = value_dict.get("value")
            decoded_raw = decode_checkpoint_value(raw_dc)
            if isinstance(type_key_dc, str):
                try:
                    module_name, class_name = type_key_dc.split(":", 1)
                    module = sys.modules.get(module_name)
                    if module is None:
                        module = importlib.import_module(module_name)
                    cls_dc: Any = getattr(module, class_name)
                    constructed = _instantiate_checkpoint_dataclass(cls_dc, decoded_raw)
                    if constructed is not None:
                        return constructed
                except Exception as exc:
                    logger.debug(f"Failed to decode dataclass {type_key_dc}: {exc}; returning raw value")
            return decoded_raw

        # 通常のdict：再帰的にデコード
        decoded: dict[str, Any] = {}
        for k_any, v_any in value_dict.items():
            decoded[k_any] = decode_checkpoint_value(v_any)
        return decoded
    if isinstance(value, list):
        # isinstanceチェック後、値をlist[Any]として扱いデコード
        value_list: list[Any] = value  # type: ignore[assignment]
        return [decode_checkpoint_value(v_any) for v_any in value_list]
    return value


def _instantiate_checkpoint_dataclass(cls: type[Any], payload: Any) -> Any | None:
    if not isinstance(cls, type):
        logger.debug(f"Checkpoint decoder received non-type dataclass reference: {cls!r}")
        return None

    if isinstance(payload, dict):
        try:
            return cls(**payload)  # type: ignore[arg-type]
        except TypeError as exc:
            logger.debug(f"Checkpoint decoder could not call {cls.__name__}(**payload): {exc}")
        except Exception as exc:
            logger.warning(f"Checkpoint decoder encountered unexpected error calling {cls.__name__}(**payload): {exc}")
        try:
            instance = object.__new__(cls)
        except Exception as exc:
            logger.debug(f"Checkpoint decoder could not allocate {cls.__name__} without __init__: {exc}")
            return None
        for key, val in payload.items():  # type: ignore[attr-defined]
            try:
                setattr(instance, key, val)  # type: ignore[arg-type]
            except Exception as exc:
                logger.debug(f"Checkpoint decoder could not set attribute {key} on {cls.__name__}: {exc}")
        return instance

    try:
        return cls(payload)  # type: ignore[call-arg]
    except TypeError as exc:
        logger.debug(f"Checkpoint decoder could not call {cls.__name__}({payload!r}): {exc}")
    except Exception as exc:
        logger.warning(f"Checkpoint decoder encountered unexpected error calling {cls.__name__}({payload!r}): {exc}")
    return None


def _supports_model_protocol(obj: object) -> bool:
    """辞書シリアライズフックを公開するオブジェクトを検出します。"""
    try:
        obj_type: type[Any] = type(obj)
    except Exception:
        return False

    has_to_dict = hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None))  # type: ignore[arg-type]
    has_from_dict = hasattr(obj_type, "from_dict") and callable(getattr(obj_type, "from_dict", None))

    has_to_json = hasattr(obj, "to_json") and callable(getattr(obj, "to_json", None))  # type: ignore[arg-type]
    has_from_json = hasattr(obj_type, "from_json") and callable(getattr(obj_type, "from_json", None))

    return (has_to_dict and has_from_dict) or (has_to_json and has_from_json)


def _import_qualified_name(qualname: str) -> type[Any] | None:
    if ":" not in qualname:
        return None
    module_name, class_name = qualname.split(":", 1)
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    attr: Any = module
    for part in class_name.split("."):
        attr = getattr(attr, part)
    return attr if isinstance(attr, type) else None
