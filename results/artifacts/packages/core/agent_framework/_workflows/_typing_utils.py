# Copyright (c) Microsoft. All rights reserved.

import logging
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _coerce_to_type(value: Any, target_type: type[T]) -> T | None:
    """値をtarget_typeにベストエフォートで変換します。

    Args:
        value: 変換対象の値（dict、dataclass、または__dict__を持つオブジェクト）
        target_type: 変換先の型

    Returns:
        変換に成功した場合はtarget_typeのインスタンス、失敗した場合はNone

    """
    if isinstance(value, target_type):
        return value  # type: ignore[return-value]

    # dataclassインスタンスや__dict__を持つオブジェクトをまずdictに変換します。
    value_as_dict: dict[str, Any]
    if not isinstance(value, dict):
        if is_dataclass(value):
            value_as_dict = {f.name: getattr(value, f.name) for f in fields(value)}
        else:
            value_dict = getattr(value, "__dict__", None)
            if isinstance(value_dict, dict):
                value_as_dict = cast(dict[str, Any], value_dict)
            else:
                return None
    else:
        value_as_dict = cast(dict[str, Any], value)

    # dictからtarget_typeのインスタンスを構築しようとします。
    ctor_kwargs: dict[str, Any] = dict(value_as_dict)

    if is_dataclass(target_type):
        field_names = {f.name for f in fields(target_type)}
        ctor_kwargs = {k: v for k, v in value_as_dict.items() if k in field_names}

    try:
        return target_type(**ctor_kwargs)  # type: ignore[call-arg,return-value]
    except TypeError as exc:
        logger.debug(f"_coerce_to_type could not call {target_type.__name__}(**..): {exc}")
    except Exception as exc:  # pragma: no cover - unexpected constructor failure
        logger.warning(
            f"_coerce_to_type encountered unexpected error calling {target_type.__name__} constructor: {exc}"
        )

    # フォールバック：__init__なしでインスタンスを作成し属性を設定しようとします。
    try:
        instance = object.__new__(target_type)
    except Exception as exc:  # pragma: no cover - pathological type
        logger.debug(f"_coerce_to_type could not allocate {target_type.__name__} without __init__: {exc}")
        return None

    for key, val in value_as_dict.items():
        try:
            setattr(instance, key, val)
        except Exception as exc:
            logger.debug(
                f"_coerce_to_type could not set {target_type.__name__}.{key} during fallback assignment: {exc}"
            )
            continue
    return instance  # type: ignore[return-value]


def is_instance_of(data: Any, target_type: type | UnionType | Any) -> bool:
    """データがtarget_typeのインスタンスかどうかをチェックします。

    Args:
        data (Any): チェック対象のデータ。
        target_type (type): チェック対象の型。

    Returns:
        bool: dataがtarget_typeのインスタンスならTrue、そうでなければFalse。

    """
    # ケース0: target_typeがAnyの場合 - 常にTrueを返す
    if target_type is Any:
        return True

    origin = get_origin(target_type)
    args = get_args(target_type)

    # ケース1: originがNoneの場合、target_typeはジェネリック型ではないことを意味する
    if origin is None:
        return isinstance(data, target_type)

    # ケース2: target_typeがOptional[T]またはUnion[T1, T2, ...]の場合 Optional[T]は実際にはUnion[T,
    # None]と同じである
    if origin is UnionType:
        return any(is_instance_of(data, arg) for arg in args)

    # ケース2b: typing.Union（レガシーなUnion構文）を処理する
    if origin is Union:
        return any(is_instance_of(data, arg) for arg in args)

    # ケース3: target_typeがジェネリック型の場合
    if origin in [list, set]:
        return isinstance(data, origin) and (
            not args or all(any(is_instance_of(item, arg) for arg in args) for item in data)  # type: ignore[misc]
        )  # type: ignore

    # ケース4: target_typeがタプルの場合
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:  # Tuple[T, ...] case
            element_type = args[0]
            return isinstance(data, tuple) and all(is_instance_of(item, element_type) for item in data)  # type: ignore[misc]
        if len(args) == 1 and args[0] is Ellipsis:  # Tuple[...] case
            return isinstance(data, tuple)
        if len(args) == 0:
            return isinstance(data, tuple)
        return (
            isinstance(data, tuple)
            and len(data) == len(args)  # type: ignore
            and all(is_instance_of(item, arg) for item, arg in zip(data, args, strict=False))  # type: ignore
        )

    # ケース5: target_typeが辞書の場合
    if origin is dict:
        return isinstance(data, dict) and (
            not args
            or all(
                is_instance_of(key, args[0]) and is_instance_of(value, args[1])
                for key, value in data.items()  # type: ignore
            )
        )

    # ケース6: target_typeがRequestResponse[T, U]の場合 - ジェネリックパラメータを検証する
    if origin and hasattr(origin, "__name__") and origin.__name__ == "RequestResponse":
        if not isinstance(data, origin):
            return False
        # RequestResponse[TRequest, TResponse]のジェネリックパラメータを検証する
        if len(args) >= 2:
            request_type, response_type = args[0], args[1]
            # original_requestがTRequestに一致し、dataがTResponseに一致するかをチェックする
            if (
                hasattr(data, "original_request")
                and data.original_request is not None
                and not is_instance_of(data.original_request, request_type)
            ):
                # Checkpointのデコードではoriginal_requestが単純なマッピングのままになることがある。
                # その場合、下流のハンドラやバリデータが完全に型付けされたRequestResponseインスタンスを受け取れるように、
                # 期待されるリクエスト型に強制変換する。
                original_request = data.original_request
                if isinstance(original_request, Mapping):
                    coerced = _coerce_to_type(dict(original_request), request_type)  # type: ignore[arg-type]
                    if coerced is None or not isinstance(coerced, request_type):
                        return False
                    data.original_request = coerced
                else:
                    return False
            if hasattr(data, "data") and data.data is not None and not is_instance_of(data.data, response_type):
                return False
        return True

    # ケース7: その他のカスタムジェネリッククラス - origin型のみをチェックする
    # ジェネリッククラスの場合、dataがorigin型のインスタンスかどうかをチェックする
    # ジェネリックパラメータの検証は型システムで処理されるため、ランタイムでは行わない
    if origin and hasattr(origin, "__name__"):
        return isinstance(data, origin)

    # フォールバック: ここに到達した場合、dataがtarget_typeのインスタンスであると仮定する
    return isinstance(data, target_type)


def is_type_compatible(source_type: type | UnionType | Any, target_type: type | UnionType | Any) -> bool:
    """source_typeがtarget_typeと互換性があるかをチェックする。

    ある型が互換性があるとは、source_typeの値をtarget_typeの変数に代入できることを意味する。
    例えば:
    - list[ChatMessage]はlist[str | ChatMessage]と互換性がある
    - strはstr | intと互換性がある
    - intはAnyと互換性がある

    Args:
        source_type: 代入元の型
        target_type: 代入先の型

    Returns:
        bool: source_typeがtarget_typeと互換性があればTrue、そうでなければFalse

    """
    # ケース0: target_typeがAnyの場合 - 常に互換性あり
    if target_type is Any:
        return True

    # ケース1: 完全な型一致の場合
    if source_type == target_type:
        return True

    source_origin = get_origin(source_type)
    source_args = get_args(source_type)
    target_origin = get_origin(target_type)
    target_args = get_args(target_type)

    # ケース2: targetがUnion/Optionalの場合 - sourceが任意のtargetメンバーに一致すれば互換性あり
    if target_origin is Union or target_origin is UnionType:
        # 特別なケース: sourceもUnionの場合、各sourceメンバーが少なくとも1つのtargetメンバーと互換性があるかチェックする
        if source_origin is Union or source_origin is UnionType:
            return all(
                any(is_type_compatible(source_arg, target_arg) for target_arg in target_args)
                for source_arg in source_args
            )
        # sourceがUnionでない場合、任意のtargetメンバーと互換性があるかチェックする
        return any(is_type_compatible(source_type, arg) for arg in target_args)

    # ケース3: sourceがUnionでtargetがUnionでない場合 - 各sourceメンバーがtargetと互換性がある必要がある
    if source_origin is Union or source_origin is UnionType:
        return all(is_type_compatible(arg, target_type) for arg in source_args)

    # ケース4: 両方とも非ジェネリック型の場合
    if source_origin is None and target_origin is None:
        # issubclassは両方が実際の型であり、UnionTypeやAnyでない場合のみ呼び出す
        if isinstance(source_type, type) and isinstance(target_type, type):
            try:
                return issubclass(source_type, target_type)
            except TypeError:
                # issubclassが機能しないケース（特殊なフォームなど）を処理する
                return False
        return source_type == target_type

    # ケース5: 異なるコンテナ型は互換性がない
    if source_origin != target_origin:
        return False

    # ケース6: 同じコンテナ型 - ジェネリック引数をチェックする
    if source_origin in [list, set]:
        if not source_args and not target_args:
            return True  # 両方とも型指定なしの場合
        if not source_args or not target_args:
            return True  # 一方が型指定なしの場合 - 互換性ありと仮定する
        # コレクションの場合、sourceの要素型はtargetの要素型と互換性がある必要がある
        return is_type_compatible(source_args[0], target_args[0])

    # ケース7: タプルの互換性
    if source_origin is tuple:
        if not source_args and not target_args:
            return True  # 両方とも型指定なしのタプルの場合
        if not source_args or not target_args:
            return True  # 一方が型指定なしの場合 - 互換性ありと仮定する

        # Tuple[T, ...]（可変長）を処理する
        if len(source_args) == 2 and source_args[1] is Ellipsis:
            if len(target_args) == 2 and target_args[1] is Ellipsis:
                return is_type_compatible(source_args[0], target_args[0])
            return False  # 可変長は固定長に代入できない

        if len(target_args) == 2 and target_args[1] is Ellipsis:
            # 固定長は要素型が互換性があれば可変長に代入できる
            return all(is_type_compatible(source_arg, target_args[0]) for source_arg in source_args)

        # 固定長タプルは長さが同じで要素型が互換性がある必要がある
        if len(source_args) != len(target_args):
            return False
        return all(is_type_compatible(s_arg, t_arg) for s_arg, t_arg in zip(source_args, target_args, strict=False))

    # ケース8: 辞書の互換性
    if source_origin is dict:
        if not source_args and not target_args:
            return True  # 両方とも型指定なしの辞書の場合
        if not source_args or not target_args:
            return True  # 一方が型指定なしの場合 - 互換性ありと仮定する
        if len(source_args) != 2 or len(target_args) != 2:
            return False  # 不正な辞書型の場合
        # キーと値の型の両方が互換性がある必要がある
        return is_type_compatible(source_args[0], target_args[0]) and is_type_compatible(source_args[1], target_args[1])

    # ケース9: カスタムジェネリッククラス - originが同じで引数が互換性があるかチェックする
    if source_origin and target_origin and source_origin == target_origin:
        if not source_args and not target_args:
            return True  # 両方とも型指定なしのジェネリックの場合
        if not source_args or not target_args:
            return True  # 一方が型指定なしの場合 - 互換性ありと仮定する
        if len(source_args) != len(target_args):
            return False  # 型パラメータの数が異なる場合
        return all(is_type_compatible(s_arg, t_arg) for s_arg, t_arg in zip(source_args, target_args, strict=False))

    # ケース10: フォールバック - sourceがtargetのサブクラスかチェックする（非ジェネリック型の場合）
    if source_origin is None and target_origin is None:
        try:
            # issubclassは両方が実際の型であり、UnionTypeやAnyでない場合のみ呼び出す
            if isinstance(source_type, type) and isinstance(target_type, type):
                return issubclass(source_type, target_type)
            return source_type == target_type
        except TypeError:
            return False

    return False
