# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union

from agent_framework._workflows import RequestInfoMessage, RequestResponse
from agent_framework._workflows._typing_utils import is_instance_of, is_type_compatible


def test_basic_types() -> None:
    """基本的な組み込み型をテストします。"""
    assert is_instance_of(5, int)
    assert is_instance_of("hello", str)
    assert is_instance_of(None, type(None))


def test_union_types() -> None:
    """union型（|）とoptional型をテストします。"""
    assert is_instance_of(5, int | str)
    assert is_instance_of("hello", int | str)
    assert is_instance_of(5, Union[int, str])
    assert not is_instance_of(5.0, int | str)


def test_list_types() -> None:
    """さまざまな要素型を持つlist型をテストします。"""
    assert is_instance_of([], list)
    assert is_instance_of([1, 2, 3], list)
    assert is_instance_of([1, 2, 3], list[int])
    assert is_instance_of([1, 2, 3], list[int | str])
    assert is_instance_of([1, "a", 3], list[int | str])
    assert is_instance_of([1, "a", 3], list[Union[int, str]])
    assert not is_instance_of([1, 2.0, 3], dict)
    assert not is_instance_of([1, 2.0, 3], list[int | str])


def test_tuple_types() -> None:
    """固定長および可変長のtuple型をテストします。"""
    assert is_instance_of((1, "a"), tuple)
    assert is_instance_of((1, "a"), tuple[int, str])
    assert is_instance_of((1, "a", 3), tuple[int | str, ...])
    assert is_instance_of((1, 2.0, "a"), tuple[...])  # type: ignore
    assert not is_instance_of((1, 2.0, 3), tuple[int | str, ...])
    assert not is_instance_of((1, 2.0, 3), dict)


def test_dict_types() -> None:
    """型付きのキーと値を持つdictionary型をテストします。"""
    assert is_instance_of({"key": "value"}, dict)
    assert is_instance_of({"key": "value"}, dict[str, str])
    assert is_instance_of({"key": 5, "another_key": "value"}, dict[str, int | str])
    assert not is_instance_of({"key": 5, "another_key": 3.0}, dict[str, int | str])
    assert not is_instance_of({"key": 5, "another_key": 3.0}, list)


def test_set_types() -> None:
    """さまざまな要素型を持つset型をテストします。"""
    assert is_instance_of({1, 2, 3}, set)
    assert is_instance_of({1, 2, 3}, set[int])
    assert is_instance_of({1, 2, 3}, set[int | str])
    assert is_instance_of({1, "a", 3}, set[int | str])
    assert is_instance_of({1, "a", 3}, set[Union[int, str]])
    assert is_instance_of(set(), set[int])
    assert not is_instance_of({1, 2.0, 3}, set[int | str])
    assert not is_instance_of({1, 2, 3}, list)
    assert not is_instance_of({1, 2, 3}, dict)


def test_any_type() -> None:
    """Any型をテストします - すべての値を受け入れるはずです。"""
    assert is_instance_of(5, Any)
    assert is_instance_of("hello", Any)
    assert is_instance_of([1, 2, 3], Any)


def test_nested_types() -> None:
    """複雑なネスト型構造をテストします。"""
    assert is_instance_of([{"key": [1, 2]}, {"another_key": [3]}], list[dict[str, list[int]]])
    assert not is_instance_of([{"key": [1, 2]}, {"another_key": [3.0]}], list[dict[str, list[int]]])


def test_custom_type() -> None:
    """カスタムオブジェクトの型チェックをテストします。"""

    @dataclass
    class CustomClass:
        value: int

    instance = CustomClass(10)
    assert is_instance_of(instance, CustomClass)
    assert not is_instance_of(instance, dict)


def test_request_response_type() -> None:
    """RequestResponseのジェネリック型チェックをテストします。"""

    request_instance = RequestResponse[RequestInfoMessage, str](
        data="approve",
        request_id="req-1",
        original_request=RequestInfoMessage(),
    )

    class CustomRequestInfoMessage(RequestInfoMessage):
        info: str

    assert is_instance_of(request_instance, RequestResponse[RequestInfoMessage, str])
    assert not is_instance_of(request_instance, RequestResponse[CustomRequestInfoMessage, str])


def test_custom_generic_type() -> None:
    """カスタムジェネリック型チェックをテストします。"""

    T = TypeVar("T")
    U = TypeVar("U")

    class CustomClass(Generic[T, U]):
        def __init__(self, request: T, response: U, extra: Any | None = None) -> None:
            self.request = request
            self.response = response
            self.extra = extra

    instance = CustomClass[int, str](request=5, response="response")

    assert is_instance_of(instance, CustomClass[int, str])
    # ジェネリックパラメータはランタイムで厳密には強制されません。
    assert is_instance_of(instance, CustomClass[str, str])


def test_edge_cases() -> None:
    """エッジケースや異常なシナリオをテストします。"""
    assert is_instance_of([], list[int])  # 空のリストは有効であるべきです。
    assert is_instance_of((), tuple[int, ...])  # 空のタプルは有効であるべきです。
    assert is_instance_of({}, dict[str, int])  # 空の辞書は有効であるべきです。
    assert is_instance_of(None, int | None)  # Noneを含むOptional型。
    assert not is_instance_of(5, str | None)  # 一致する型のないOptional型。


def test_type_compatibility_basic() -> None:
    """基本的な型互換性のシナリオをテストします。"""
    # 正確な型の一致。
    assert is_type_compatible(str, str)
    assert is_type_compatible(int, int)

    # Any互換性。
    assert is_type_compatible(str, Any)
    assert is_type_compatible(list[int], Any)

    # サブクラス互換性。
    class Animal:
        pass

    class Dog(Animal):
        pass

    assert is_type_compatible(Dog, Animal)
    assert not is_type_compatible(Animal, Dog)


def test_type_compatibility_unions() -> None:
    """Union型との型互換性をテストします。"""
    # ソースがターゲットのunionメンバーに一致します。
    assert is_type_compatible(str, Union[str, int])
    assert is_type_compatible(int, Union[str, int])
    assert not is_type_compatible(float, Union[str, int])

    # ソースunion - すべてのメンバーがターゲットと互換性が必要です。
    assert is_type_compatible(Union[str, int], Union[str, int, float])
    assert not is_type_compatible(Union[str, int, bytes], Union[str, int])


def test_type_compatibility_collections() -> None:
    """コレクション型との型互換性をテストします。"""

    # List互換性 - 主要なユースケース。
    @dataclass
    class ChatMessage:
        text: str

    assert is_type_compatible(list[ChatMessage], list[Union[str, ChatMessage]])
    assert is_type_compatible(list[str], list[Union[str, ChatMessage]])
    assert not is_type_compatible(list[Union[str, ChatMessage]], list[ChatMessage])

    # Dict互換性。
    assert is_type_compatible(dict[str, int], dict[str, Union[int, float]])
    assert not is_type_compatible(dict[str, Union[int, float]], dict[str, int])

    # Set互換性。
    assert is_type_compatible(set[str], set[Union[str, int]])
    assert not is_type_compatible(set[Union[str, int]], set[str])


def test_type_compatibility_tuples() -> None:
    """タプル型との型互換性をテストします。"""
    # 固定長タプル。
    assert is_type_compatible(tuple[str, int], tuple[Union[str, bytes], Union[int, float]])
    assert not is_type_compatible(tuple[str, int], tuple[str, int, bool])  # 異なる長さ。

    # 可変長タプル。
    assert is_type_compatible(tuple[str, ...], tuple[Union[str, bytes], ...])
    assert is_type_compatible(tuple[str, int, bool], tuple[Union[str, int, bool], ...])
    assert not is_type_compatible(tuple[str, ...], tuple[str, int])  # 可変長から固定長へ。


def test_type_compatibility_complex() -> None:
    """複雑なネスト型の互換性をテストします。"""

    @dataclass
    class Message:
        content: str

    # 複雑なネスト構造。
    source = list[dict[str, Message]]
    target = list[dict[Union[str, bytes], Union[str, Message]]]
    assert is_type_compatible(source, target)

    # 互換性のないネスト構造。
    incompatible_target = list[dict[Union[str, bytes], int]]
    assert not is_type_compatible(source, incompatible_target)
