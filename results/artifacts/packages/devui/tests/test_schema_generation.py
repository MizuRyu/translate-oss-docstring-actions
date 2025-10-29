# Copyright (c) Microsoft. All rights reserved.
"""異なる入力タイプに対するスキーマ生成をテストします。"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# 親パッケージをパスに追加します。
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework_devui._utils import generate_input_schema


@dataclass
class InputData:
    text: str
    source: str


@dataclass
class Address:
    street: str
    city: str
    zipcode: str


@dataclass
class PersonData:
    name: str
    age: int
    address: Address


def test_builtin_types_schema_generation():
    """組み込みタイプに対するスキーマ生成をテストします。"""
    # strスキーマをテストします。
    str_schema = generate_input_schema(str)
    assert str_schema is not None
    assert isinstance(str_schema, dict)

    # dictスキーマをテストします。
    dict_schema = generate_input_schema(dict)
    assert dict_schema is not None
    assert isinstance(dict_schema, dict)

    # intスキーマをテストします。
    int_schema = generate_input_schema(int)
    assert int_schema is not None
    assert isinstance(int_schema, dict)


def test_dataclass_schema_generation():
    """dataclassに対するスキーマ生成をテストします。"""
    schema = generate_input_schema(InputData)

    assert schema is not None
    assert isinstance(schema, dict)

    # 基本的なスキーマ構造のチェック。
    if "properties" in schema:
        properties = schema["properties"]
        assert "text" in properties
        assert "source" in properties


def test_chat_message_schema_generation():
    """ChatMessage（SerializationMixin）に対するスキーマ生成をテストします。"""
    try:
        from agent_framework import ChatMessage

        schema = generate_input_schema(ChatMessage)
        assert schema is not None
        assert isinstance(schema, dict)

    except ImportError:
        pytest.skip("ChatMessage not available - agent_framework not installed")


def test_pydantic_model_schema_generation():
    """Pydanticモデルに対するスキーマ生成をテストします。"""
    try:
        from pydantic import BaseModel, Field

        class UserInput(BaseModel):
            name: str = Field(description="User's name")
            age: int = Field(description="User's age")
            email: str | None = Field(default=None, description="Optional email")

        schema = generate_input_schema(UserInput)
        assert schema is not None
        assert isinstance(schema, dict)

        # プロパティの存在をチェックします。
        if "properties" in schema:
            properties = schema["properties"]
            assert "name" in properties
            assert "age" in properties
            assert "email" in properties

    except ImportError:
        pytest.skip("Pydantic not available")


def test_nested_dataclass_schema_generation():
    """ネストしたdataclassに対するスキーマ生成をテストします。"""
    schema = generate_input_schema(PersonData)

    assert schema is not None
    assert isinstance(schema, dict)

    # 基本的なスキーマ構造のチェック。
    if "properties" in schema:
        properties = schema["properties"]
        assert "name" in properties
        assert "age" in properties
        assert "address" in properties


def test_schema_generation_error_handling():
    """無効な入力に対するスキーマ生成をテストします。"""
    # 非タイプオブジェクトでテストします - 問題なく処理できるはずです。
    try:
        # 問題を引き起こす可能性のある非タイプオブジェクトを使用します。
        schema = generate_input_schema("not_a_type")  # type: ignore
        # 例外が発生しなければ、結果は有効であるべきです。
        if schema is not None:
            assert isinstance(schema, dict)
    except (TypeError, ValueError, AttributeError):
        # エラーが発生しても許容されます。
        pass


if __name__ == "__main__":
    # 手動実行用のシンプルなテストランナー。
    pytest.main([__file__, "-v"])
