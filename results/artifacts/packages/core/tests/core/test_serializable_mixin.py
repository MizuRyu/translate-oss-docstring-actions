# Copyright (c) Microsoft. All rights reserved.

"""SerializationMixin機能のテスト。"""

import logging
from typing import Any

from agent_framework._serialization import SerializationMixin


class TestSerializationMixin:
    """SerializationMixin のシリアライズ、デシリアライズ、および依存性注入をテストします。"""

    def test_basic_serialization(self):
        """基本的な to_dict と from_dict の機能をテストします。"""

        class TestClass(SerializationMixin):
            def __init__(self, value: str, number: int):
                self.value = value
                self.number = number

        obj = TestClass(value="test", number=42)
        data = obj.to_dict()

        assert data["type"] == "test_class"
        assert data["value"] == "test"
        assert data["number"] == 42

        restored = TestClass.from_dict(data)
        assert restored.value == "test"
        assert restored.number == 42

    def test_injectable_dependency_no_warning(self, caplog):
        """注入可能な依存関係がデバッグログをトリガーしないことをテストします。"""

        class TestClass(SerializationMixin):
            INJECTABLE = {"client"}

            def __init__(self, value: str, client: Any = None):
                self.value = value
                self.client = client

        mock_client = "mock_client_instance"

        with caplog.at_level(logging.DEBUG):
            obj = TestClass.from_dict(
                {"type": "test_class", "value": "test"},
                dependencies={"test_class": {"client": mock_client}},
            )

        assert obj.value == "test"
        assert obj.client == mock_client
        # 注入可能な依存関係についてはデバッグメッセージがログに記録されるべきではありません。
        assert not any("is not in INJECTABLE set" in record.message for record in caplog.records)

    def test_non_injectable_dependency_logs_debug(self, caplog):
        """注入不可能な依存関係がデバッグログをトリガーすることをテストします。"""

        class TestClass(SerializationMixin):
            INJECTABLE = {"client"}

            def __init__(self, value: str, other: Any = None):
                self.value = value
                self.other = other

        mock_other = "mock_other_instance"

        with caplog.at_level(logging.DEBUG):
            obj = TestClass.from_dict(
                {"type": "test_class", "value": "test"},
                dependencies={"test_class": {"other": mock_other}},
            )

        assert obj.value == "test"
        assert obj.other == mock_other
        # 注入不可能な依存関係についてはデバッグメッセージがログに記録されるべきです。
        debug_messages = [record.message for record in caplog.records if record.levelname == "DEBUG"]
        assert any("is not in INJECTABLE set" in msg for msg in debug_messages)
        assert any("other" in msg for msg in debug_messages)
        assert any("client" in msg for msg in debug_messages)  # 利用可能な注入可能なものを言及すべきです。

    def test_multiple_dependencies_mixed_injectable(self, caplog):
        """注入可能および注入不可能な依存関係の両方を含む場合のテスト。"""

        class TestClass(SerializationMixin):
            INJECTABLE = {"client", "logger"}

            def __init__(
                self,
                value: str,
                client: Any = None,
                logger: Any = None,
                other: Any = None,
            ):
                self.value = value
                self.client = client
                self.logger = logger
                self.other = other

        mock_client = "mock_client"
        mock_logger = "mock_logger"
        mock_other = "mock_other"

        with caplog.at_level(logging.DEBUG):
            obj = TestClass.from_dict(
                {"type": "test_class", "value": "test"},
                dependencies={
                    "test_class": {
                        "client": mock_client,
                        "logger": mock_logger,
                        "other": mock_other,
                    }
                },
            )

        assert obj.value == "test"
        assert obj.client == mock_client
        assert obj.logger == mock_logger
        assert obj.other == mock_other

        # 'other' のみがデバッグログをトリガーするべきです。
        debug_messages = [record.message for record in caplog.records if record.levelname == "DEBUG"]
        assert any("other" in msg and "is not in INJECTABLE set" in msg for msg in debug_messages)
        # 'client' と 'logger' は注入不可能な依存関係として言及されるべきではありません。
        assert not any("Dependency 'client'" in msg and "is not in INJECTABLE set" in msg for msg in debug_messages)
        assert not any("Dependency 'logger'" in msg and "is not in INJECTABLE set" in msg for msg in debug_messages)

    def test_no_injectable_set_defined(self, caplog):
        """INJECTABLE が定義されていない場合（空のセットがデフォルト）の動作をテストします。"""

        class TestClass(SerializationMixin):
            def __init__(self, value: str, client: Any = None):
                self.value = value
                self.client = client

        mock_client = "mock_client"

        with caplog.at_level(logging.DEBUG):
            obj = TestClass.from_dict(
                {"type": "test_class", "value": "test"},
                dependencies={"test_class": {"client": mock_client}},
            )

        assert obj.value == "test"
        assert obj.client == mock_client
        # INJECTABLE がデフォルトで空のため、デバッグメッセージがログに記録されるべきです。
        debug_messages = [record.message for record in caplog.records if record.levelname == "DEBUG"]
        assert any("client" in msg and "is not in INJECTABLE set" in msg for msg in debug_messages)

    def test_default_exclude_serialization(self):
        """DEFAULT_EXCLUDE フィールドが to_dict() に含まれないことをテストします。"""

        class TestClass(SerializationMixin):
            DEFAULT_EXCLUDE = {"secret"}

            def __init__(self, value: str, secret: str):
                self.value = value
                self.secret = secret

        obj = TestClass(value="test", secret="hidden")
        data = obj.to_dict()

        assert "value" in data
        assert "secret" not in data
        assert data["value"] == "test"

    def test_roundtrip_with_injectable_dependency(self):
        """注入可能な依存関係を用いた完全なシリアライズ/デシリアライズの往復をテストします。"""

        class TestClass(SerializationMixin):
            INJECTABLE = {"client"}
            DEFAULT_EXCLUDE = {"client"}

            def __init__(self, value: str, number: int, client: Any = None):
                self.value = value
                self.number = number
                self.client = client

        mock_client = "mock_client"
        obj = TestClass(value="test", number=42, client=mock_client)

        # シリアライズ
        data = obj.to_dict()
        assert data["value"] == "test"
        assert data["number"] == 42
        assert "client" not in data  # シリアライズから除外

        # 依存性注入を用いたデシリアライズ
        restored = TestClass.from_dict(data, dependencies={"test_class": {"client": mock_client}})
        assert restored.value == "test"
        assert restored.number == 42
        assert restored.client == mock_client
