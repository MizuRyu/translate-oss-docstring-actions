# Copyright (c) Microsoft. All rights reserved.

import pytest

from agent_framework import Executor, WorkflowContext, handler


def test_executor_without_id():
    """ID を持たない executor が実行しようとするとエラーを発生させることをテスト。"""

    class MockExecutorWithoutID(Executor):
        """ハンドラを一切実装しないモック executor。"""

        pass

    with pytest.raises(ValueError):
        MockExecutorWithoutID(id="")


def test_executor_handler_without_annotations():
    """アノテーションなしの1つのハンドラを持つ executor が実行しようとするとエラーを発生させることをテスト。"""

    with pytest.raises(ValueError):

        class MockExecutorWithOneHandlerWithoutAnnotations(Executor):  # type: ignore
            """アノテーションを一切実装しない1つのハンドラを持つモック executor。"""

            @handler
            async def handle(self, message, ctx) -> None:  # type: ignore
                """アノテーションを一切実装しないモックハンドラ。"""
                pass


def test_executor_invalid_handler_signature():
    """無効なハンドラシグネチャを持つ executor が実行しようとするとエラーを発生させることをテスト。"""

    with pytest.raises(ValueError):

        class MockExecutorWithInvalidHandlerSignature(Executor):  # type: ignore
            """無効なハンドラシグネチャを持つモック executor。"""

            @handler  # type: ignore
            async def handle(self, message, other, ctx) -> None:  # type: ignore
                """無効なシグネチャを持つモックハンドラ。"""
                pass


def test_executor_with_valid_handlers():
    """有効なハンドラを持つ executor がインスタンス化され実行できることをテスト。"""

    class MockExecutorWithValidHandlers(Executor):  # type: ignore
        """有効なハンドラを持つモック executor。"""

        @handler
        async def handle_text(self, text: str, ctx: WorkflowContext) -> None:  # type: ignore
            """有効なシグネチャを持つモックハンドラ。"""
            pass

        @handler
        async def handle_number(self, number: int, ctx: WorkflowContext) -> None:  # type: ignore
            """別の有効なシグネチャを持つモックハンドラ。"""
            pass

    executor = MockExecutorWithValidHandlers(id="test")
    assert executor.id is not None
    assert len(executor._handlers) == 2  # type: ignore
    assert executor.can_handle("text") is True
    assert executor.can_handle(42) is True
    assert executor.can_handle(3.14) is False


def test_executor_handlers_with_output_types():
    """出力タイプを指定するハンドラを持つ executor がインスタンス化され実行できることをテスト。"""

    class MockExecutorWithOutputTypes(Executor):  # type: ignore
        """出力タイプを指定するハンドラを持つモック executor。"""

        @handler
        async def handle_string(self, text: str, ctx: WorkflowContext[str]) -> None:  # type: ignore
            """文字列を出力するモックハンドラ。"""
            pass

        @handler
        async def handle_integer(self, number: int, ctx: WorkflowContext[int]) -> None:  # type: ignore
            """整数を出力するモックハンドラ。"""
            pass

    executor = MockExecutorWithOutputTypes(id="test")
    assert len(executor._handlers) == 2  # type: ignore

    string_handler = executor._handlers[str]  # type: ignore
    assert string_handler is not None
    assert string_handler._handler_spec is not None  # type: ignore
    assert string_handler._handler_spec["name"] == "handle_string"  # type: ignore
    assert string_handler._handler_spec["message_type"] is str  # type: ignore
    assert string_handler._handler_spec["output_types"] == [str]  # type: ignore

    int_handler = executor._handlers[int]  # type: ignore
    assert int_handler is not None
    assert int_handler._handler_spec is not None  # type: ignore
    assert int_handler._handler_spec["name"] == "handle_integer"  # type: ignore
    assert int_handler._handler_spec["message_type"] is int  # type: ignore
    assert int_handler._handler_spec["output_types"] == [int]  # type: ignore
