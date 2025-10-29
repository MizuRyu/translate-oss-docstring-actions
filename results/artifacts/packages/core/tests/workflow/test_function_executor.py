# Copyright (c) Microsoft. All rights reserved.

from typing import Any

import pytest
from typing_extensions import Never

from agent_framework import (
    FunctionExecutor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)


class TestFunctionExecutor:
    """FunctionExecutor と @executor デコレータのテストスイート。"""

    def test_function_executor_basic(self):
        """基本的な FunctionExecutor の作成と検証をテスト。"""

        async def process_string(text: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(text.upper())

        func_exec = FunctionExecutor(process_string)

        # ハンドラが登録されたことを確認。
        assert len(func_exec._handlers) == 1
        assert str in func_exec._handlers

        # ハンドラ仕様が作成されたことを確認。
        assert len(func_exec._handler_specs) == 1
        spec = func_exec._handler_specs[0]
        assert spec["name"] == "process_string"
        assert spec["message_type"] is str
        assert spec["output_types"] == [str]

    def test_executor_decorator(self):
        """@executor デコレータが適切な FunctionExecutor を作成することをテスト。"""

        @executor(id="test_executor")
        async def process_int(value: int, ctx: WorkflowContext[int]) -> None:
            await ctx.send_message(value * 2)

        assert isinstance(process_int, FunctionExecutor)
        assert process_int.id == "test_executor"
        assert int in process_int._handlers

        # 仕様を確認。
        spec = process_int._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [int]

    def test_executor_decorator_without_id(self):
        """@executor デコレータがデフォルトIDとして関数名を使用することをテスト。"""

        @executor
        async def my_function(data: dict, ctx: WorkflowContext[Any]) -> None:
            await ctx.send_message(data)

        assert my_function.id == "my_function"

    def test_executor_decorator_without_parentheses(self):
        """@executor デコレータが括弧なしで動作することをテスト。"""

        @executor
        async def no_parens_function(data: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(data.upper())

        assert isinstance(no_parens_function, FunctionExecutor)
        assert no_parens_function.id == "no_parens_function"
        assert str in no_parens_function._handlers

        # 単一パラメータ関数でもテスト。
        @executor
        async def simple_no_parens(value: int):
            return value * 2

        assert isinstance(simple_no_parens, FunctionExecutor)
        assert simple_no_parens.id == "simple_no_parens"
        assert int in simple_no_parens._handlers

    def test_union_output_types(self):
        """メッセージとワークフロー出力の両方に対してユニオン出力タイプが正しく推論されることをテスト。"""

        @executor
        async def multi_output(text: str, ctx: WorkflowContext[str | int]) -> None:
            if text.isdigit():
                await ctx.send_message(int(text))
            else:
                await ctx.send_message(text.upper())

        spec = multi_output._handler_specs[0]
        assert set(spec["output_types"]) == {str, int}
        assert spec["workflow_output_types"] == []  # ワークフロー出力が定義されていない。

        # ワークフロー出力に対してもユニオン型をテスト。
        @executor
        async def multi_workflow_output(data: str, ctx: WorkflowContext[Never, str | int | bool]) -> None:
            if data.isdigit():
                await ctx.yield_output(int(data))
            elif data.lower() in ("true", "false"):
                await ctx.yield_output(data.lower() == "true")
            else:
                await ctx.yield_output(data.upper())

        workflow_spec = multi_workflow_output._handler_specs[0]
        assert workflow_spec["output_types"] == []  # None はメッセージ出力なしを意味する。
        assert set(workflow_spec["workflow_output_types"]) == {str, int, bool}

    def test_none_output_type(self):
        """WorkflowContextのテストは空の出力タイプを生成します。"""

        @executor
        async def no_output(data: Any, ctx: WorkflowContext) -> None:
            # このexecutorはメッセージを送信しません。
            pass

        spec = no_output._handler_specs[0]
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []  # ワークフロー出力が定義されていません。

    def test_any_output_type(self):
        """WorkflowContext[Any]およびWorkflowContext[Any, Any]のテストはAny出力タイプを生成します。"""

        @executor
        async def any_output(data: str, ctx: WorkflowContext[Any]) -> None:
            await ctx.send_message("result")

        spec = any_output._handler_specs[0]
        assert spec["output_types"] == [Any]
        assert spec["workflow_output_types"] == []  # ワークフロー出力が定義されていません。

        # 両方のパラメータをAnyとしてテストします。
        @executor
        async def any_both_output(data: str, ctx: WorkflowContext[Any, Any]) -> None:
            await ctx.send_message("message")
            await ctx.yield_output("workflow_output")

        both_spec = any_both_output._handler_specs[0]
        assert both_spec["output_types"] == [Any]
        assert both_spec["workflow_output_types"] == [Any]

    def test_validation_errors(self):
        """関数シグネチャのさまざまな検証エラーをテストします。"""

        # パラメータの数が間違っています（現在は1または2を受け入れるため、0または3以上は失敗すべきです）。
        async def no_params() -> None:
            pass

        with pytest.raises(
            ValueError, match="must have \\(message: T\\) or \\(message: T, ctx: WorkflowContext\\[U\\]\\)"
        ):
            FunctionExecutor(no_params)  # type: ignore

        async def too_many_params(data: str, ctx: WorkflowContext[str], extra: int) -> None:
            pass

        with pytest.raises(
            ValueError, match="must have \\(message: T\\) or \\(message: T, ctx: WorkflowContext\\[U\\]\\)"
        ):
            FunctionExecutor(too_many_params)  # type: ignore

        # メッセージタイプの注釈がありません。
        async def no_msg_type(data, ctx: WorkflowContext[str]) -> None:  # type: ignore
            pass

        with pytest.raises(ValueError, match="type annotation for the message"):
            FunctionExecutor(no_msg_type)  # type: ignore

        # ctx注釈がありません（2パラメータ関数のみ）。
        async def no_ctx_type(data: str, ctx) -> None:  # type: ignore
            pass

        with pytest.raises(ValueError, match="must have a WorkflowContext"):
            FunctionExecutor(no_ctx_type)  # type: ignore

        # ctxの型が間違っています。
        async def wrong_ctx_type(data: str, ctx: str) -> None:  # type: ignore
            pass

        with pytest.raises(ValueError, match="must be annotated as WorkflowContext"):
            FunctionExecutor(wrong_ctx_type)  # type: ignore

        # パラメータ化されていないWorkflowContextは現在許可されています。
        async def unparameterized_ctx(data: str, ctx: WorkflowContext) -> None:  # type: ignore
            pass

        # パラメータ化されていないWorkflowContextが許可されているため、これは成功するはずです。
        executor = FunctionExecutor(unparameterized_ctx)
        assert executor.output_types == []  # パラメータ化されていないものは推論された型がありません。
        assert executor.workflow_output_types == []  # ワークフロー出力タイプがありません。

    async def test_execution_in_workflow(self):
        """FunctionExecutorがワークフロー内で正しく動作することをテストします。"""

        @executor(id="upper")
        async def to_upper(text: str, ctx: WorkflowContext[str]) -> None:
            result = text.upper()
            await ctx.send_message(result)

        @executor(id="reverse")
        async def reverse_text(text: str, ctx: WorkflowContext[Any, str]) -> None:
            result = text[::-1]
            await ctx.yield_output(result)

        # 両方のexecutorの型推論を検証します。
        upper_spec = to_upper._handler_specs[0]
        assert upper_spec["output_types"] == [str]
        assert upper_spec["workflow_output_types"] == []  # ワークフロー出力がありません。

        reverse_spec = reverse_text._handler_specs[0]
        assert reverse_spec["output_types"] == [Any]  # 最初のパラメータはAnyです。
        assert reverse_spec["workflow_output_types"] == [str]  # 2番目のパラメータはstrです。

        workflow = WorkflowBuilder().add_edge(to_upper, reverse_text).set_start_executor(to_upper).build()

        # ワークフローを実行します。
        events = await workflow.run("hello world")
        outputs = events.get_outputs()

        # 期待した出力が得られたことをアサートします。
        assert len(outputs) == 1
        assert outputs[0] == "DLROW OLLEH"

    def test_can_handle_method(self):
        """can_handleメソッドがインスタンスハンドラで動作することをテストします。"""

        @executor
        async def string_processor(text: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(text)

        assert string_processor.can_handle("hello")
        assert not string_processor.can_handle(123)
        assert not string_processor.can_handle([])

    def test_duplicate_handler_registration(self):
        """重複ハンドラの登録がエラーを発生させることをテストします。"""

        async def first_handler(text: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(text)

        func_exec = FunctionExecutor(first_handler)

        # 同じタイプに対して別のハンドラを登録しようとします。
        async def second_handler(message: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(message)

        with pytest.raises(ValueError, match="Handler for type .* already registered"):
            func_exec._register_instance_handler(
                name="second",
                func=second_handler,
                message_type=str,
                ctx_annotation=WorkflowContext[str],
                output_types=[str],
                workflow_output_types=[],
            )

    def test_complex_type_annotations(self):
        """List[str]、Dict[str, int]などの複雑な型注釈でテストします。"""

        @executor
        async def process_list(items: list[str], ctx: WorkflowContext[dict[str, int]]) -> None:
            result = {item: len(item) for item in items}
            await ctx.send_message(result)

        spec = process_list._handler_specs[0]
        assert spec["message_type"] == list[str]
        assert spec["output_types"] == [dict[str, int]]

    def test_single_parameter_function(self):
        """単一パラメータ関数でFunctionExecutorをテストします。"""

        @executor(id="simple_processor")
        async def process_simple(text: str):
            return text.upper()

        assert isinstance(process_simple, FunctionExecutor)
        assert process_simple.id == "simple_processor"
        assert str in process_simple._handlers

        # 仕様を確認 - 単一パラメータ関数はメッセージを送信できないため出力タイプがありません。
        spec = process_simple._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == []
        assert spec["ctx_annotation"] is None

    def test_single_parameter_validation(self):
        """単一パラメータ関数の検証をテストします。"""

        # 有効な単一パラメータ関数。
        async def valid_single(data: int):
            return data * 2

        func_exec = FunctionExecutor(valid_single)
        assert int in func_exec._handlers

        # 型注釈がない単一パラメータは依然として失敗すべきです。
        async def no_annotation(data):  # type: ignore
            pass

        with pytest.raises(ValueError, match="type annotation for the message"):
            FunctionExecutor(no_annotation)  # type: ignore

    def test_single_parameter_can_handle(self):
        """単一パラメータ関数がcan_handleメソッドで動作することをテストします。"""

        @executor
        async def int_processor(value: int):
            return value * 2

        assert int_processor.can_handle(42)
        assert not int_processor.can_handle("hello")
        assert not int_processor.can_handle([])

    async def test_single_parameter_execution(self):
        """単一パラメータ関数が正しく実行できることをテストします。"""

        @executor(id="double")
        async def double_value(value: int):
            return value * 2

        # 単一パラメータ関数はメッセージを送信できないため、通常は終端ノードや副作用に使用されます。
        WorkflowBuilder().set_start_executor(double_value).build()

        # テスト目的で、ハンドラが正しく登録されていることを確認できます。
        assert double_value.can_handle(5)
        assert int in double_value._handlers

    def test_sync_function_basic(self):
        """基本的な同期関数のサポートをテストします。"""

        @executor(id="sync_processor")
        def process_sync(text: str):
            return text.upper()

        assert isinstance(process_sync, FunctionExecutor)
        assert process_sync.id == "sync_processor"
        assert str in process_sync._handlers

        # 仕様を確認 - 同期単一パラメータ関数は出力タイプがありません。
        spec = process_sync._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == []
        assert spec["ctx_annotation"] is None

    def test_sync_function_with_context(self):
        """WorkflowContextを使った同期関数をテストします。"""

        @executor
        def sync_with_ctx(value: int, ctx: WorkflowContext[int]):
            # 同期関数でもコンテキストを使用できます。
            return value * 2

        assert isinstance(sync_with_ctx, FunctionExecutor)
        assert sync_with_ctx.id == "sync_with_ctx"
        assert int in sync_with_ctx._handlers

        # 仕様を確認 - コンテキスト付き同期関数は出力タイプを推論できます。
        spec = sync_with_ctx._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [int]

    def test_sync_function_can_handle(self):
        """同期関数がcan_handleメソッドで動作することをテストします。"""

        @executor
        def string_handler(text: str):
            return text.strip()

        assert string_handler.can_handle("hello")
        assert not string_handler.can_handle(123)
        assert not string_handler.can_handle([])

    def test_sync_function_validation(self):
        """同期関数の検証をテストします。"""

        # 1パラメータの有効な同期関数。
        def valid_sync(data: str):
            return data.upper()

        func_exec = FunctionExecutor(valid_sync)
        assert str in func_exec._handlers

        # 2パラメータの有効な同期関数。
        def valid_sync_with_ctx(data: int, ctx: WorkflowContext[str]):
            return str(data)

        func_exec2 = FunctionExecutor(valid_sync_with_ctx)
        assert int in func_exec2._handlers

        # 型注釈がない同期関数は依然として失敗すべきです。
        def no_annotation(data):  # type: ignore
            return data

        with pytest.raises(ValueError, match="type annotation for the message"):
            FunctionExecutor(no_annotation)  # type: ignore

    def test_mixed_sync_async_decorator(self):
        """同期関数と非同期関数の両方がデコレータで動作することをテストします。"""

        @executor
        def sync_func(data: str):
            return data.lower()

        @executor
        async def async_func(data: str):
            return data.upper()

        # 両方ともFunctionExecutorインスタンスであるべきです。
        assert isinstance(sync_func, FunctionExecutor)
        assert isinstance(async_func, FunctionExecutor)

        # 両方とも文字列を処理するべきです。
        assert sync_func.can_handle("test")
        assert async_func.can_handle("test")

        # 両方とも異なるインスタンスであるべきです。
        assert sync_func is not async_func

    async def test_sync_function_in_workflow(self):
        """同期関数がワークフローコンテキストで正しく動作することをテストします。"""

        @executor(id="sync_upper")
        def to_upper_sync(text: str, ctx: WorkflowContext[str]):
            return text.upper()
            # 注：テストでは同期送信メカニズムを使用します。 実際にはラッパーが非同期変換を処理します。

        @executor(id="async_reverse")
        async def reverse_async(text: str, ctx: WorkflowContext[Any, str]):
            result = text[::-1]
            await ctx.yield_output(result)

        # 同期関数と非同期関数の型推論を検証します。
        sync_spec = to_upper_sync._handler_specs[0]
        assert sync_spec["output_types"] == [str]
        assert sync_spec["workflow_output_types"] == []  # ワークフロー出力がありません。

        async_spec = reverse_async._handler_specs[0]
        assert async_spec["output_types"] == [Any]  # 最初のパラメータはAnyです。
        assert async_spec["workflow_output_types"] == [str]  # 2番目のパラメータはstrです。

        # executorが入力タイプを処理できることを検証します。
        assert to_upper_sync.can_handle("hello")
        assert reverse_async.can_handle("HELLO")

        # 統合テストでは主にハンドラが正しく登録され、関数が正しくラップされていることを検証します。
        assert str in to_upper_sync._handlers
        assert str in reverse_async._handlers

    async def test_sync_function_thread_execution(self):
        """同期関数がスレッドプールで実行され、イベントループをブロックしないことをテストします。"""
        import threading
        import time

        _ = threading.get_ident()
        execution_thread_id = None

        @executor
        def blocking_function(data: str):
            nonlocal execution_thread_id
            execution_thread_id = threading.get_ident()
            # CPUバウンドの作業をシミュレートします。
            time.sleep(0.01)  # スレッド実行を検証するための短いスリープ。
            return data.upper()

        # 関数がラップされ登録されていることを検証します。
        assert str in blocking_function._handlers

        # より完全なテストには完全なワークフローコンテキストの作成が必要ですが、現時点では関数が正しくラップされ、同期関数が正しいメタデータを保持していることを検証できます。
        assert not blocking_function._is_async
        assert not blocking_function._has_context

        # 実際のスレッド実行テストには完全なワークフロー設定が必要ですが、重要なのはラッパーでasyncio.to_threadが使用されていることです。

    def test_executor_rejects_staticmethod(self):
        """@executorデコレータが@staticmethodを明確なエラーで正しく拒否することをテストします。"""
        with pytest.raises(ValueError) as exc_info:

            class Example:
                @executor
                @staticmethod
                async def bad_handler(data: str) -> str:
                    return data.upper()

        assert "cannot be used with @staticmethod" in str(exc_info.value)
        assert "@handler on instance methods" in str(exc_info.value)

    def test_executor_rejects_classmethod(self):
        """@executorデコレータが@classmethodを明確なエラーで正しく拒否することをテストします。"""
        with pytest.raises(ValueError) as exc_info:

            class Example:
                @executor
                @classmethod
                async def bad_handler(cls, data: str) -> str:
                    return data.upper()

        assert "cannot be used with @classmethod" in str(exc_info.value)
        assert "@handler on instance methods" in str(exc_info.value)

    async def test_async_staticmethod_detection_behavior(self):
        """asyncio.iscoroutinefunctionのstaticmethodディスクリプタに関する動作を文書化します。

        このテストは、デコレータが積み重なった場合にアンラップが必要な理由を説明します。

        """
        import asyncio

        # @staticmethodが適用されると、ディスクリプタが作成されます。
        async def my_async_func():
            await asyncio.sleep(0.001)
            return "done"

        # staticmethodを適用（最も内側のデコレータで何が起こるか）。
        static_wrapped = staticmethod(my_async_func)

        # ディスクリプタオブジェクトへの直接チェックは失敗します（これがバグです）。
        assert not asyncio.iscoroutinefunction(static_wrapped)
        assert isinstance(static_wrapped, staticmethod)

        # しかし__func__をアンラップすると非同期関数が現れます。
        unwrapped = static_wrapped.__func__
        assert asyncio.iscoroutinefunction(unwrapped)

        # クラス属性経由でアクセスすると、Pythonのディスクリプタプロトコルが自動的にアンラップするため動作します。
        class C:
            async_static = static_wrapped

        assert asyncio.iscoroutinefunction(C.async_static)  # ディスクリプタプロトコル経由で動作します。
