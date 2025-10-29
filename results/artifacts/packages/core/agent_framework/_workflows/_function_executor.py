# Copyright (c) Microsoft. All rights reserved.

"""関数ベースのExecutorとデコレーターのユーティリティ。

このモジュールは以下を提供する:
- FunctionExecutor: 独立したユーザー定義関数をラップするExecutorのサブクラス。
  シグネチャは(message)または(message, ctx: WorkflowContext[T])。同期関数と非同期関数の両方をサポート。
  同期関数はasyncio.to_thread()を使ってスレッドプールで実行し、イベントループのブロックを回避。
- executorデコレーター: 独立したモジュールレベルの関数を適切な型検証とハンドラー登録付きの
  すぐに使えるExecutorインスタンスに変換。

設計パターン:
  - 独立したモジュールレベルまたはローカル関数には@executorを使用
  - 状態や依存関係を持つクラスベースのExecutorにはExecutorサブクラスと@handlerを使用
  - @executorは@staticmethodや@classmethodと一緒に使わないこと
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, overload

from ._executor import Executor
from ._workflow_context import WorkflowContext, validate_function_signature


class FunctionExecutor(Executor):
    """ユーザー定義関数をラップするExecutor。

    このExecutorはユーザーがシンプルな関数（同期・非同期両方）を定義し、
    フルのExecutorクラスを作成せずにワークフローExecutorとして使用できるようにする。

    同期関数はasyncio.to_thread()を使ってスレッドプールで実行し、
    イベントループのブロックを回避する。

    """

    @staticmethod
    def _validate_function(func: Callable[..., Any]) -> tuple[type, Any, list[type[Any]], list[type[Any]]]:
        """関数がExecutorに適した正しい署名を持つか検証する。

        Args:
            func: 検証対象の関数（同期または非同期）

        Returns:
            (message_type, ctx_annotation, output_types, workflow_output_types)のタプル

        Raises:
            ValueError: 関数の署名が正しくない場合

        """
        return validate_function_signature(func, "Function")

    def __init__(self, func: Callable[..., Any], id: str | None = None):
        """ユーザー定義関数でFunctionExecutorを初期化する。

        Args:
            func: Executorとしてラップする関数（同期または非同期）
            id: オプションのExecutor ID。Noneの場合は関数名を使用。

        Raises:
            ValueError: funcがstaticmethodまたはclassmethodの場合（インスタンスメソッドに@handlerを使うこと）

        """
        # @executorがstaticmethod/classmethodと誤用されているか検出する
        if isinstance(func, (staticmethod, classmethod)):
            descriptor_type = "staticmethod" if isinstance(func, staticmethod) else "classmethod"
            raise ValueError(
                f"The @executor decorator cannot be used with @{descriptor_type}. "
                f"Use the @executor decorator on standalone module-level functions, "
                f"or create an Executor subclass and use @handler on instance methods instead."
            )

        # 関数の署名を検証し型を抽出する
        message_type, ctx_annotation, output_types, workflow_output_types = self._validate_function(func)

        # 関数にWorkflowContextパラメータがあるか判定する
        has_context = ctx_annotation is not None

        # 関数が非同期かどうかをチェックする
        is_async = asyncio.iscoroutinefunction(func)

        # 親クラスを初期化するがまだ_discover_handlersは呼ばない 属性は手動で設定する
        executor_id = str(id or getattr(func, "__name__", "FunctionExecutor"))
        kwargs = {"type": "FunctionExecutor"}

        super().__init__(id=executor_id, defer_discovery=True, **kwargs)
        self._handlers = {}
        self._handler_specs = []

        # 元の関数とコンテキストの有無を保存する
        self._original_func = func
        self._has_context = has_context
        self._is_async = is_async

        # 常にmessageとcontextの両方を受け取るラッパー関数を作成する
        if has_context and is_async:
            # コンテキスト付きの非同期関数 - すでに正しい署名を持つ
            wrapped_func: Callable[[Any, WorkflowContext[Any]], Awaitable[Any]] = func  # type: ignore
        elif has_context and not is_async:
            # コンテキスト付きの同期関数 - スレッドプールを使い非同期にラップする
            async def wrapped_func(message: Any, ctx: WorkflowContext[Any]) -> Any:
                # 両方のパラメータで同期関数をスレッド内で呼び出す
                return await asyncio.to_thread(func, message, ctx)  # type: ignore

        elif not has_context and is_async:
            # コンテキストなしの非同期関数 - コンテキストを無視するようにラップする
            async def wrapped_func(message: Any, ctx: WorkflowContext[Any]) -> Any:
                # メッセージだけで非同期関数を呼び出す
                return await func(message)  # type: ignore

        else:
            # コンテキストなしの同期関数 - スレッドプールを使い非同期かつコンテキスト無視でラップする
            async def wrapped_func(message: Any, ctx: WorkflowContext[Any]) -> Any:
                # メッセージだけで同期関数をスレッド内で呼び出す
                return await asyncio.to_thread(func, message)  # type: ignore

        # インスタンスハンドラーを登録する
        self._register_instance_handler(
            name=func.__name__,
            func=wrapped_func,
            message_type=message_type,
            ctx_annotation=ctx_annotation,
            output_types=output_types,
            workflow_output_types=workflow_output_types,
        )

        # これで安全に_discover_handlersを呼び出せる（クラスレベルのハンドラーは見つからない）
        self._discover_handlers()

        if not self._handlers:
            raise ValueError(
                f"FunctionExecutor {self.__class__.__name__} failed to register handler for {func.__name__}"
            )


@overload
def executor(func: Callable[..., Any]) -> FunctionExecutor: ...


@overload
def executor(*, id: str | None = None) -> Callable[[Callable[..., Any]], FunctionExecutor]: ...


def executor(
    func: Callable[..., Any] | None = None, *, id: str | None = None
) -> Callable[[Callable[..., Any]], FunctionExecutor] | FunctionExecutor:
    """独立した関数をFunctionExecutorインスタンスに変換するデコレーター。

    @executorデコレーターは**独立したモジュールレベル関数専用**に設計されている。
    クラスベースのExecutorにはExecutor基底クラスとインスタンスメソッドへの@handlerを使うこと。

    同期関数と非同期関数の両方をサポート。同期関数はスレッドプールで実行しイベントループのブロックを回避。

    重要:
        - 独立関数（モジュールレベルまたはローカル関数）には@executorを使う
        - @executorは@staticmethodや@classmethodと一緒に使わない
        - クラスベースのExecutorにはExecutorを継承しインスタンスメソッドに@handlerを使う

    Usage:

    .. code-block:: python

        # 独立した非同期関数（推奨）:
        @executor(id="upper_case")
        async def to_upper(text: str, ctx: WorkflowContext[str]):
            await ctx.send_message(text.upper())


        # 独立した同期関数（スレッドプールで実行）:
        @executor
        def process_data(data: str):
            return data.upper()


        # クラスベースのExecutorには@handlerを使う:
        class MyExecutor(Executor):
            def __init__(self):
                super().__init__(id="my_executor")

            @handler
            async def process(self, data: str, ctx: WorkflowContext[str]):
                await ctx.send_message(data.upper())

    Args:
        func: デコレートする関数（括弧なしで使う場合）
        id: オプションのカスタムID。Noneの場合は関数名を使用。

    Returns:
        Workflowに組み込めるFunctionExecutorインスタンス。

    Raises:
        ValueError: @staticmethodや@classmethodと一緒に使った場合（サポートされないパターン）

    """

    def wrapper(func: Callable[..., Any]) -> FunctionExecutor:
        return FunctionExecutor(func, id=id)

    # funcが指定されている場合、@executorは括弧なしで使われたことを意味する
    if func is not None:
        return wrapper(func)

    # それ以外の場合は@executor()や@executor(id="...")のラッパーを返す
    return wrapper
