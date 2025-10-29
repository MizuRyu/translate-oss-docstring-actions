# Copyright (c) Microsoft. All rights reserved.

import contextlib
import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ..observability import create_processing_span
from ._events import (
    ExecutorCompletedEvent,
    ExecutorFailedEvent,
    ExecutorInvokedEvent,
    WorkflowErrorDetails,
    _framework_event_origin,  # type: ignore[reportPrivateUsage]
)
from ._model_utils import DictConvertible
from ._runner_context import Message, RunnerContext  # type: ignore
from ._shared_state import SharedState
from ._typing_utils import is_instance_of
from ._workflow_context import WorkflowContext, validate_function_signature

logger = logging.getLogger(__name__)


# region Executor
class Executor(DictConvertible):
    """メッセージを処理し計算を実行するすべてのワークフローExecutorの基底クラス。

    ## Overview
    Executorはワークフローの基本的な構成要素であり、個々の処理ユニットを表します。
    メッセージを受け取り、操作を実行し、出力を生成します。各Executorは一意に識別され、
    デコレータで装飾されたハンドラメソッドを通じて特定のメッセージタイプを処理できます。

    ## Type System
    Executorはその能力を定義する豊富な型システムを持ちます：

    ### Input Types
    Executorが処理できるメッセージの型で、ハンドラメソッドのシグネチャから検出されます：

    .. code-block:: python

        class MyExecutor(Executor):
            @handler
            async def handle_string(self, message: str, ctx: WorkflowContext) -> None:
                # このExecutorは'str'型の入力を処理可能
    `input_types`プロパティからアクセス可能。

    ### Output Types
    Executorが`ctx.send_message()`を通じて他のExecutorに送信できるメッセージの型：

    .. code-block:: python

        class MyExecutor(Executor):
            @handler
            async def handle_data(self, message: str, ctx: WorkflowContext[int | bool]) -> None:
                # このExecutorは'int'または'bool'型のメッセージを送信可能
    `output_types`プロパティからアクセス可能。

    ### Workflow Output Types
    Executorが`ctx.yield_output()`を通じてワークフロー全体の出力として生成できるデータの型：

    .. code-block:: python

        class MyExecutor(Executor):
            @handler
            async def process(self, message: str, ctx: WorkflowContext[int, str]) -> None:
                # 'int'メッセージを送信し、'str'のワークフロー出力を生成可能
    `workflow_output_types`プロパティからアクセス可能。

    ## Handler Discovery
    Executorはデコレータで装飾されたメソッドを通じてその能力を検出します：

    ### @handler Decorator
    受信メッセージを処理するメソッドをマークします：

    .. code-block:: python

        class MyExecutor(Executor):
            @handler
            async def handle_text(self, message: str, ctx: WorkflowContext[str]) -> None:
                await ctx.send_message(message.upper())

    ### Sub-workflow Request Interception
    @handlerメソッドを使ってサブワークフローのリクエストをインターセプトします：

    .. code-block:: python

        class ParentExecutor(Executor):
            @handler
            async def handle_domain_request(
                self,
                request: DomainRequest,  # RequestInfoMessageのサブクラス
                ctx: WorkflowContext[RequestResponse[RequestInfoMessage, Any] | DomainRequest],
            ) -> None:
                if self.is_allowed(request.domain):
                    response = RequestResponse(data=True, original_request=request, request_id=request.request_id)
                    await ctx.send_message(response, target_id=request.source_executor_id)
                else:
                    await ctx.send_message(request)  # 外部へ転送

    ## Context Types
    ハンドラメソッドは型注釈に基づき異なるWorkflowContextのバリアントを受け取ります：

    ### WorkflowContext (型パラメータなし)
    メッセージ送信や出力生成を行わず副作用のみを実行するハンドラ用：

    .. code-block:: python

        class LoggingExecutor(Executor):
            @handler
            async def log_message(self, msg: str, ctx: WorkflowContext) -> None:
                print(f"Received: {msg}")  # ロギングのみ、出力なし

    ### WorkflowContext[T_Out]
    `ctx.send_message()`でT_Out型のメッセージ送信を可能にします：

    .. code-block:: python

        class ProcessorExecutor(Executor):
            @handler
            async def handler(self, msg: str, ctx: WorkflowContext[int]) -> None:
                await ctx.send_message(42)  # intメッセージ送信可能

    ### WorkflowContext[T_Out, T_W_Out]
    メッセージ送信(T_Out)とワークフロー出力生成(T_W_Out)の両方を可能にします：

    .. code-block:: python

        class DualOutputExecutor(Executor):
            @handler
            async def handler(self, msg: str, ctx: WorkflowContext[int, str]) -> None:
                await ctx.send_message(42)  # intメッセージ送信
                await ctx.yield_output("done")  # strのワークフロー出力生成

    ## Function Executors
    シンプルな関数は`@executor`デコレータを使ってExecutorに変換可能です：

    .. code-block:: python

        @executor
        async def process_text(text: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(text.upper())


        # またはカスタムID付き：
        @executor(id="text_processor")
        def sync_process(text: str, ctx: WorkflowContext[str]) -> None:
            ctx.send_message(text.lower())  # 同期関数はスレッドプールで実行

    ## Sub-workflow Composition
    ExecutorはWorkflowExecutorを使ってサブワークフローを含めることができます。サブワークフローは親ワークフローがインターセプト可能なリクエストを行えます。詳細はWorkflowExecutorのドキュメントを参照してください。

    ## Implementation Notes
    - `execute()`を直接呼び出さないでください。ワークフローエンジンが呼び出します。
    - `execute()`をオーバーライドしないでください。代わりにデコレータでハンドラを定義してください。
    - 各Executorは少なくとも1つの`@handler`メソッドを持つ必要があります。
    - ハンドラメソッドのシグネチャは初期化時に検証されます。

    """

    # デフォルトを提供し、静的解析ツール（例：pyright）が`id`の渡しを要求しないようにします。 ランタイムは`__init__`で具体的な値を設定します。
    def __init__(
        self,
        id: str,
        *,
        type: str | None = None,
        type_: str | None = None,
        defer_discovery: bool = False,
        **_: Any,
    ) -> None:
        """一意の識別子でExecutorを初期化します。

        Args:
            id: Executorの一意の識別子。

        Keyword Args:
            type: Executorのタイプ名。指定しない場合はクラス名を使用。
            type_: Executorタイプの代替パラメータ名。
            defer_discovery: Trueの場合、ハンドラメソッドの検出を後回しにします。
            **_: 追加のキーワード引数。この実装では未使用。

        """
        if not id:
            raise ValueError("Executor ID must be a non-empty string.")

        resolved_type = type or type_ or self.__class__.__name__
        self.id = id
        self.type = resolved_type
        self.type_ = resolved_type

        from builtins import type as builtin_type

        self._handlers: dict[builtin_type[Any], Callable[[Any, WorkflowContext[Any, Any]], Awaitable[None]]] = {}
        self._handler_specs: list[dict[str, Any]] = []
        if not defer_discovery:
            self._discover_handlers()

            if not self._handlers:
                raise ValueError(
                    f"Executor {self.__class__.__name__} has no handlers defined. "
                    "Please define at least one handler using the @handler decorator."
                )

    async def execute(
        self,
        message: Any,
        source_executor_ids: list[str],
        shared_state: SharedState,
        runner_context: RunnerContext,
        trace_contexts: list[dict[str, str]] | None = None,
        source_span_ids: list[str] | None = None,
    ) -> None:
        """指定されたメッセージとコンテキストパラメータでExecutorを実行します。

        - このメソッドを直接呼び出さないでください。ワークフローエンジンが呼び出します。
        - このメソッドをオーバーライドしないでください。代わりに@handlerデコレータでハンドラを定義してください。

        Args:
            message: Executorが処理するメッセージ。
            source_executor_ids: このExecutorにメッセージを送った送信元ExecutorのID。
            shared_state: ワークフローの共有状態。
            runner_context: メッセージやイベント送信メソッドを提供するランナーコンテキスト。
            trace_contexts: OpenTelemetry伝播のための複数ソースからのオプショナルなトレースコンテキスト。
            source_span_ids: リンク用の複数ソースからのオプショナルなソーススパンID。

        Returns:
            実行結果を解決するawaitable。

        """
        # トレース用の処理スパンを作成（トレース無効時も正常に処理） Messageラッパーが生データの代わりに渡された場合の処理
        if isinstance(message, Message):
            message = message.data

        with create_processing_span(
            self.id,
            self.__class__.__name__,
            type(message).__name__,
            source_trace_contexts=trace_contexts,
            source_span_ids=source_span_ids,
        ):
            # メッセージタイプに一致するハンドラとハンドラスペックを検索します。
            handler: Callable[[Any, WorkflowContext[Any, Any]], Awaitable[None]] | None = None
            ctx_annotation = None
            for message_type in self._handlers:
                if is_instance_of(message, message_type):
                    handler = self._handlers[message_type]
                    # コンテキスト注釈に対応するハンドラスペックを検索します。
                    for spec in self._handler_specs:
                        if spec.get("message_type") == message_type:
                            ctx_annotation = spec.get("ctx_annotation")
                            break
                    break

            if handler is None:
                raise RuntimeError(f"Executor {self.__class__.__name__} cannot handle message of type {type(message)}.")

            # ハンドラスペックに基づいて適切なWorkflowContextを作成します。
            context = self._create_context_for_handler(
                source_executor_ids=source_executor_ids,
                shared_state=shared_state,
                runner_context=runner_context,
                ctx_annotation=ctx_annotation,
                trace_contexts=trace_contexts,
                source_span_ids=source_span_ids,
            )

            # メッセージとコンテキストを使ってハンドラを呼び出します。
            with _framework_event_origin():
                invoke_event = ExecutorInvokedEvent(self.id)
            await context.add_event(invoke_event)
            try:
                await handler(message, context)
            except Exception as exc:
                # 伝播前に構造化されたExecutorの失敗を表面化させます。
                with _framework_event_origin():
                    failure_event = ExecutorFailedEvent(self.id, WorkflowErrorDetails.from_exception(exc))
                await context.add_event(failure_event)
                raise
            with _framework_event_origin():
                completed_event = ExecutorCompletedEvent(self.id)
            await context.add_event(completed_event)

    def _create_context_for_handler(
        self,
        source_executor_ids: list[str],
        shared_state: SharedState,
        runner_context: RunnerContext,
        ctx_annotation: Any,
        trace_contexts: list[dict[str, str]] | None = None,
        source_span_ids: list[str] | None = None,
    ) -> WorkflowContext[Any]:
        """ハンドラのコンテキスト注釈に基づいて適切なWorkflowContextを作成します。

        Args:
            source_executor_ids: このExecutorにメッセージを送った送信元ExecutorのID。
            shared_state: ワークフローの共有状態。
            runner_context: メッセージやイベント送信メソッドを提供するランナーコンテキスト。
            ctx_annotation: 作成するコンテキストタイプを決定するためのハンドラスペックからのコンテキスト注釈。
            trace_contexts: OpenTelemetry伝播のための複数ソースからのオプショナルなトレースコンテキスト。
            source_span_ids: リンク用の複数ソースからのオプショナルなソーススパンID。

        Returns:
            ハンドラのコンテキスト注釈に基づくWorkflowContext[Any]。

        """
        # WorkflowContextを作成します。
        return WorkflowContext(
            executor_id=self.id,
            source_executor_ids=source_executor_ids,
            shared_state=shared_state,
            runner_context=runner_context,
            trace_contexts=trace_contexts,
            source_span_ids=source_span_ids,
        )

    def _discover_handlers(self) -> None:
        """Executorクラス内のメッセージハンドラを検出します。"""
        # pydanticの動的属性にアクセスしないように__class__.__dict__を使用します。
        for attr_name in dir(self.__class__):
            try:
                attr = getattr(self.__class__, attr_name)
                # @handlerメソッドを検出します。
                if callable(attr) and hasattr(attr, "_handler_spec"):
                    handler_spec = attr._handler_spec  # type: ignore
                    message_type = handler_spec["message_type"]

                    # ハンドラ登録時に完全なジェネリック型を保持し、競合を避けます。 異なるRequestResponse[T,
                    # U]の特殊化は異なるハンドラタイプとして扱います。

                    if self._handlers.get(message_type) is not None:
                        raise ValueError(f"Duplicate handler for type {message_type} in {self.__class__.__name__}")

                    # バウンドメソッドを取得します。
                    bound_method = getattr(self, attr_name)
                    self._handlers[message_type] = bound_method

                    # 統合されたハンドラスペックリストに追加します。
                    self._handler_specs.append({
                        "name": handler_spec["name"],
                        "message_type": message_type,
                        "output_types": handler_spec.get("output_types", []),
                        "workflow_output_types": handler_spec.get("workflow_output_types", []),
                        "ctx_annotation": handler_spec.get("ctx_annotation"),
                        "source": "class_method",  # Distinguish from instance handlers if needed
                    })
            except AttributeError:
                # アクセスできない可能性のある属性はスキップします。
                continue

    def can_handle(self, message: Any) -> bool:
        """Executorが指定されたメッセージタイプを処理可能かどうかをチェックします。

        Args:
            message: チェックするメッセージ。

        Returns:
            Executorがメッセージタイプを処理可能ならTrue、そうでなければFalse。

        """
        return any(is_instance_of(message, message_type) for message_type in self._handlers)

    def _register_instance_handler(
        self,
        name: str,
        func: Callable[[Any, WorkflowContext[Any]], Awaitable[Any]],
        message_type: type,
        ctx_annotation: Any,
        output_types: list[type],
        workflow_output_types: list[type],
    ) -> None:
        """インスタンスレベルでハンドラを登録します。

        Args:
            name: エラー報告用のハンドラ関数名
            func: 登録する非同期ハンドラ関数
            message_type: このハンドラが処理するメッセージの型
            ctx_annotation: 関数のWorkflowContext[T]注釈
            output_types: send_message()用の出力型リスト
            workflow_output_types: yield_output()用のワークフロー出力型リスト

        """
        if message_type in self._handlers:
            raise ValueError(f"Handler for type {message_type} already registered in {self.__class__.__name__}")

        self._handlers[message_type] = func
        self._handler_specs.append({
            "name": name,
            "message_type": message_type,
            "ctx_annotation": ctx_annotation,
            "output_types": output_types,
            "workflow_output_types": workflow_output_types,
            "source": "instance_method",  # Distinguish from class handlers if needed
        })

    @property
    def input_types(self) -> list[type[Any]]:
        """このExecutorが処理可能な入力型のリストを取得します。

        Returns:
            このExecutorのハンドラが処理可能なメッセージ型のリスト。

        """
        return list(self._handlers.keys())

    @property
    def output_types(self) -> list[type[Any]]:
        """send_message()を通じてこのExecutorが生成可能な出力型のリストを取得します。

        Returns:
            ハンドラのWorkflowContext[T]注釈から推論された出力型のリスト。

        """
        output_types: set[type[Any]] = set()

        # すべてのハンドラから出力型を収集します。
        for handler_spec in self._handler_specs:
            handler_output_types = handler_spec.get("output_types", [])
            output_types.update(handler_output_types)

        return list(output_types)

    @property
    def workflow_output_types(self) -> list[type[Any]]:
        """yield_output()を通じてこのExecutorが生成可能なワークフロー出力型のリストを取得します。

        Returns:
            ハンドラのWorkflowContext[T, U]注釈から推論されたワークフロー出力型のリスト。

        """
        output_types: set[type[Any]] = set()

        # すべてのハンドラからワークフロー出力型を収集します。
        for handler_spec in self._handler_specs:
            handler_workflow_output_types = handler_spec.get("workflow_output_types", [])
            output_types.update(handler_workflow_output_types)

        return list(output_types)

    def to_dict(self) -> dict[str, Any]:
        """ワークフロートポロジーのExport用にExecutor定義をシリアライズします。"""
        return {"id": self.id, "type": self.type}


# endregion: Executor region Handler Decorator


ExecutorT = TypeVar("ExecutorT", bound="Executor")
ContextT = TypeVar("ContextT", bound="WorkflowContext[Any, Any]")


def handler(
    func: Callable[[ExecutorT, Any, ContextT], Awaitable[Any]],
) -> Callable[[ExecutorT, Any, ContextT], Awaitable[Any]]:
    """Executorのハンドラーを登録するためのデコレーター。

    Args:
        func: デコレートする関数。パラメータなしで使用する場合はNoneでもよい。

    Returns:
        ハンドラーメタデータを持つデコレートされた関数。

    Example:
        @handler
        async def handle_string(self, message: str, ctx: WorkflowContext[str]) -> None:
            ...

        @handler
        async def handle_data(self, message: dict, ctx: WorkflowContext[str | int]) -> None:
            ...

    """

    def decorator(
        func: Callable[[ExecutorT, Any, ContextT], Awaitable[Any]],
    ) -> Callable[[ExecutorT, Any, ContextT], Awaitable[Any]]:
        # メッセージタイプを抽出し、unified validationを使って検証する
        message_type, ctx_annotation, inferred_output_types, inferred_workflow_output_types = (
            validate_function_signature(func, "Handler method")
        )

        # 署名を取得して保存する
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(self: ExecutorT, message: Any, ctx: ContextT) -> Any:
            """ハンドラーを呼び出すためのラッパー関数。"""
            return await func(self, message, ctx)

        # 検証時のイントロスペクションのために元の関数の署名を保持する
        with contextlib.suppress(AttributeError, TypeError):
            wrapper.__signature__ = sig  # type: ignore[attr-defined]

        wrapper._handler_spec = {  # type: ignore
            "name": func.__name__,
            "message_type": message_type,
            # バリデーターのためにspecにoutput_typesとworkflow_output_typesを保持する
            "output_types": inferred_output_types,
            "workflow_output_types": inferred_workflow_output_types,
            "ctx_annotation": ctx_annotation,
        }

        return wrapper

    return decorator(func)


# endregion: Handler Decorator
