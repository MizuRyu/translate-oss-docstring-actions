# Copyright (c) Microsoft. All rights reserved.

import inspect
import logging
from collections.abc import Callable
from types import UnionType
from typing import Any, Generic, Union, cast, get_args, get_origin

from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind
from typing_extensions import Never, TypeVar

from ..observability import OtelAttr, create_workflow_span
from ._const import EXECUTOR_STATE_KEY
from ._events import (
    WorkflowEvent,
    WorkflowEventSource,
    WorkflowFailedEvent,
    WorkflowLifecycleEvent,
    WorkflowOutputEvent,
    WorkflowStartedEvent,
    WorkflowStatusEvent,
    WorkflowWarningEvent,
    _framework_event_origin,  # type: ignore
)
from ._runner_context import Message, RunnerContext
from ._shared_state import SharedState

T_Out = TypeVar("T_Out", default=Never)
T_W_Out = TypeVar("T_W_Out", default=Never)


logger = logging.getLogger(__name__)


def infer_output_types_from_ctx_annotation(ctx_annotation: Any) -> tuple[list[type[Any]], list[type[Any]]]:
    """WorkflowContextのジェネリックパラメータからメッセージタイプとワークフロー出力タイプを推論します。

    Examples:
    - WorkflowContext -> ([], [])
    - WorkflowContext[str] -> ([str], [])
    - WorkflowContext[str, int] -> ([str], [int])
    - WorkflowContext[str | int, bool | int] -> ([str, int], [bool, int])
    - WorkflowContext[Union[str, int], Union[bool, int]] -> ([str, int], [bool, int])
    - WorkflowContext[Any] -> ([Any], [])
    - WorkflowContext[Any, Any] -> ([Any], [Any])
    - WorkflowContext[Never, Never] -> ([], [])
    - WorkflowContext[Never, int] -> ([], [int])

    Returns:
        (message_types, workflow_output_types) のタプル
    """
    # アノテーションがないかパラメータ化されていない場合は、空のリストを返します
    try:
        origin = get_origin(ctx_annotation)
    except Exception:
        origin = None

    # アノテーションが非添字のWorkflowContextの場合、推論するものはありません
    if origin is None:
        return [], []

    # WorkflowContext[T_Out, T_W_Out]を期待します
    if origin is not WorkflowContext:
        return [], []

    args = list(get_args(ctx_annotation))
    if not args:
        return [], []

    # WorkflowContext[T_Out] -> T_Outからmessage_typesを取得し、workflow output typesはなし
    if len(args) == 1:
        t = args[0]
        t_origin = get_origin(t)
        if t is Any:
            return [cast(type[Any], Any)], []

        if t_origin in (Union, UnionType):
            message_types = [arg for arg in get_args(t) if arg is not Any and arg is not Never]
            return message_types, []

        if t is Never:
            return [], []
        return [t], []

    # WorkflowContext[T_Out, T_W_Out] ->
    # T_Outからmessage_typesを、T_W_Outからworkflow_output_typesを取得します
    t_out, t_w_out = args[:2]  # 引数が複数ある場合は最初の2つを取得します

    # message_typesのためにT_Outを処理します
    message_types = []
    t_out_origin = get_origin(t_out)
    if t_out is Any:
        message_types = [cast(type[Any], Any)]
    elif t_out is not Never:
        if t_out_origin in (Union, UnionType):
            message_types = [arg for arg in get_args(t_out) if arg is not Any and arg is not Never]
        else:
            message_types = [t_out]

    # workflow_output_typesのためにT_W_Outを処理します
    workflow_output_types = []
    t_w_out_origin = get_origin(t_w_out)
    if t_w_out is Any:
        workflow_output_types = [cast(type[Any], Any)]
    elif t_w_out is not Never:
        if t_w_out_origin in (Union, UnionType):
            workflow_output_types = [arg for arg in get_args(t_w_out) if arg is not Any and arg is not Never]
        else:
            workflow_output_types = [t_w_out]

    return message_types, workflow_output_types


def _is_workflow_context_type(annotation: Any) -> bool:
    """アノテーションがWorkflowContext、WorkflowContext[T]、またはWorkflowContext[T, U]を表しているかどうかをチェックします"""
    origin = get_origin(annotation)
    if origin is WorkflowContext:
        return True
    # 生のクラスが使われている場合も処理します
    return annotation is WorkflowContext


def validate_workflow_context_annotation(
    annotation: Any,
    parameter_name: str,
    context_description: str,
) -> tuple[list[type[Any]], list[type[Any]]]:
    """WorkflowContextのアノテーションを検証し、推論された型を返します。

    Args:
        annotation: 検証する型アノテーション
        parameter_name: パラメータ名（エラーメッセージ用）
        context_description: コンテキストの説明（例: "Function func1", "Handler method"）

    Returns:
        (output_types, workflow_output_types) のタプル

    Raises:
        ValueError: アノテーションが無効な場合に発生します
    """
    if annotation == inspect.Parameter.empty:
        raise ValueError(
            f"{context_description} {parameter_name} must have a WorkflowContext, "
            f"WorkflowContext[T] or WorkflowContext[T, U] type annotation, "
            f"where T is output message type and U is workflow output type"
        )

    if not _is_workflow_context_type(annotation):
        raise ValueError(
            f"{context_description} {parameter_name} must be annotated as "
            f"WorkflowContext, WorkflowContext[T], or WorkflowContext[T, U], "
            f"got {annotation}"
        )

    # WorkflowContext[T]またはWorkflowContext[T, U]の型引数を検証します
    type_args = get_args(annotation)

    if len(type_args) > 2:
        raise ValueError(
            f"{context_description} {parameter_name} must have at most 2 type arguments, "
            "WorkflowContext, WorkflowContext[T], or WorkflowContext[T, U], "
            f"got {len(type_args)} arguments"
        )

    if type_args:
        # 値が有効な型アノテーションかどうかをチェックするヘルパー関数
        def _is_type_like(x: Any) -> bool:
            """値が型のようなエンティティ（クラス、type、またはtypingの構造体）かどうかをチェックします"""
            return isinstance(x, type) or get_origin(x) is not None or x is Never

        for i, type_arg in enumerate(type_args):
            param_description = "T_Out" if i == 0 else "T_W_Out"

            # Anyを明示的に許可します
            if type_arg is Any:
                continue

            # ユニオン型かどうかをチェックし、各メンバーを検証します
            union_origin = get_origin(type_arg)
            if union_origin in (Union, UnionType):
                union_members = get_args(type_arg)
                invalid_members = [m for m in union_members if not _is_type_like(m) and m is not Any]
                if invalid_members:
                    raise ValueError(
                        f"{context_description} {parameter_name} {param_description} "
                        f"contains invalid type entries: {invalid_members}. "
                        f"Use proper types or typing generics"
                    )
            else:
                # 有効な型かどうかをチェックします
                if not _is_type_like(type_arg):
                    raise ValueError(
                        f"{context_description} {parameter_name} {param_description} "
                        f"contains invalid type entry: {type_arg}. "
                        f"Use proper types or typing generics"
                    )

    return infer_output_types_from_ctx_annotation(annotation)


def validate_function_signature(
    func: Callable[..., Any], context_description: str
) -> tuple[type, Any, list[type[Any]], list[type[Any]]]:
    """executor関数の関数シグネチャを検証します。

    Args:
        func: 検証する関数
        context_description: エラーメッセージ用の説明（例: "Function", "Handler method"）

    Returns:
        (message_type, ctx_annotation, output_types, workflow_output_types) のタプル

    Raises:
        ValueError: 関数シグネチャが無効な場合に発生します
    """
    signature = inspect.signature(func)
    params = list(signature.parameters.values())

    # コンテキストに基づいて期待されるパラメータ数を決定します
    expected_counts: tuple[int, ...]
    if context_description.startswith("Function"):
        # 関数executor: (message) または (message, ctx)
        expected_counts = (1, 2)
        param_description = "(message: T) or (message: T, ctx: WorkflowContext[U])"
    else:
        # ハンドラメソッド: (self, message, ctx)
        expected_counts = (3,)
        param_description = "(self, message: T, ctx: WorkflowContext[U])"

    if len(params) not in expected_counts:
        raise ValueError(
            f"{context_description} {func.__name__} must have {param_description}. Got {len(params)} parameters."
        )

    # メッセージパラメータを抽出します（関数はインデックス0、メソッドはインデックス1）
    message_param_idx = 0 if context_description.startswith("Function") else 1
    message_param = params[message_param_idx]

    # メッセージパラメータに型アノテーションがあるかチェックします
    if message_param.annotation == inspect.Parameter.empty:
        raise ValueError(f"{context_description} {func.__name__} must have a type annotation for the message parameter")

    message_type = message_param.annotation

    # コンテキストパラメータがあるかチェックします
    ctx_param_idx = message_param_idx + 1
    if len(params) > ctx_param_idx:
        ctx_param = params[ctx_param_idx]
        output_types, workflow_output_types = validate_workflow_context_annotation(
            ctx_param.annotation, f"parameter '{ctx_param.name}'", context_description
        )
        ctx_annotation = ctx_param.annotation
    else:
        # コンテキストパラメータなし（関数executorのみ有効）
        if not context_description.startswith("Function"):
            raise ValueError(f"{context_description} {func.__name__} must have a WorkflowContext parameter")
        output_types, workflow_output_types = [], []
        ctx_annotation = None

    return message_type, ctx_annotation, output_types, workflow_output_types


_FRAMEWORK_LIFECYCLE_EVENT_TYPES: tuple[type[WorkflowEvent], ...] = cast(
    tuple[type[WorkflowEvent], ...],
    tuple(get_args(WorkflowLifecycleEvent))
    or (
        WorkflowStartedEvent,
        WorkflowStatusEvent,
        WorkflowFailedEvent,
    ),
)


class WorkflowContext(Generic[T_Out, T_W_Out]):
    """executorがワークフローや他のexecutorとやり取りするための実行コンテキスト。

    ## 概要
    WorkflowContextは、executorがメッセージを送信し、出力をyieldし、状態を管理し、より広範なワークフローエコシステムとやり取りするための制御されたインターフェースを提供します。ジェネリックパラメータによって型安全性を強制しつつ、内部のランタイムコンポーネントへの直接アクセスを防ぎます。

    ## 型パラメータ
    コンテキストは異なる操作の型安全性を強制するためにパラメータ化されています：

    ### WorkflowContext（パラメータなし）
    メッセージ送信や出力yieldを行わず副作用のみを実行するexecutor向け：

    .. code-block:: python

        async def log_handler(message: str, ctx: WorkflowContext) -> None:
            print(f"Received: {message}")  # 副作用のみ

    ### WorkflowContext[T_Out]
    他のexecutorにT_Out型のメッセージを送信可能：

    .. code-block:: python

        async def processor(message: str, ctx: WorkflowContext[int]) -> None:
            result = len(message)
            await ctx.send_message(result)  # 下流のexecutorにintを送信

    ### WorkflowContext[T_Out, T_W_Out]
    メッセージ送信（T_Out）とワークフロー出力のyield（T_W_Out）の両方を可能にします：

    .. code-block:: python

        async def dual_output(message: str, ctx: WorkflowContext[int, str]) -> None:
            await ctx.send_message(42)  # intメッセージを送信
            await ctx.yield_output("complete")  # strのワークフロー出力をyield

    ### Union型
    複数の型はユニオン表記で指定可能：

    .. code-block:: python

        async def flexible(message: str, ctx: WorkflowContext[int | str, bool | dict]) -> None:
            await ctx.send_message("text")  # または42を送信
            await ctx.yield_output(True)  # または{"status": "done"}をyield

    """

    def __init__(
        self,
        executor_id: str,
        source_executor_ids: list[str],
        shared_state: SharedState,
        runner_context: RunnerContext,
        trace_contexts: list[dict[str, str]] | None = None,
        source_span_ids: list[str] | None = None,
    ):
        """指定されたworkflow contextでexecutorコンテキストを初期化します。

        Args:
            executor_id: このコンテキストが属するexecutorの一意の識別子
            source_executor_ids: このexecutorにメッセージを送ったソースexecutorのIDリスト。複数のソースから集約メッセージを受け取るfan_inシナリオをサポートするためリストです。
            shared_state: ワークフローの共有状態
            runner_context: メッセージやイベント送信メソッドを提供するrunnerコンテキスト
            trace_contexts: OpenTelemetry伝播のための複数ソースからのオプショナルなトレースコンテキスト
            source_span_ids: リンク付け用の複数ソースからのオプショナルなソーススパンID（ネスト用ではありません）

        """
        self._executor_id = executor_id
        self._source_executor_ids = source_executor_ids
        self._runner_context = runner_context
        self._shared_state = shared_state

        # リンク付けのためのトレースコンテキストとソーススパンIDを保存します（複数ソース対応）
        self._trace_contexts = trace_contexts or []
        self._source_span_ids = source_span_ids or []

        if not self._source_executor_ids:
            raise ValueError("source_executor_ids cannot be empty. At least one source executor ID is required.")

    async def send_message(self, message: T_Out, target_id: str | None = None) -> None:
        """ワークフローコンテキストにメッセージを送信します。

        Args:
            message: 送信するメッセージ。このコンテキストで宣言された出力型に準拠している必要があります。
            target_id: メッセージ送信先のexecutorのID。Noneの場合はすべてのターゲットexecutorに送信されます。

        """
        global OBSERVABILITY_SETTINGS
        from ..observability import OBSERVABILITY_SETTINGS

        # パブリッシングスパンを作成します（現在のトレースコンテキストを自動的に継承）
        attributes: dict[str, str] = {OtelAttr.MESSAGE_TYPE: type(message).__name__}
        if target_id:
            attributes[OtelAttr.MESSAGE_DESTINATION_EXECUTOR_ID] = target_id
        with create_workflow_span(OtelAttr.MESSAGE_SEND_SPAN, attributes, kind=SpanKind.PRODUCER) as span:
            # Messageラッパーを作成します
            msg = Message(data=message, source_id=self._executor_id, target_id=target_id)

            # トレースが有効な場合、現在のトレースコンテキストを注入します
            if OBSERVABILITY_SETTINGS.ENABLED and span and span.is_recording():  # type: ignore[name-defined]
                trace_context: dict[str, str] = {}
                inject(trace_context)  # メッセージ伝播のために現在のトレースコンテキストを注入します

                msg.trace_contexts = [trace_context]
                msg.source_span_ids = [format(span.get_span_context().span_id, "016x")]

            await self._runner_context.send_message(msg)

    async def yield_output(self, output: T_W_Out) -> None:
        """ワークフローの出力を設定します。

        Args:
            output: yieldする出力。このコンテキストで宣言されたワークフロー出力型に準拠している必要があります。

        """
        with _framework_event_origin():
            event = WorkflowOutputEvent(data=output, source_executor_id=self._executor_id)
        await self._runner_context.add_event(event)

    async def add_event(self, event: WorkflowEvent) -> None:
        """ワークフローコンテキストにイベントを追加します。"""
        if event.origin == WorkflowEventSource.EXECUTOR and isinstance(event, _FRAMEWORK_LIFECYCLE_EVENT_TYPES):
            event_name = event.__class__.__name__
            warning_msg = (
                f"Executor '{self._executor_id}' attempted to emit {event_name}, "
                "which is reserved for framework lifecycle notifications. The "
                "event was ignored."
            )
            logger.warning(warning_msg)
            await self._runner_context.add_event(WorkflowWarningEvent(warning_msg))
            return
        await self._runner_context.add_event(event)

    async def get_shared_state(self, key: str) -> Any:
        """共有状態から値を取得します。"""
        return await self._shared_state.get(key)

    async def set_shared_state(self, key: str, value: Any) -> None:
        """共有状態に値を設定します。"""
        await self._shared_state.set(key, value)

    def get_source_executor_id(self) -> str:
        """このexecutorにメッセージを送ったソースexecutorのIDを取得します。

        Raises:
            RuntimeError: ソースexecutorが複数ある場合、このメソッドはエラーを発生させます。

        """
        if len(self._source_executor_ids) > 1:
            raise RuntimeError(
                "Cannot get source executor ID when there are multiple source executors. "
                "Access the full list via the source_executor_ids property instead."
            )
        return self._source_executor_ids[0]

    @property
    def source_executor_ids(self) -> list[str]:
        """このexecutorにメッセージを送ったソースexecutorのIDリストを取得します。"""
        return self._source_executor_ids

    @property
    def shared_state(self) -> SharedState:
        """共有状態を取得します。"""
        return self._shared_state

    async def set_executor_state(self, state: dict[str, Any]) -> None:
        """予約されたキーの下にexecutorの状態を共有状態に保存します。

        executorは、再開に必要な最小限の状態をキャプチャしたJSONシリアライズ可能な辞書をこれで呼び出します。以前保存された状態は置き換えられます。

        """
        has_existing_states = await self._shared_state.has(EXECUTOR_STATE_KEY)
        if has_existing_states:
            existing_states = await self._shared_state.get(EXECUTOR_STATE_KEY)
        else:
            existing_states = {}

        if not isinstance(existing_states, dict):
            raise ValueError("Existing executor states in shared state is not a dictionary.")

        existing_states[self._executor_id] = state
        await self._shared_state.set(EXECUTOR_STATE_KEY, existing_states)

    async def get_executor_state(self) -> dict[str, Any] | None:
        """このexecutorの以前に永続化された状態を取得します（存在する場合）。"""
        has_existing_states = await self._shared_state.has(EXECUTOR_STATE_KEY)
        if not has_existing_states:
            return None

        existing_states = await self._shared_state.get(EXECUTOR_STATE_KEY)
        if not isinstance(existing_states, dict):
            raise ValueError("Existing executor states in shared state is not a dictionary.")

        return existing_states.get(self._executor_id)

    def is_streaming(self) -> bool:
        """ワークフローがストリーミングモードで実行されているかどうかをチェックします。

        Returns:
            run_stream()で開始された場合はTrue、run()で開始された場合はFalseを返します。

        """
        return self._runner_context.is_streaming()
