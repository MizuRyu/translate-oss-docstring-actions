# Copyright (c) Microsoft. All rights reserved.

import contextlib
import importlib
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Generic, TypeVar, cast

from ._events import (
    RequestInfoEvent,  # type: ignore[reportPrivateUsage]
)
from ._executor import Executor, handler
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class PendingRequestDetails:
    """チェックポイントにキャプチャされた保留中リクエストの軽量情報。"""

    request_id: str
    prompt: str | None = None
    draft: str | None = None
    iteration: int | None = None
    source_executor_id: str | None = None
    original_request: "RequestInfoMessage | dict[str, Any] | None" = None


@dataclass
class PendingRequestSnapshot:
    """内部追跡のための保留中リクエストのスナップショット。

    このスナップショットはJSONシリアライズ可能であり、
    必要に応じて元のリクエストを再構築するのに十分な情報を含みます。

    """

    request_id: str
    source_executor_id: str
    request_type: str
    request_as_json_safe_dict: dict[str, Any]


@dataclass
class RequestInfoMessage:
    """ワークフロー内のすべてのリクエストメッセージの基底クラス。

    RequestInfoExecutorにルーティングされるべきメッセージはすべてこのクラスを継承しなければなりません。
    これにより型安全性が保証され、リクエスト／レスポンスパターンが明示されます。

    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for correlating requests and responses."""

    source_executor_id: str | None = None
    """ID of the executor expecting a response to this request.
    May differ from the executor that sent the request if intercepted and forwarded."""


TRequest = TypeVar("TRequest", bound="RequestInfoMessage")
TResponse = TypeVar("TResponse")


@dataclass
class RequestResponse(Generic[TRequest, TResponse]):
    """ワークフローにおけるリクエスト／レスポンス相関のためのレスポンスタイプ。

    このタイプはRequestInfoExecutorによって使用され、
    元のリクエストコンテキストを含む相関レスポンスを作成し、適切なメッセージルーティングを可能にします。

    """

    data: TResponse
    """The response data returned from handling the request."""

    original_request: TRequest
    """The original request that this response corresponds to."""

    request_id: str
    """The ID of the original request."""


# endregion: Request/Response Types region Request Info Executor
class RequestInfoExecutor(Executor):
    """ワークフロー内のリクエスト／レスポンスパターンを処理する組み込みExecutor。

    このExecutorは外部情報リクエストのゲートウェイとして機能します。
    リクエストメッセージを受け取るとリクエスト詳細を保存しRequestInfoEventを発行します。
    外部からレスポンスが提供されると、それをメッセージとして発行します。

    """

    _PENDING_SHARED_STATE_KEY: ClassVar[str] = "_af_pending_request_info"

    def __init__(self, id: str):
        """一意のIDでRequestInfoExecutorを初期化します。

        Args:
            id: このRequestInfoExecutorの一意のID。

        """
        super().__init__(id=id)
        self._request_events: dict[str, RequestInfoEvent] = {}

    # region Public Methods

    @handler
    async def handle_request(self, message: RequestInfoMessage, ctx: WorkflowContext) -> None:
        """指定されたメッセージでRequestInfoExecutorを実行します。"""
        # messageのsource_executor_idを使用可能なら使い、そうでなければcontextを使用します。
        source_executor_id = message.source_executor_id or ctx.get_source_executor_id()

        event = RequestInfoEvent(
            request_id=message.request_id,
            source_executor_id=source_executor_id,
            request_type=type(message),
            request_data=message,
        )
        self._request_events[message.request_id] = event
        await self._record_pending_request(message, source_executor_id, ctx)
        await ctx.add_event(event)

    async def handle_response(
        self,
        response_data: Any,
        request_id: str,
        ctx: WorkflowContext[RequestResponse[RequestInfoMessage, Any]],
    ) -> None:
        """リクエストへのレスポンスを処理します。

        Args:
            request_id: このレスポンスが対応するリクエストのID。
            response_data: レスポンスで返されるデータ。
            ctx: レスポンス送信のためのworkflowコンテキスト。

        """
        event = self._request_events.get(request_id)
        if event is None:
            event = await self._rehydrate_request_event(request_id, cast(WorkflowContext, ctx))
        if event is None:
            raise ValueError(f"No request found with ID: {request_id}")

        self._request_events.pop(request_id, None)

        # レスポンスデータと元のリクエストの両方を含む相関レスポンスを作成します。
        if not isinstance(event.data, RequestInfoMessage):
            raise TypeError(f"Expected RequestInfoMessage, got {type(event.data)}")
        correlated_response = RequestResponse(data=response_data, original_request=event.data, request_id=request_id)
        await ctx.send_message(correlated_response, target_id=event.source_executor_id)

        await self._erase_pending_request(request_id, cast(WorkflowContext, ctx))

    def snapshot_state(self) -> dict[str, Any]:
        """チェックポイント復元がシームレスに再開できるように保留中リクエストをシリアライズします。"""

        def _encode_event(event: RequestInfoEvent) -> dict[str, Any] | None:
            if event.data is None or not isinstance(event.data, RequestInfoMessage):
                logger.warning(
                    f"RequestInfoExecutor {self.id} encountered invalid event data for request ID {event.request_id}: "
                    f"{type(event.data).__name__}. This request will be skipped in the checkpoint."
                )
                return None

            payload = self._encode_request_payload(event.data, event.data.__class__)

            return {
                "source_executor_id": event.source_executor_id,
                "request_type": f"{event.request_type.__module__}:{event.request_type.__qualname__}",
                "request_data": payload,
            }

        return {
            "request_events": {
                rid: encoded
                for rid, event in self._request_events.items()
                if (encoded := _encode_event(event)) is not None
            },
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """チェックポイント状態から保留中リクエストの管理情報を復元します。"""
        self._request_events.clear()
        stored_events = state.get("request_events", {})

        for request_id, payload in stored_events.items():
            request_type_qual = payload.get("request_type", "")
            try:
                request_type = _import_qualname(request_type_qual)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.debug(
                    "RequestInfoExecutor %s failed to import %s during restore: %s",
                    self.id,
                    request_type_qual,
                    exc,
                )
                request_type = RequestInfoMessage
            request_data_meta = payload.get("request_data", {})
            request_data = self._decode_request_data(request_data_meta)
            event = RequestInfoEvent(
                request_id=request_id,
                source_executor_id=payload.get("source_executor_id", ""),
                request_type=request_type,
                request_data=request_data,
            )
            self._request_events[request_id] = event

    async def has_pending_request(self, request_id: str, ctx: WorkflowContext) -> bool:
        """指定されたIDの保留中のRequestがあるかどうかを確認します。

        Args:
            request_id: 確認するRequestのID。
            ctx: 必要に応じてStateにアクセスするためのworkflowのContext。

        Returns: Requestが保留中であればTrue、そうでなければFalse。
        """
        if request_id in self._request_events:
            return True

        pending_requests = await self._retrieve_existing_pending_requests(ctx)
        return request_id in pending_requests

    # endregion: Public Methods region: Internal Methods

    async def _record_pending_request(
        self,
        message: RequestInfoMessage,
        source_executor_id: str,
        ctx: WorkflowContext,
    ) -> None:
        """チェックポイント目的でExecutorのStateに保留中のRequestを記録します。"""
        pending_request_snapshot = self._build_pending_request_snapshot(message, source_executor_id)

        existing_pending_requests = await self._retrieve_existing_pending_requests(ctx)
        existing_pending_requests[message.request_id] = pending_request_snapshot

        await self._persist_to_executor_state(existing_pending_requests, ctx)

    async def _erase_pending_request(self, request_id: str, ctx: WorkflowContext) -> None:
        """チェックポイント目的で処理済みの保留中RequestをExecutorのStateから消去します。"""
        existing_pending_requests = await self._retrieve_existing_pending_requests(ctx)
        if request_id in existing_pending_requests:
            existing_pending_requests.pop(request_id)
            await self._persist_to_executor_state(existing_pending_requests, ctx)

    async def _retrieve_existing_pending_requests(self, ctx: WorkflowContext) -> dict[str, PendingRequestSnapshot]:
        """ExecutorのStateから既存の保留中Requestを取得します。"""
        executor_state = await ctx.get_executor_state()
        if executor_state is None:
            return {}

        stored_requests = executor_state.get(self._PENDING_SHARED_STATE_KEY, {})
        if not isinstance(stored_requests, dict):
            raise TypeError(f"Unexpected type for pending requests: {type(stored_requests).__name__}")

        # 内容を検証します
        for key, value in stored_requests.items():  # type: ignore
            if not isinstance(key, str) or not isinstance(value, PendingRequestSnapshot):
                raise TypeError(
                    "Invalid pending request entry in executor state. "
                    "Key must be `str` and value must be `PendingRequestSnapshot`."
                )

        return stored_requests  # type: ignore

    async def _persist_to_executor_state(
        self, pending: dict[str, PendingRequestSnapshot], ctx: WorkflowContext
    ) -> None:
        """現在の保留中RequestをExecutorのStateに永続化します。"""
        executor_state = await ctx.get_executor_state() or {}
        executor_state[self._PENDING_SHARED_STATE_KEY] = pending
        await ctx.set_executor_state(executor_state)

    def _build_pending_request_snapshot(
        self, request: RequestInfoMessage, source_executor_id: str
    ) -> PendingRequestSnapshot:
        """チェックポイント用に保留中Requestのスナップショットを作成します。"""
        request_as_json_safe_dict = self._convert_request_to_json_safe_dict(request)

        return PendingRequestSnapshot(
            request_id=request.request_id,
            source_executor_id=source_executor_id,
            request_type=f"{type(request).__module__}:{type(request).__name__}",
            request_as_json_safe_dict=request_as_json_safe_dict,
        )

    def _encode_request_payload(self, request_data: RequestInfoMessage, data_cls: type[Any]) -> dict[str, Any]:
        if is_dataclass(request_data) and not isinstance(request_data, type):
            dataclass_instance = cast(Any, request_data)
            safe_value = _make_json_safe(asdict(dataclass_instance))
            return {
                "kind": "dataclass",
                "type": f"{data_cls.__module__}:{data_cls.__qualname__}",
                "value": safe_value,
            }

        to_dict_fn = getattr(request_data, "to_dict", None)
        if callable(to_dict_fn):
            try:
                dumped = to_dict_fn()
            except TypeError:
                dumped = to_dict_fn()
            safe_value = _make_json_safe(dumped)
            return {
                "kind": "dict",
                "type": f"{data_cls.__module__}:{data_cls.__qualname__}",
                "value": safe_value,
            }

        to_json_fn = getattr(request_data, "to_json", None)
        if callable(to_json_fn):
            try:
                dumped = to_json_fn()
            except TypeError:
                dumped = to_json_fn()
            converted = dumped
            if isinstance(dumped, (str, bytes, bytearray)):
                decoded: str | bytes | bytearray
                if isinstance(dumped, (bytes, bytearray)):
                    try:
                        decoded = dumped.decode()
                    except Exception:
                        decoded = dumped
                else:
                    decoded = dumped
                try:
                    converted = json.loads(decoded)
                except Exception:
                    converted = decoded
            safe_value = _make_json_safe(converted)
            return {
                "kind": "dict" if isinstance(converted, dict) else "json",
                "type": f"{data_cls.__module__}:{data_cls.__qualname__}",
                "value": safe_value,
            }

        return {
            "kind": "raw",
            "type": f"{data_cls.__module__}:{data_cls.__qualname__}",
            "value": self._convert_request_to_json_safe_dict(request_data),
        }

    def _decode_request_data(self, metadata: dict[str, Any]) -> RequestInfoMessage:
        kind = metadata.get("kind")
        type_name = metadata.get("type", "")
        value: Any = metadata.get("value", {})
        if type_name:
            try:
                imported = _import_qualname(type_name)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.debug(
                    "RequestInfoExecutor %s failed to import %s during decode: %s",
                    self.id,
                    type_name,
                    exc,
                )
                imported = RequestInfoMessage
        else:
            imported = RequestInfoMessage
        target_cls: type[RequestInfoMessage]
        if isinstance(imported, type) and issubclass(imported, RequestInfoMessage):
            target_cls = imported
        else:
            target_cls = RequestInfoMessage

        if kind == "dataclass" and isinstance(value, dict):
            with contextlib.suppress(TypeError):
                return target_cls(**value)  # type: ignore[arg-type]

        # 以前はpydanticを"dict"として保存していたチェックポイントの後方互換処理
        if kind in {"dict", "pydantic", "json"} and isinstance(value, dict):
            from_dict = getattr(target_cls, "from_dict", None)
            if callable(from_dict):
                with contextlib.suppress(Exception):
                    return cast(RequestInfoMessage, from_dict(value))

        if kind == "json" and isinstance(value, str):
            from_json = getattr(target_cls, "from_json", None)
            if callable(from_json):
                with contextlib.suppress(Exception):
                    return cast(RequestInfoMessage, from_json(value))
            with contextlib.suppress(Exception):
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return self._decode_request_data({"kind": "dict", "type": type_name, "value": parsed})

        if isinstance(value, dict):
            with contextlib.suppress(TypeError):
                return target_cls(**value)  # type: ignore[arg-type]
            instance = object.__new__(target_cls)
            instance.__dict__.update(value)  # type: ignore[arg-type]
            return instance

        with contextlib.suppress(Exception):
            return target_cls()
        return RequestInfoMessage()

    def _convert_request_to_json_safe_dict(self, request: RequestInfoMessage) -> dict[str, Any]:
        try:
            data = _make_json_safe(asdict(request))
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            raise ValueError(f"Failed to convert {type(request).__name__} to dict")
        except Exception as exc:
            logger.error(f"RequestInfoExecutor {self.id} failed to serialize request: {exc}")
            raise RuntimeError(
                f"Failed to serialize request `{type(request).__name__}`: {exc}\n"
                "Make sure request is a dataclass and derive from `RequestInfoMessage`."
            ) from exc

    async def _rehydrate_request_event(self, request_id: str, ctx: WorkflowContext) -> RequestInfoEvent | None:
        pending_requests = await self._retrieve_existing_pending_requests(ctx)
        if (snapshot := pending_requests.get(request_id)) is None:
            return None

        request = self._construct_request_from_snapshot(snapshot)
        if request is None:
            return None

        event = RequestInfoEvent(
            request_id=request_id,
            source_executor_id=snapshot.source_executor_id,
            request_type=type(request),
            request_data=request,
        )
        self._request_events[request_id] = event
        return event

    def _construct_request_from_snapshot(self, snapshot: PendingRequestSnapshot) -> RequestInfoMessage | None:
        json_safe_dict = snapshot.request_as_json_safe_dict

        request_cls: type[RequestInfoMessage] = RequestInfoMessage
        request_type_str = snapshot.request_type
        if isinstance(request_type_str, str) and ":" in request_type_str:
            module_name, class_name = request_type_str.split(":", 1)
            try:
                module = importlib.import_module(module_name)
                candidate = getattr(module, class_name)
                if isinstance(candidate, type) and issubclass(candidate, RequestInfoMessage):
                    request_cls = candidate
            except Exception as exc:
                logger.warning(f"RequestInfoExecutor {self.id} could not import {module_name}.{class_name}: {exc}")
                request_cls = RequestInfoMessage

        request: RequestInfoMessage | None = self._instantiate_request(request_cls, json_safe_dict)

        if request is None and request_cls is not RequestInfoMessage:
            request = self._instantiate_request(RequestInfoMessage, json_safe_dict)

        if request is None:
            logger.warning(
                f"RequestInfoExecutor {self.id} could not reconstruct request "
                f"{request_type_str or RequestInfoMessage.__name__} from snapshot keys {sorted(json_safe_dict.keys())}"
            )
            return None

        for key, value in json_safe_dict.items():
            if key == "request_id":
                continue
            try:
                setattr(request, key, value)
            except Exception as exc:
                logger.debug(
                    f"RequestInfoExecutor {self.id} could not set attribute {key} on {type(request).__name__}: {exc}"
                )
                continue

        snapshot_request_id = snapshot.request_id
        if isinstance(snapshot_request_id, str) and snapshot_request_id:
            try:
                request.request_id = snapshot_request_id
            except Exception as exc:
                logger.debug(
                    f"RequestInfoExecutor {self.id} could not apply snapshot "
                    f"request_id to {type(request).__name__}: {exc}"
                )

        return request

    def _instantiate_request(
        self,
        request_cls: type[RequestInfoMessage],
        details: dict[str, Any],
    ) -> RequestInfoMessage | None:
        try:
            from_dict = getattr(request_cls, "from_dict", None)
            if callable(from_dict):
                return cast(RequestInfoMessage, from_dict(details))
        except (TypeError, ValueError) as exc:
            logger.debug(f"RequestInfoExecutor {self.id} failed to hydrate {request_cls.__name__} via from_dict: {exc}")
        except Exception as exc:
            logger.warning(
                f"RequestInfoExecutor {self.id} encountered unexpected error during "
                f"{request_cls.__name__}.from_dict: {exc}"
            )

        if is_dataclass(request_cls):
            try:
                field_names = {f.name for f in fields(request_cls)}
                ctor_kwargs = {name: details[name] for name in field_names if name in details}
                return request_cls(**ctor_kwargs)
            except (TypeError, ValueError) as exc:
                logger.debug(
                    f"RequestInfoExecutor {self.id} could not instantiate dataclass "
                    f"{request_cls.__name__} with snapshot data: {exc}"
                )
            except Exception as exc:
                logger.warning(
                    f"RequestInfoExecutor {self.id} encountered unexpected error "
                    f"constructing dataclass {request_cls.__name__}: {exc}"
                )

        try:
            instance = request_cls()
        except Exception as exc:
            logger.warning(
                f"RequestInfoExecutor {self.id} could not instantiate {request_cls.__name__} without arguments: {exc}"
            )
            return None

        for key, value in details.items():
            if key == "request_id":
                continue
            try:
                setattr(instance, key, value)
            except Exception as exc:
                logger.debug(
                    f"RequestInfoExecutor {self.id} could not set attribute {key} on "
                    f"{request_cls.__name__} during instantiation: {exc}"
                )
                continue

        return instance

    # endregion: Internal Methods


# region: Utility Functions


def _make_json_safe(value: Any) -> Any:
    """値を再帰的にJSON安全な表現に変換します。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        safe_dict: dict[str, Any] = {}
        for key, val in value.items():  # type: ignore[attr-defined]
            safe_dict[str(key)] = _make_json_safe(val)  # type: ignore[arg-type]
        return safe_dict
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_make_json_safe(item) for item in value]  # type: ignore[misc]
    return repr(value)


def _import_qualname(qualname: str) -> type[Any]:
    """'module:TypeName'形式の修飾名から型をImportします。"""
    module_name, _, type_name = qualname.partition(":")
    if not module_name or not type_name:
        raise ValueError(f"Invalid qualified name: {qualname}")
    module = importlib.import_module(module_name)
    attr: Any = module
    for part in type_name.split("."):
        attr = getattr(attr, part)
    if not isinstance(attr, type):
        raise TypeError(f"Resolved object is not a type: {qualname}")
    return attr


# endregion: Utility Functions
