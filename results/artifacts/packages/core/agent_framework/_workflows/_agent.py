# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    FunctionApprovalRequestContent,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    UsageDetails,
)

from ..exceptions import AgentExecutionException
from ._events import (
    AgentRunUpdateEvent,
    RequestInfoEvent,
    WorkflowEvent,
)
from ._message_utils import normalize_messages_input
from ._typing_utils import is_type_compatible

if TYPE_CHECKING:
    from ._workflow import Workflow

logger = logging.getLogger(__name__)


class WorkflowAgent(BaseAgent):
    """ワークフローをラップし、エージェントとして公開する`Agent`のサブクラス。"""

    # リクエスト情報関数名のクラス変数
    REQUEST_INFO_FUNCTION_NAME: ClassVar[str] = "request_info"

    @dataclass
    class RequestInfoFunctionArgs:
        request_id: str
        data: Any

        def to_dict(self) -> dict[str, Any]:
            return {"request_id": self.request_id, "data": self.data}

        def to_json(self) -> str:
            return json.dumps(self.to_dict())

        @classmethod
        def from_dict(cls, payload: dict[str, Any]) -> "WorkflowAgent.RequestInfoFunctionArgs":
            return cls(request_id=payload.get("request_id", ""), data=payload.get("data"))

        @classmethod
        def from_json(cls, raw: str) -> "WorkflowAgent.RequestInfoFunctionArgs":
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("RequestInfoFunctionArgs JSON payload must decode to a mapping")
            return cls.from_dict(data)

    def __init__(
        self,
        workflow: "Workflow",
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """WorkflowAgentを初期化します。

        引数:
            workflow: エージェントとしてラップするワークフロー。

        キーワード引数:
            id: エージェントの一意識別子。Noneの場合は生成されます。
            name: エージェントの名前（オプション）。
            description: エージェントの説明（オプション）。
            **kwargs: BaseAgentに渡される追加のキーワード引数。

        """
        if id is None:
            id = f"WorkflowAgent_{uuid.uuid4().hex[:8]}"
        # まず標準のBaseAgentパラメータで初期化します ワークフローの開始executorがエージェント向けメッセージ入力を処理できるか検証します
        try:
            start_executor = workflow.get_start_executor()
        except KeyError as exc:  # Defensive: workflow lacks a configured entry point
            raise ValueError("Workflow's start executor is not defined.") from exc

        if not any(is_type_compatible(list[ChatMessage], input_type) for input_type in start_executor.input_types):
            raise ValueError("Workflow's start executor cannot handle list[ChatMessage]")

        super().__init__(id=id, name=name, description=description, **kwargs)
        self._workflow: "Workflow" = workflow
        self._pending_requests: dict[str, RequestInfoEvent] = {}

    @property
    def workflow(self) -> "Workflow":
        return self._workflow

    @property
    def pending_requests(self) -> dict[str, RequestInfoEvent]:
        return self._pending_requests

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """ワークフローエージェントからレスポンスを取得します（非ストリーミング）。

        このメソッドはすべてのストリーミング更新を収集し、単一のレスポンスに統合します。

        引数:
            messages: ワークフローに送信するメッセージ。

        キーワード引数:
            thread: 会話スレッド。Noneの場合は新しいスレッドが作成されます。
            **kwargs: 追加のキーワード引数。

        戻り値:
            AgentRunResponseとしての最終的なワークフローレスポンス。

        """
        # すべてのストリーミング更新を収集します
        response_updates: list[AgentRunResponseUpdate] = []
        input_messages = normalize_messages_input(messages)
        thread = thread or self.get_new_thread()
        response_id = str(uuid.uuid4())

        async for update in self._run_stream_impl(input_messages, response_id):
            response_updates.append(update)

        # 更新を最終レスポンスに変換します。
        response = self.merge_updates(response_updates, response_id)

        # スレッドに新しいメッセージ（入力およびレスポンスメッセージの両方）を通知します
        await self._notify_thread_of_new_messages(thread, input_messages, response.messages)

        return response

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """ワークフローエージェントからのレスポンス更新をストリームします。

        引数:
            messages: ワークフローに送信するメッセージ。

        キーワード引数:
            thread: 会話スレッド。Noneの場合は新しいスレッドが作成されます。
            **kwargs: 追加のキーワード引数。

        生成:
            ワークフロー実行の進行を表すAgentRunResponseUpdateオブジェクト。

        """
        input_messages = normalize_messages_input(messages)
        thread = thread or self.get_new_thread()
        response_updates: list[AgentRunResponseUpdate] = []
        response_id = str(uuid.uuid4())

        async for update in self._run_stream_impl(input_messages, response_id):
            response_updates.append(update)
            yield update

        # 更新を最終レスポンスに変換します。
        response = self.merge_updates(response_updates, response_id)

        # スレッドに新しいメッセージ（入力およびレスポンスメッセージの両方）を通知します
        await self._notify_thread_of_new_messages(thread, input_messages, response.messages)

    async def _run_stream_impl(
        self,
        input_messages: list[ChatMessage],
        response_id: str,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """ストリーミング実行の内部実装。

        引数:
            input_messages: 処理する正規化された入力メッセージ。
            response_id: このワークフロー実行の一意のレスポンスID。

        生成:
            ワークフロー実行の進行を表すAgentRunResponseUpdateオブジェクト。

        """
        # 関数レスポンスがあるかどうかに基づいてイベントストリームを決定します
        if bool(self.pending_requests):
            # これは継続です - 関数レスポンスを返すにはsend_responses_streamingを使用してください
            logger.info(f"Continuing workflow to address {len(self.pending_requests)} requests")

            # 入力メッセージから関数レスポンスを抽出し、保留中のリクエストがある場合は関数レスポンスのみがメッセージに存在することを保証します。
            function_responses = self._extract_function_responses(input_messages)

            # 満たされた保留中のリクエストをポップします。
            for request_id in list(self.pending_requests.keys()):
                if request_id in function_responses:
                    self.pending_requests.pop(request_id)

            # 注意: 一部の保留中リクエストが満たされない可能性があり、これについてはワークフローに任せます -- エージェントはこれに関して意見を持ちません。
            event_stream = self.workflow.send_responses_streaming(function_responses)
        else:
            # ストリーミングでワークフローを実行します（初回実行または関数レスポンスなし） 新しい入力メッセージを直接ワークフローに渡します。
            event_stream = self.workflow.run_stream(input_messages)

        # ストリームからのイベントを処理します。
        async for event in event_stream:
            # ワークフローイベントをエージェント更新に変換します。
            update = self._convert_workflow_event_to_agent_update(response_id, event)
            if update:
                yield update

    def _convert_workflow_event_to_agent_update(
        self,
        response_id: str,
        event: WorkflowEvent,
    ) -> AgentRunResponseUpdate | None:
        """ワークフローイベントをAgentRunResponseUpdateに変換します。

        AgentRunUpdateEventとRequestInfoEventのみ処理し、それ以外は関連しません。
        イベントが関連しない場合はNoneを返します。

        """
        match event:
            case AgentRunUpdateEvent(data=update):
                # エージェントストリーミングイベントでの更新の直接パススルー
                if update:
                    return cast(AgentRunResponseUpdate, update)
                return None

            case RequestInfoEvent(request_id=request_id):
                # 後で相関付けるために保留中のリクエストを保存します。
                self.pending_requests[request_id] = event

                args = self.RequestInfoFunctionArgs(request_id=request_id, data=event.data).to_dict()

                function_call = FunctionCallContent(
                    call_id=request_id,
                    name=self.REQUEST_INFO_FUNCTION_NAME,
                    arguments=args,
                )
                approval_request = FunctionApprovalRequestContent(
                    id=request_id,
                    function_call=function_call,
                    additional_properties={"request_id": request_id},
                )
                return AgentRunResponseUpdate(
                    contents=[function_call, approval_request],
                    role=Role.ASSISTANT,
                    author_name=self.name,
                    response_id=response_id,
                    message_id=str(uuid.uuid4()),
                    created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
            case _:
                # エージェント以外のワークフローイベントを無視します。
                pass
        # 上記2つのイベントのみを扱い、それ以外は破棄します。
        return None

    def _extract_function_responses(self, input_messages: list[ChatMessage]) -> dict[str, Any]:
        """入力メッセージから関数レスポンスを抽出します。"""
        function_responses: dict[str, Any] = {}
        for message in input_messages:
            for content in message.contents:
                if isinstance(content, FunctionApprovalResponseContent):
                    # 関数の引数を解析してリクエストペイロードを復元する
                    arguments_payload = content.function_call.arguments
                    if isinstance(arguments_payload, str):
                        try:
                            parsed_args = self.RequestInfoFunctionArgs.from_json(arguments_payload)
                        except ValueError as exc:
                            raise AgentExecutionException(
                                "FunctionApprovalResponseContent arguments must decode to a mapping."
                            ) from exc
                    elif isinstance(arguments_payload, dict):
                        parsed_args = self.RequestInfoFunctionArgs.from_dict(arguments_payload)
                    else:
                        raise AgentExecutionException(
                            "FunctionApprovalResponseContent arguments must be a mapping or JSON string."
                        )

                    request_id = parsed_args.request_id or content.id
                    if not content.approved:
                        raise AgentExecutionException(f"Request '{request_id}' was not approved by the caller.")

                    if request_id in self.pending_requests:
                        function_responses[request_id] = parsed_args.data
                    elif bool(self.pending_requests):
                        raise AgentExecutionException(
                            "Only responses for pending requests are allowed when there are outstanding approvals."
                        )
                elif isinstance(content, FunctionResultContent):
                    request_id = content.call_id
                    if request_id in self.pending_requests:
                        response_data = content.result if hasattr(content, "result") else str(content)
                        function_responses[request_id] = response_data
                    elif bool(self.pending_requests):
                        raise AgentExecutionException(
                            "Only function responses for pending requests are allowed while requests are outstanding."
                        )
                else:
                    if bool(self.pending_requests):
                        raise AgentExecutionException("Unexpected content type while awaiting request info responses.")
        return function_responses

    class _ResponseState(TypedDict):
        """message_idごとにレスポンスの更新をグループ化するための状態。"""

        by_msg: dict[str, list[AgentRunResponseUpdate]]
        dangling: list[AgentRunResponseUpdate]

    @staticmethod
    def merge_updates(updates: list[AgentRunResponseUpdate], response_id: str) -> AgentRunResponse:
        """ストリーミング更新を単一のAgentRunResponseにマージします。

        動作:
        - response_idごとに更新をグループ化し、各response_id内ではmessage_idごとにグループ化し、message_idのない更新用の保留バケットを保持します。
        - 各グループ（メッセージごとおよび保留分）をAgentRunResponse.from_agent_run_response_updatesを使って中間のAgentRunResponseに変換し、created_atでソートしてマージします。
        - response_idのない更新からのメッセージは最後に追加（グローバル保留）し、メタデータを集約します。

        Args:
            updates: マージするAgentRunResponseUpdateオブジェクトのリスト。
            response_id: 返されるAgentRunResponseに設定するレスポンス識別子。

        Returns:
            処理順にメッセージが並び、メタデータが集約されたAgentRunResponse。

        """
        # PHASE 1: RESPONSE_IDおよびMESSAGE_IDによる更新のグループ化
        states: dict[str, WorkflowAgent._ResponseState] = {}
        global_dangling: list[AgentRunResponseUpdate] = []

        for u in updates:
            if u.response_id:
                state = states.setdefault(u.response_id, {"by_msg": {}, "dangling": []})
                by_msg = state["by_msg"]
                dangling = state["dangling"]
                if u.message_id:
                    by_msg.setdefault(u.message_id, []).append(u)
                else:
                    dangling.append(u)
            else:
                global_dangling.append(u)

        # HELPER FUNCTIONS
        def _parse_dt(value: str | None) -> tuple[int, datetime | str | None]:
            if not value:
                return (1, None)
            v = value
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            try:
                return (0, datetime.fromisoformat(v))
            except Exception:
                return (0, v)

        def _sum_usage(a: UsageDetails | None, b: UsageDetails | None) -> UsageDetails | None:
            if a is None:
                return b
            if b is None:
                return a
            return a + b

        def _merge_responses(current: AgentRunResponse | None, incoming: AgentRunResponse) -> AgentRunResponse:
            if current is None:
                return incoming
            raw_list: list[object] = []

            def _add_raw(value: object) -> None:
                if isinstance(value, list):
                    raw_list.extend(cast(list[object], value))
                else:
                    raw_list.append(value)

            if current.raw_representation is not None:
                _add_raw(current.raw_representation)
            if incoming.raw_representation is not None:
                _add_raw(incoming.raw_representation)
            return AgentRunResponse(
                messages=(current.messages or []) + (incoming.messages or []),
                response_id=current.response_id or incoming.response_id,
                created_at=incoming.created_at or current.created_at,
                usage_details=_sum_usage(current.usage_details, incoming.usage_details),
                raw_representation=raw_list if raw_list else None,
                additional_properties=incoming.additional_properties or current.additional_properties,
            )

        # PHASE 2: グループ化された更新をレスポンスに変換してマージ
        final_messages: list[ChatMessage] = []
        merged_usage: UsageDetails | None = None
        latest_created_at: str | None = None
        merged_additional_properties: dict[str, Any] | None = None
        raw_representations: list[object] = []

        for grouped_response_id in states:
            state = states[grouped_response_id]
            by_msg = state["by_msg"]
            dangling = state["dangling"]

            per_message_responses: list[AgentRunResponse] = []
            for _, msg_updates in by_msg.items():
                if msg_updates:
                    per_message_responses.append(AgentRunResponse.from_agent_run_response_updates(msg_updates))
            if dangling:
                per_message_responses.append(AgentRunResponse.from_agent_run_response_updates(dangling))

            per_message_responses.sort(key=lambda r: _parse_dt(r.created_at))

            aggregated: AgentRunResponse | None = None
            for resp in per_message_responses:
                if resp.response_id and grouped_response_id and resp.response_id != grouped_response_id:
                    resp.response_id = grouped_response_id
                aggregated = _merge_responses(aggregated, resp)

            if aggregated:
                final_messages.extend(aggregated.messages)
                if aggregated.usage_details:
                    merged_usage = _sum_usage(merged_usage, aggregated.usage_details)
                if aggregated.created_at and (
                    not latest_created_at or _parse_dt(aggregated.created_at) > _parse_dt(latest_created_at)
                ):
                    latest_created_at = aggregated.created_at
                if aggregated.additional_properties:
                    if merged_additional_properties is None:
                        merged_additional_properties = {}
                    merged_additional_properties.update(aggregated.additional_properties)
                raw_value = aggregated.raw_representation
                if raw_value:
                    cast_value = cast(object | list[object], raw_value)
                    if isinstance(cast_value, list):
                        raw_representations.extend(cast(list[object], cast_value))
                    else:
                        raw_representations.append(cast_value)

        # PHASE 3: グローバル保留更新の処理（RESPONSE_IDなし）
        if global_dangling:
            flattened = AgentRunResponse.from_agent_run_response_updates(global_dangling)
            final_messages.extend(flattened.messages)
            if flattened.usage_details:
                merged_usage = _sum_usage(merged_usage, flattened.usage_details)
            if flattened.created_at and (
                not latest_created_at or _parse_dt(flattened.created_at) > _parse_dt(latest_created_at)
            ):
                latest_created_at = flattened.created_at
            if flattened.additional_properties:
                if merged_additional_properties is None:
                    merged_additional_properties = {}
                merged_additional_properties.update(flattened.additional_properties)
            flat_raw = flattened.raw_representation
            if flat_raw:
                cast_flat = cast(object | list[object], flat_raw)
                if isinstance(cast_flat, list):
                    raw_representations.extend(cast(list[object], cast_flat))
                else:
                    raw_representations.append(cast_flat)

        # PHASE 4: 入力RESPONSE_IDを用いた最終レスポンスの構築
        return AgentRunResponse(
            messages=final_messages,
            response_id=response_id,
            created_at=latest_created_at,
            usage_details=merged_usage,
            raw_representation=raw_representations if raw_representations else None,
            additional_properties=merged_additional_properties,
        )
