# Copyright (c) Microsoft. All rights reserved.

"""Agent Frameworkのメッセージmapper実装。"""

import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Union
from uuid import uuid4

from openai.types.responses import (
    Response,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseFailedEvent,
    ResponseInProgressEvent,
)

from .models import (
    AgentFrameworkRequest,
    CustomResponseOutputItemAddedEvent,
    CustomResponseOutputItemDoneEvent,
    ExecutorActionItem,
    InputTokensDetails,
    OpenAIResponse,
    OutputTokensDetails,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionResultComplete,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTraceEventComplete,
    ResponseUsage,
    ResponseWorkflowEventComplete,
)

logger = logging.getLogger(__name__)

# すべての可能なイベントタイプの型エイリアス
EventType = Union[
    ResponseStreamEvent,
    ResponseWorkflowEventComplete,
    ResponseOutputItemAddedEvent,
    ResponseTraceEventComplete,
]


def _serialize_content_recursive(value: Any) -> Any:
    """Agent FrameworkのContentオブジェクトを再帰的にJSON互換の値にシリアライズします。

    これはjson.dumps()で直接シリアライズできないネストされたContentオブジェクト（例えばFunctionResultContent.result内のTextContent）を処理します。

    Args:
        value: シリアライズする値（Contentオブジェクト、辞書、リスト、プリミティブなど）

    Returns:
        すべてのContentオブジェクトが辞書やプリミティブに変換されたJSONシリアライズ可能なバージョン

    """
    # Noneおよび基本的なJSONシリアライズ可能な型を処理する
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # SerializationMixinかどうかをチェックする（すべてのContentタイプを含む）
    # Contentオブジェクトはto_dict()メソッドを持つ必要がある
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict", None)):
        try:
            return value.to_dict()
        except Exception as e:
            # to_dict()が失敗した場合は他の方法にフォールスルーする
            logger.debug(f"Failed to serialize with to_dict(): {e}")

    # 辞書を処理する - 値を再帰的に処理する
    if isinstance(value, dict):
        return {key: _serialize_content_recursive(val) for key, val in value.items()}

    # リストおよびタプルを処理する - 要素を再帰的に処理する
    if isinstance(value, (list, tuple)):
        serialized = [_serialize_content_recursive(item) for item in value]
        # テキストContentを含む単一アイテムリストの場合はテキストだけを抽出する これはMCPケースでresult =
        # [TextContent(text="Hello")]のとき 出力を"Hello"にしたい場合に対応し、 出力が'[{"type": "text",
        # "text": "Hello"}]'にならないようにするための処理です。
        if len(serialized) == 1 and isinstance(serialized[0], dict) and serialized[0].get("type") == "text":
            return serialized[0].get("text", "")
        return serialized

    # model_dump()を持つ他のオブジェクトに対してはそれを試す
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump", None)):
        try:
            return value.model_dump()
        except Exception as e:
            logger.debug(f"Failed to serialize with model_dump(): {e}")

    # そのまま返してjson.dumpsに任せる（非シリアライズ可能な型の場合はTypeErrorが発生する可能性あり）
    return value


class MessageMapper:
    """Agent Frameworkのメッセージ/レスポンスをOpenAI形式にマッピングする。"""

    def __init__(self, max_contexts: int = 1000) -> None:
        """Agent Frameworkのメッセージmapperを初期化します。

        Args:
            max_contexts: メモリに保持する最大コンテキスト数（デフォルト: 1000）

        """
        self.sequence_counter = 0
        self._conversion_contexts: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._max_contexts = max_contexts

        # 最終的なResponse.usage（OpenAI標準）のためにリクエストごとの使用量を追跡する
        self._usage_accumulator: dict[str, dict[str, int]] = {}

        # 12種類すべてのAgent Frameworkコンテンツタイプのコンテンツタイプmapperを登録する
        self.content_mappers = {
            "TextContent": self._map_text_content,
            "TextReasoningContent": self._map_reasoning_content,
            "FunctionCallContent": self._map_function_call_content,
            "FunctionResultContent": self._map_function_result_content,
            "ErrorContent": self._map_error_content,
            "UsageContent": self._map_usage_content,
            "DataContent": self._map_data_content,
            "UriContent": self._map_uri_content,
            "HostedFileContent": self._map_hosted_file_content,
            "HostedVectorStoreContent": self._map_hosted_vector_store_content,
            "FunctionApprovalRequestContent": self._map_approval_request_content,
            "FunctionApprovalResponseContent": self._map_approval_response_content,
        }

    async def convert_event(self, raw_event: Any, request: AgentFrameworkRequest) -> Sequence[Any]:
        """単一のAgent FrameworkイベントをOpenAIイベントに変換します。

        Args:
            raw_event: Agent Frameworkイベント（AgentRunResponseUpdate、WorkflowEventなど）
            request: コンテキスト用の元のリクエスト

        Returns:
            OpenAIレスポンスストリームイベントのリスト

        """
        context = self._get_or_create_context(request)

        # エラーイベントを処理する
        if isinstance(raw_event, dict) and raw_event.get("type") == "error":
            return [await self._create_error_event(raw_event.get("message", "Unknown error"), context)]

        # トレースコレクターからのResponseTraceEventオブジェクトを処理する
        from .models import ResponseTraceEvent

        if isinstance(raw_event, ResponseTraceEvent):
            return [
                ResponseTraceEventComplete(
                    type="response.trace.complete",
                    data=raw_event.data,
                    item_id=context["item_id"],
                    sequence_number=self._next_sequence(context),
                )
            ]

        # Agentのライフサイクルイベントを最初に処理する
        from .models._openai_custom import AgentCompletedEvent, AgentFailedEvent, AgentStartedEvent

        if isinstance(raw_event, (AgentStartedEvent, AgentCompletedEvent, AgentFailedEvent)):
            return await self._convert_agent_lifecycle_event(raw_event, context)

        # 適切なisinstanceチェックのためにAgent Frameworkの型をインポートする
        try:
            from agent_framework import AgentRunResponse, AgentRunResponseUpdate, WorkflowEvent
            from agent_framework._workflows._events import AgentRunUpdateEvent

            # AgentRunUpdateEventを処理する - AgentRunResponseUpdateをラップするworkflowイベント
            # これは一般的なWorkflowEventチェックの前に確認する必要がある
            if isinstance(raw_event, AgentRunUpdateEvent):
                # イベントのdata属性からAgentRunResponseUpdateを抽出する
                if raw_event.data and isinstance(raw_event.data, AgentRunResponseUpdate):
                    return await self._convert_agent_update(raw_event.data, context)
                # データがなければ一般的なworkflowイベントとして扱う
                return await self._convert_workflow_event(raw_event, context)

            # 完全なagentレスポンス（AgentRunResponse）を処理する - 非ストリーミングagent実行用
            if isinstance(raw_event, AgentRunResponse):
                return await self._convert_agent_response(raw_event, context)

            # agentの更新（AgentRunResponseUpdate）を処理する - 直接agent実行用
            if isinstance(raw_event, AgentRunResponseUpdate):
                return await self._convert_agent_update(raw_event, context)

            # workflowイベント（WorkflowEventを継承する任意のクラス）を処理する
            if isinstance(raw_event, WorkflowEvent):
                return await self._convert_workflow_event(raw_event, context)

        except ImportError as e:
            logger.warning(f"Could not import Agent Framework types: {e}")
            # 属性ベースの検出にフォールバックする
            if hasattr(raw_event, "contents"):
                return await self._convert_agent_update(raw_event, context)
            if hasattr(raw_event, "__class__") and "Event" in raw_event.__class__.__name__:
                return await self._convert_workflow_event(raw_event, context)

        # 不明なイベントタイプ
        return [await self._create_unknown_event(raw_event, context)]

    async def aggregate_to_response(self, events: Sequence[Any], request: AgentFrameworkRequest) -> OpenAIResponse:
        """ストリーミングイベントを最終的なOpenAIレスポンスに集約する。

        Args:
            events: OpenAIストリームイベントのリスト
            request: コンテキスト用の元のリクエスト

        Returns:
            最終的に集約されたOpenAIレスポンス

        """
        try:
            # イベントからテキストコンテンツを抽出する
            content_parts = []

            for event in events:
                # ResponseTextDeltaEventからデルタテキストを抽出する
                if hasattr(event, "delta") and hasattr(event, "type") and event.type == "response.output_text.delta":
                    content_parts.append(event.delta)

            # コンテンツを結合する
            full_content = "".join(content_parts)

            # 適切なOpenAIレスポンスを作成する
            response_output_text = ResponseOutputText(type="output_text", text=full_content, annotations=[])

            response_output_message = ResponseOutputMessage(
                type="message",
                role="assistant",
                content=[response_output_text],
                id=f"msg_{uuid.uuid4().hex[:8]}",
                status="completed",
            )

            # アキュムレータから使用量を取得する（OpenAI標準）
            request_id = str(id(request))
            usage_data = self._usage_accumulator.get(request_id)

            if usage_data:
                usage = ResponseUsage(
                    input_tokens=usage_data["input_tokens"],
                    output_tokens=usage_data["output_tokens"],
                    total_tokens=usage_data["total_tokens"],
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                )
                # アキュムレータをクリーンアップする
                del self._usage_accumulator[request_id]
            else:
                # フォールバック：使用量が追跡されていない場合に推定する
                input_token_count = len(str(request.input)) // 4 if request.input else 0
                output_token_count = len(full_content) // 4
                usage = ResponseUsage(
                    input_tokens=input_token_count,
                    output_tokens=output_token_count,
                    total_tokens=input_token_count + output_token_count,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                )

            return OpenAIResponse(
                id=f"resp_{uuid.uuid4().hex[:12]}",
                object="response",
                created_at=datetime.now().timestamp(),
                model=request.model,
                output=[response_output_message],
                usage=usage,
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )

        except Exception as e:
            logger.exception(f"Error aggregating response: {e}")
            return await self._create_error_response(str(e), request)
        finally:
            # 集約後にコンテキストを削除してメモリリークを防止する これはストリーミングが正常に完了した一般的なケースを処理するためのものです
            request_key = id(request)
            if self._conversion_contexts.pop(request_key, None):
                logger.debug(f"Cleaned up context for request {request_key} after aggregation")

    def _get_or_create_context(self, request: AgentFrameworkRequest) -> dict[str, Any]:
        """このリクエストの変換コンテキストを取得または作成します。

        最大コンテキスト数に達した場合はLRU削除を使用して無制限のメモリ増加を防ぎます。

        Args:
            request: コンテキストを取得するリクエスト

        Returns:
            変換コンテキストの辞書

        """
        request_key = id(request)

        if request_key not in self._conversion_contexts:
            # 容量に達した場合、最も古いコンテキストを削除する（LRU削除）
            if len(self._conversion_contexts) >= self._max_contexts:
                evicted_key, _ = self._conversion_contexts.popitem(last=False)
                logger.debug(f"Evicted oldest context (key={evicted_key}) - at max capacity ({self._max_contexts})")

            self._conversion_contexts[request_key] = {
                "sequence_counter": 0,
                "item_id": f"msg_{uuid.uuid4().hex[:8]}",
                "content_index": 0,
                "output_index": 0,
                "request_id": str(request_key),  # For usage accumulation
                "request": request,  # Store the request for model name access
                # アクティブな関数呼び出しを追跡する: {call_id: {name, item_id, args_chunks}}
                "active_function_calls": {},
            }
        else:
            # 末尾に移動する（LRUのために最近使用されたことを示す）
            self._conversion_contexts.move_to_end(request_key)

        return self._conversion_contexts[request_key]

    def _next_sequence(self, context: dict[str, Any]) -> int:
        """イベントの次のシーケンス番号を取得する。

        Args:
            context: Conversion context

        Returns:
            次のシーケンス番号

        """
        context["sequence_counter"] += 1
        return int(context["sequence_counter"])

    async def _convert_agent_update(self, update: Any, context: dict[str, Any]) -> Sequence[Any]:
        """Agentのテキスト更新を適切なcontent partイベントに変換する。

        Args:
            update: Agent run response update
            context: Conversion context

        Returns:
            OpenAIレスポンスストリームイベントのリスト

        """
        events: list[Any] = []

        try:
            # 異なる更新タイプを処理する
            if not hasattr(update, "contents") or not update.contents:
                return events

            # テキストコンテンツをストリーミングしているか確認する
            has_text_content = any(content.__class__.__name__ == "TextContent" for content in update.contents)

            # テキストコンテンツがあり、まだメッセージを作成していない場合は、メッセージを作成する
            if has_text_content and "current_message_id" not in context:
                message_id = f"msg_{uuid4().hex[:8]}"
                context["current_message_id"] = message_id
                context["output_index"] = context.get("output_index", -1) + 1

                # メッセージ出力アイテムを追加する
                events.append(
                    ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        output_index=context["output_index"],
                        sequence_number=self._next_sequence(context),
                        item=ResponseOutputMessage(
                            type="message", id=message_id, role="assistant", content=[], status="in_progress"
                        ),
                    )
                )

                # テキスト用のcontent partを追加する
                context["content_index"] = 0
                events.append(
                    ResponseContentPartAddedEvent(
                        type="response.content_part.added",
                        output_index=context["output_index"],
                        content_index=context["content_index"],
                        item_id=message_id,
                        sequence_number=self._next_sequence(context),
                        part=ResponseOutputText(type="output_text", text="", annotations=[]),
                    )
                )

            # 各contentアイテムを処理する
            for content in update.contents:
                content_type = content.__class__.__name__

                # TextContentに対して適切なdeltaイベントを使用する特別な処理
                if content_type == "TextContent" and "current_message_id" in context:
                    # 適切なdeltaイベントを介してテキストコンテンツをストリーミングする
                    events.append(
                        ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            output_index=context["output_index"],
                            content_index=context.get("content_index", 0),
                            item_id=context["current_message_id"],
                            delta=content.text,
                            logprobs=[],  # We don't have logprobs from Agent Framework
                            sequence_number=self._next_sequence(context),
                        )
                    )
                elif content_type in self.content_mappers:
                    # 他のcontentタイプには既存のマッパーを使用する
                    mapped_events = await self.content_mappers[content_type](content, context)
                    if mapped_events is not None:  # Handle None returns (e.g., UsageContent)
                        if isinstance(mapped_events, list):
                            events.extend(mapped_events)
                        else:
                            events.append(mapped_events)
                else:
                    # 不明なcontentタイプに対する優雅なフォールバック
                    events.append(await self._create_unknown_content_event(content, context))

                # 同じパート内のテキストdeltaではcontent_indexを増やさない
                if content_type != "TextContent":
                    context["content_index"] = context.get("content_index", 0) + 1

        except Exception as e:
            logger.warning(f"Error converting agent update: {e}")
            events.append(await self._create_error_event(str(e), context))

        return events

    async def _convert_agent_response(self, response: Any, context: dict[str, Any]) -> Sequence[Any]:
        """完全なAgentRunResponseをOpenAIイベントに変換する。

        これは、agent.run()がストリーミングではなく完全なAgentRunResponseを返す非ストリーミングのagent実行を処理する。

        Args:
            response: Agent run response (AgentRunResponse)
            context: Conversion context

        Returns:
            OpenAIレスポンスストリームイベントのリスト

        """
        events: list[Any] = []

        try:
            # レスポンスからすべてのメッセージを抽出する
            messages = getattr(response, "messages", [])

            # 各メッセージのコンテンツをストリーミングイベントに変換する
            for message in messages:
                if hasattr(message, "contents") and message.contents:
                    for content in message.contents:
                        content_type = content.__class__.__name__

                        if content_type in self.content_mappers:
                            mapped_events = await self.content_mappers[content_type](content, context)
                            if mapped_events is not None:  # Handle None returns (e.g., UsageContent)
                                if isinstance(mapped_events, list):
                                    events.extend(mapped_events)
                                else:
                                    events.append(mapped_events)
                        else:
                            # 不明なcontentタイプに対する優雅なフォールバック
                            events.append(await self._create_unknown_content_event(content, context))

                        context["content_index"] += 1

            # 使用情報があれば追加する
            usage_details = getattr(response, "usage_details", None)
            if usage_details:
                from agent_framework import UsageContent

                usage_content = UsageContent(details=usage_details)
                await self._map_usage_content(usage_content, context)
                # 注意: _map_usage_contentはNoneを返す - 最終Response.usageのために使用を蓄積する

        except Exception as e:
            logger.warning(f"Error converting agent response: {e}")
            events.append(await self._create_error_event(str(e), context))

        return events

    async def _convert_agent_lifecycle_event(self, event: Any, context: dict[str, Any]) -> Sequence[Any]:
        """AgentのライフサイクルイベントをOpenAIレスポンスイベントに変換する。

        Args:
            event: AgentStartedEvent, AgentCompletedEvent, または AgentFailedEvent
            context: Conversion context

        Returns:
            OpenAIレスポンスストリームイベントのリスト

        """
        from .models._openai_custom import AgentCompletedEvent, AgentFailedEvent, AgentStartedEvent

        try:
            # コンテキストからモデル名を取得する（agent名）
            model_name = context.get("request", {}).model if context.get("request") else "agent"

            if isinstance(event, AgentStartedEvent):
                execution_id = f"agent_{uuid4().hex[:12]}"
                context["execution_id"] = execution_id

                # Responseオブジェクトを作成する
                response_obj = Response(
                    id=f"resp_{execution_id}",
                    object="response",
                    created_at=float(time.time()),
                    model=model_name,
                    output=[],
                    status="in_progress",
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                )

                # createdとin_progressの両方のイベントを発行する
                return [
                    ResponseCreatedEvent(
                        type="response.created", sequence_number=self._next_sequence(context), response=response_obj
                    ),
                    ResponseInProgressEvent(
                        type="response.in_progress", sequence_number=self._next_sequence(context), response=response_obj
                    ),
                ]

            if isinstance(event, AgentCompletedEvent):
                execution_id = context.get("execution_id", f"agent_{uuid4().hex[:12]}")

                response_obj = Response(
                    id=f"resp_{execution_id}",
                    object="response",
                    created_at=float(time.time()),
                    model=model_name,
                    output=[],
                    status="completed",
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                )

                return [
                    ResponseCompletedEvent(
                        type="response.completed", sequence_number=self._next_sequence(context), response=response_obj
                    )
                ]

            if isinstance(event, AgentFailedEvent):
                execution_id = context.get("execution_id", f"agent_{uuid4().hex[:12]}")

                # エラーオブジェクトを作成する
                response_error = ResponseError(
                    message=str(event.error) if event.error else "Unknown error", code="server_error"
                )

                response_obj = Response(
                    id=f"resp_{execution_id}",
                    object="response",
                    created_at=float(time.time()),
                    model=model_name,
                    output=[],
                    status="failed",
                    error=response_error,
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                )

                return [
                    ResponseFailedEvent(
                        type="response.failed", sequence_number=self._next_sequence(context), response=response_obj
                    )
                ]

            return []

        except Exception as e:
            logger.warning(f"Error converting agent lifecycle event: {e}")
            return [await self._create_error_event(str(e), context)]

    async def _convert_workflow_event(self, event: Any, context: dict[str, Any]) -> Sequence[Any]:
        """ワークフローイベントを標準のOpenAIイベントオブジェクトに変換する。

        Args:
            event: Workflow event
            context: Conversion context

        Returns:
            OpenAIレスポンスストリームイベントのリスト

        """
        try:
            event_class = event.__class__.__name__

            # レスポンスレベルのイベント - 適切なOpenAIオブジェクトを構築する
            if event_class == "WorkflowStartedEvent":
                workflow_id = getattr(event, "workflow_id", str(uuid4()))
                context["workflow_id"] = workflow_id

                # 適切な構築のためにResponseタイプをインポートする
                from openai.types.responses import Response

                # 適切なOpenAIイベントオブジェクトを返す
                events: list[Any] = []

                # モデル名を決定する - リクエストモデルを使用するか、デフォルトで"workflow"を使用する
                # リクエストモデルはagentの場合はagent名、workflowの場合はworkflow名になる
                model_name = context.get("request", {}).model if context.get("request") else "workflow"

                # 必要なすべてのフィールドを持つ完全なResponseオブジェクトを作成する
                response_obj = Response(
                    id=f"resp_{workflow_id}",
                    object="response",
                    created_at=float(time.time()),
                    model=model_name,  # Use the actual model/agent name
                    output=[],  # Empty output list initially
                    status="in_progress",
                    # 安全なデフォルト値を持つ必須フィールド
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                )

                # 最初にresponse.createdを発行する
                events.append(
                    ResponseCreatedEvent(
                        type="response.created", sequence_number=self._next_sequence(context), response=response_obj
                    )
                )

                # 次にresponse.in_progressを発行する（同じresponseオブジェクトを再利用）
                events.append(
                    ResponseInProgressEvent(
                        type="response.in_progress", sequence_number=self._next_sequence(context), response=response_obj
                    )
                )

                return events

            if event_class in ["WorkflowCompletedEvent", "WorkflowOutputEvent"]:
                workflow_id = context.get("workflow_id", str(uuid4()))

                # 適切な構築のためにResponseタイプをインポートする
                from openai.types.responses import Response

                # コンテキストからモデル名を取得する
                model_name = context.get("request", {}).model if context.get("request") else "workflow"

                # 完了状態のための完全なResponseオブジェクトを作成する
                response_obj = Response(
                    id=f"resp_{workflow_id}",
                    object="response",
                    created_at=float(time.time()),
                    model=model_name,
                    output=[],  # Output should be populated by this point from text streaming
                    status="completed",
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                )

                return [
                    ResponseCompletedEvent(
                        type="response.completed", sequence_number=self._next_sequence(context), response=response_obj
                    )
                ]

            if event_class == "WorkflowFailedEvent":
                workflow_id = context.get("workflow_id", str(uuid4()))
                error_info = getattr(event, "error", None)

                # ResponseとResponseErrorタイプをインポートする
                from openai.types.responses import Response, ResponseError

                # コンテキストからモデル名を取得する
                model_name = context.get("request", {}).model if context.get("request") else "workflow"

                # エラーオブジェクトを作成する
                error_message = str(error_info) if error_info else "Unknown error"

                # ResponseErrorオブジェクトを作成する（コードは許可された値のいずれかでなければならない）
                response_error = ResponseError(
                    message=error_message,
                    code="server_error",  # Use generic server_error code for workflow failures
                )

                # 失敗状態のための完全なResponseオブジェクトを作成する
                response_obj = Response(
                    id=f"resp_{workflow_id}",
                    object="response",
                    created_at=float(time.time()),
                    model=model_name,
                    output=[],
                    status="failed",
                    error=response_error,
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                )

                return [
                    ResponseFailedEvent(
                        type="response.failed", sequence_number=self._next_sequence(context), response=response_obj
                    )
                ]

            # Executorレベルのイベント（出力アイテム）
            if event_class == "ExecutorInvokedEvent":
                executor_id = getattr(event, "executor_id", "unknown")
                item_id = f"exec_{executor_id}_{uuid4().hex[:8]}"
                context[f"exec_item_{executor_id}"] = item_id
                context["output_index"] = context.get("output_index", -1) + 1

                # 適切なタイプでExecutorActionItemを作成する
                executor_item = ExecutorActionItem(
                    type="executor_action",
                    id=item_id,
                    executor_id=executor_id,
                    status="in_progress",
                    metadata=getattr(event, "metadata", {}),
                )

                # ExecutorActionItemを受け入れるカスタムイベントタイプを使用する
                return [
                    CustomResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        output_index=context["output_index"],
                        sequence_number=self._next_sequence(context),
                        item=executor_item,
                    )
                ]

            if event_class == "ExecutorCompletedEvent":
                executor_id = getattr(event, "executor_id", "unknown")
                item_id = context.get(f"exec_item_{executor_id}", f"exec_{executor_id}_unknown")

                # 完了ステータスでExecutorActionItemを作成する
                # ExecutorCompletedEventは'result'ではなく'data'フィールドを使用する
                executor_item = ExecutorActionItem(
                    type="executor_action",
                    id=item_id,
                    executor_id=executor_id,
                    status="completed",
                    result=getattr(event, "data", None),
                )

                # カスタムイベントタイプを使用する
                return [
                    CustomResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        output_index=context.get("output_index", 0),
                        sequence_number=self._next_sequence(context),
                        item=executor_item,
                    )
                ]

            if event_class == "ExecutorFailedEvent":
                executor_id = getattr(event, "executor_id", "unknown")
                item_id = context.get(f"exec_item_{executor_id}", f"exec_{executor_id}_unknown")
                error_info = getattr(event, "error", None)

                # 失敗ステータスでExecutorActionItemを作成する
                executor_item = ExecutorActionItem(
                    type="executor_action",
                    id=item_id,
                    executor_id=executor_id,
                    status="failed",
                    error={"message": str(error_info)} if error_info else None,
                )

                # カスタムイベントタイプを使用する
                return [
                    CustomResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        output_index=context.get("output_index", 0),
                        sequence_number=self._next_sequence(context),
                        item=executor_item,
                    )
                ]

            # 情報的なワークフローイベント（ステータス、警告、エラー）を処理する
            if event_class in ["WorkflowStatusEvent", "WorkflowWarningEvent", "WorkflowErrorEvent", "RequestInfoEvent"]:
                # これらはOpenAIのライフサイクルイベントにマップされない情報イベントである デバッグの可視性のためにトレースイベントに変換する
                event_data: dict[str, Any] = {}

                # イベントタイプに基づいて関連データを抽出する
                if event_class == "WorkflowStatusEvent":
                    event_data["state"] = str(getattr(event, "state", "unknown"))
                elif event_class == "WorkflowWarningEvent":
                    event_data["message"] = str(getattr(event, "message", ""))
                elif event_class == "WorkflowErrorEvent":
                    event_data["message"] = str(getattr(event, "message", ""))
                    event_data["error"] = str(getattr(event, "error", ""))
                elif event_class == "RequestInfoEvent":
                    request_info = getattr(event, "data", {})
                    event_data["request_info"] = request_info if isinstance(request_info, dict) else str(request_info)

                # デバッグ用のトレースイベントを作成する
                trace_event = ResponseTraceEventComplete(
                    type="response.trace.complete",
                    data={
                        "trace_type": "workflow_info",
                        "event_type": event_class,
                        "data": event_data,
                        "timestamp": datetime.now().isoformat(),
                    },
                    span_id=f"workflow_info_{uuid4().hex[:8]}",
                    item_id=context["item_id"],
                    output_index=context.get("output_index", 0),
                    sequence_number=self._next_sequence(context),
                )

                return [trace_event]

            # 不明/レガシーイベントの場合でも、後方互換性のためにworkflowイベントとして発行する
            # イベントデータを取得し、SerializationMixinならシリアライズする
            raw_event_data = getattr(event, "data", None)
            serialized_event_data: dict[str, Any] | str | None = raw_event_data
            if raw_event_data is not None and hasattr(raw_event_data, "to_dict"):
                # SerializationMixinオブジェクト - JSONシリアライズのためにdictに変換する
                try:
                    serialized_event_data = raw_event_data.to_dict()
                except Exception as e:
                    logger.debug(f"Failed to serialize event data with to_dict(): {e}")
                    serialized_event_data = str(raw_event_data)

            # 構造化されたworkflowイベントを作成する（後方互換性のために保持）
            workflow_event = ResponseWorkflowEventComplete(
                type="response.workflow_event.complete",
                data={
                    "event_type": event.__class__.__name__,
                    "data": serialized_event_data,
                    "executor_id": getattr(event, "executor_id", None),
                    "timestamp": datetime.now().isoformat(),
                },
                executor_id=getattr(event, "executor_id", None),
                item_id=context["item_id"],
                output_index=context["output_index"],
                sequence_number=self._next_sequence(context),
            )

            logger.debug(f"Unhandled workflow event type: {event_class}, emitting as legacy workflow event")
            return [workflow_event]

        except Exception as e:
            logger.warning(f"Error converting workflow event: {e}")
            return [await self._create_error_event(str(e), context)]

    # Contentタイプマッパー - 包括的なマッピング計画を実装する

    async def _map_text_content(self, content: Any, context: dict[str, Any]) -> ResponseTextDeltaEvent:
        """TextContentをResponseTextDeltaEventにマップする。"""
        return self._create_text_delta_event(content.text, context)

    async def _map_reasoning_content(self, content: Any, context: dict[str, Any]) -> ResponseReasoningTextDeltaEvent:
        """TextReasoningContentをResponseReasoningTextDeltaEventにマップする。"""
        return ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            delta=content.text,
            item_id=context["item_id"],
            output_index=context["output_index"],
            content_index=context["content_index"],
            sequence_number=self._next_sequence(context),
        )

    async def _map_function_call_content(
        self, content: Any, context: dict[str, Any]
    ) -> list[ResponseFunctionCallArgumentsDeltaEvent | ResponseOutputItemAddedEvent]:
        """FunctionCallContentをOpenAIイベントにマップする（Responses API仕様に従う）。

        Agent FrameworkはFunctionCallContentを2つのパターンで発行する:
        1. 最初のイベント: call_id + name + 空または引数なし
        2. 以降のイベント: 空のcall_id/name + 引数チャンク

        発行するのは:
        1. response.output_item.added（完全なメタデータ付き）を最初のイベントで
        2. response.function_call_arguments.delta（item_idを参照）をチャンクで

        """
        events: list[ResponseFunctionCallArgumentsDeltaEvent | ResponseOutputItemAddedEvent] = []

        # ケース1: 新しい関数呼び出し（call_idとnameがある） これは関数呼び出しを確立する最初のイベントである
        if content.call_id and content.name:
            # call_idをitem_idとして使用する（単純で、call_idが呼び出しを一意に識別するため）
            item_id = content.call_id

            # 後の引数deltaのためにこの関数呼び出しを追跡する
            context["active_function_calls"][content.call_id] = {
                "item_id": item_id,
                "name": content.name,
                "arguments_chunks": [],
            }

            logger.debug(f"New function call: {content.name} (call_id={content.call_id})")

            # OpenAI仕様に従いresponse.output_item.addedイベントを発行する
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=ResponseFunctionToolCall(
                        id=content.call_id,  # Use call_id as the item id
                        call_id=content.call_id,
                        name=content.name,
                        arguments="",  # Empty initially, will be filled by deltas
                        type="function_call",
                        status="in_progress",
                    ),
                    output_index=context["output_index"],
                    sequence_number=self._next_sequence(context),
                )
            )

        # ケース2: 引数delta（contentに引数があり、call_id/nameがない可能性がある）
        if content.arguments:
            # これらの引数のためのアクティブな関数呼び出しを見つける
            active_call = self._get_active_function_call(content, context)

            if active_call:
                item_id = active_call["item_id"]

                # 引数がdictの場合は文字列に変換する（Agent Frameworkはどちらも送信する可能性がある）
                delta_str = content.arguments if isinstance(content.arguments, str) else json.dumps(content.arguments)

                # item_idを参照する引数deltaを発行する
                events.append(
                    ResponseFunctionCallArgumentsDeltaEvent(
                        type="response.function_call_arguments.delta",
                        delta=delta_str,
                        item_id=item_id,
                        output_index=context["output_index"],
                        sequence_number=self._next_sequence(context),
                    )
                )

                # デバッグのためにチャンクを追跡する
                active_call["arguments_chunks"].append(delta_str)
            else:
                logger.warning(f"Received function call arguments without active call: {content.arguments[:50]}...")

        return events

    def _get_active_function_call(self, content: Any, context: dict[str, Any]) -> dict[str, Any] | None:
        """このcontentのアクティブな関数呼び出しを見つける。

        call_idがあればそれを使用し、なければ最新の呼び出しにフォールバックする。
        これはAgent Frameworkがcall_idなしで引数チャンクを送信する場合に必要。

        Args:
            content: call_idを含む可能性のあるFunctionCallContent
            context: active_function_callsを持つConversion context

        Returns:
            アクティブな呼び出しの辞書またはNone

        """
        active_calls: dict[str, dict[str, Any]] = context["active_function_calls"]

        # contentにcall_idがあれば、それを使って正確な呼び出しを見つける
        if hasattr(content, "call_id") and content.call_id:
            result = active_calls.get(content.call_id)
            return result if result is not None else None

        # そうでなければ最新の呼び出し（最後に追加されたもの）を使う これはAgent
        # Frameworkが後続イベントでcall_idなしの引数チャンクを送信するケースを処理する
        if active_calls:
            return list(active_calls.values())[-1]

        return None

    async def _map_function_result_content(
        self, content: Any, context: dict[str, Any]
    ) -> ResponseFunctionResultComplete:
        """FunctionResultContentをDevUIカスタムイベントにマップする。

        DevUI拡張: OpenAI Responses APIは関数実行結果をストリーミングしない
        （OpenAIのモデルでは、アプリケーションが関数を実行し、APIは実行しない）。

        """
        # contentからcall_idを取得する
        call_id = getattr(content, "call_id", None)
        if not call_id:
            call_id = f"call_{uuid.uuid4().hex[:8]}"

        # 結果を抽出する
        result = getattr(content, "result", None)
        exception = getattr(content, "exception", None)

        # 結果を文字列に変換する。MCPツールからのネストされたContentオブジェクトも処理する
        if isinstance(result, str):
            output = result
        elif result is not None:
            # ネストされたContentオブジェクトを再帰的にシリアライズする（例: MCPツールから）
            serialized = _serialize_content_recursive(result)
            # まだ文字列でなければJSON文字列に変換する
            output = serialized if isinstance(serialized, str) else json.dumps(serialized)
        else:
            output = ""

        # 例外に基づいてステータスを決定する
        status = "incomplete" if exception else "completed"

        # item_idを生成する
        item_id = f"item_{uuid.uuid4().hex[:8]}"

        # DevUIカスタムイベントを返す
        return ResponseFunctionResultComplete(
            type="response.function_result.complete",
            call_id=call_id,
            output=output,
            status=status,
            item_id=item_id,
            output_index=context["output_index"],
            sequence_number=self._next_sequence(context),
            timestamp=datetime.now().isoformat(),
        )

    async def _map_error_content(self, content: Any, context: dict[str, Any]) -> ResponseErrorEvent:
        """ErrorContentをResponseErrorEventにマップする。"""
        return ResponseErrorEvent(
            type="error",
            message=getattr(content, "message", "Unknown error"),
            code=getattr(content, "error_code", None),
            param=None,
            sequence_number=self._next_sequence(context),
        )

    async def _map_usage_content(self, content: Any, context: dict[str, Any]) -> None:
        """最終Response.usageフィールドのために使用データを蓄積する。

        OpenAIは使用イベントをストリーミングしない。使用情報は最終Responseにのみ現れる。
        このメソッドはリクエストごとに使用データを蓄積し、後でResponse.usageに含める。

        Returns:
            None - イベントは発行しない（使用情報は最終Response.usageに入る）

        """
        # UsageContent.details（UsageDetailsオブジェクト）から使用情報を抽出する
        details = getattr(content, "details", None)
        total_tokens = getattr(details, "total_token_count", 0) or 0
        prompt_tokens = getattr(details, "input_token_count", 0) or 0
        completion_tokens = getattr(details, "output_token_count", 0) or 0

        # 最終Response.usageのために蓄積する
        request_id = context.get("request_id", "default")
        if request_id not in self._usage_accumulator:
            self._usage_accumulator[request_id] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        self._usage_accumulator[request_id]["input_tokens"] += prompt_tokens
        self._usage_accumulator[request_id]["output_tokens"] += completion_tokens
        self._usage_accumulator[request_id]["total_tokens"] += total_tokens

        logger.debug(f"Accumulated usage for {request_id}: {self._usage_accumulator[request_id]}")

        # イベントは返さない - 使用情報は最終Responseにのみ入る
        return

    async def _map_data_content(self, content: Any, context: dict[str, Any]) -> ResponseTraceEventComplete:
        """DataContentを構造化されたトレースイベントにマップする。"""
        return ResponseTraceEventComplete(
            type="response.trace.complete",
            data={
                "content_type": "data",
                "data": getattr(content, "data", None),
                "mime_type": getattr(content, "mime_type", "application/octet-stream"),
                "size_bytes": len(str(getattr(content, "data", ""))) if getattr(content, "data", None) else 0,
                "timestamp": datetime.now().isoformat(),
            },
            item_id=context["item_id"],
            output_index=context["output_index"],
            sequence_number=self._next_sequence(context),
        )

    async def _map_uri_content(self, content: Any, context: dict[str, Any]) -> ResponseTraceEventComplete:
        """UriContentを構造化されたトレースイベントにマップする。"""
        return ResponseTraceEventComplete(
            type="response.trace.complete",
            data={
                "content_type": "uri",
                "uri": getattr(content, "uri", ""),
                "mime_type": getattr(content, "mime_type", "text/plain"),
                "timestamp": datetime.now().isoformat(),
            },
            item_id=context["item_id"],
            output_index=context["output_index"],
            sequence_number=self._next_sequence(context),
        )

    async def _map_hosted_file_content(self, content: Any, context: dict[str, Any]) -> ResponseTraceEventComplete:
        """HostedFileContentを構造化されたトレースイベントにマップする。"""
        return ResponseTraceEventComplete(
            type="response.trace.complete",
            data={
                "content_type": "hosted_file",
                "file_id": getattr(content, "file_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
            },
            item_id=context["item_id"],
            output_index=context["output_index"],
            sequence_number=self._next_sequence(context),
        )

    async def _map_hosted_vector_store_content(
        self, content: Any, context: dict[str, Any]
    ) -> ResponseTraceEventComplete:
        """HostedVectorStoreContentを構造化されたトレースイベントにマップする。"""
        return ResponseTraceEventComplete(
            type="response.trace.complete",
            data={
                "content_type": "hosted_vector_store",
                "vector_store_id": getattr(content, "vector_store_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
            },
            item_id=context["item_id"],
            output_index=context["output_index"],
            sequence_number=self._next_sequence(context),
        )

    async def _map_approval_request_content(self, content: Any, context: dict[str, Any]) -> dict[str, Any]:
        """FunctionApprovalRequestContentをカスタムイベントにマップする。"""
        # 引数を解析して常にdictであることを保証し、JSON文字列ではないようにする
        # これはフロントエンドがJSON.stringify()を呼び出した際の二重エスケープを防止します
        arguments: dict[str, Any] = {}
        if hasattr(content, "function_call"):
            if hasattr(content.function_call, "parse_arguments"):
                # 文字列の引数をdictに変換するためにparse_arguments()を使用する
                arguments = content.function_call.parse_arguments() or {}
            else:
                # parse_argumentsが存在しない場合は直接アクセスにフォールバックする
                arguments = getattr(content.function_call, "arguments", {})

        return {
            "type": "response.function_approval.requested",
            "request_id": getattr(content, "id", "unknown"),
            "function_call": {
                "id": getattr(content.function_call, "call_id", "") if hasattr(content, "function_call") else "",
                "name": getattr(content.function_call, "name", "") if hasattr(content, "function_call") else "",
                "arguments": arguments,
            },
            "item_id": context["item_id"],
            "output_index": context["output_index"],
            "sequence_number": self._next_sequence(context),
        }

    async def _map_approval_response_content(self, content: Any, context: dict[str, Any]) -> dict[str, Any]:
        """FunctionApprovalResponseContentをカスタムイベントにマッピングする。"""
        return {
            "type": "response.function_approval.responded",
            "request_id": getattr(content, "request_id", "unknown"),
            "approved": getattr(content, "approved", False),
            "item_id": context["item_id"],
            "output_index": context["output_index"],
            "sequence_number": self._next_sequence(context),
        }

    # ヘルパーメソッド

    def _create_text_delta_event(self, text: str, context: dict[str, Any]) -> ResponseTextDeltaEvent:
        """ResponseTextDeltaEventを作成する。"""
        return ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=context["item_id"],
            output_index=context["output_index"],
            content_index=context["content_index"],
            delta=text,
            sequence_number=self._next_sequence(context),
            logprobs=[],
        )

    async def _create_error_event(self, message: str, context: dict[str, Any]) -> ResponseErrorEvent:
        """ResponseErrorEventを作成する。"""
        return ResponseErrorEvent(
            type="error", message=message, code=None, param=None, sequence_number=self._next_sequence(context)
        )

    async def _create_unknown_event(self, event_data: Any, context: dict[str, Any]) -> ResponseStreamEvent:
        """不明なイベントタイプのためのイベントを作成する。"""
        text = f"Unknown event: {event_data!s}\n"
        return self._create_text_delta_event(text, context)

    async def _create_unknown_content_event(self, content: Any, context: dict[str, Any]) -> ResponseStreamEvent:
        """不明なコンテンツタイプのためのイベントを作成する。"""
        content_type = content.__class__.__name__
        text = f"Warning: Unknown content type: {content_type}\n"
        return self._create_text_delta_event(text, context)

    async def _create_error_response(self, error_message: str, request: AgentFrameworkRequest) -> OpenAIResponse:
        """エラー応答を作成する。"""
        error_text = f"Error: {error_message}"

        response_output_text = ResponseOutputText(type="output_text", text=error_text, annotations=[])

        response_output_message = ResponseOutputMessage(
            type="message",
            role="assistant",
            content=[response_output_text],
            id=f"msg_{uuid.uuid4().hex[:8]}",
            status="completed",
        )

        usage = ResponseUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

        return OpenAIResponse(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            object="response",
            created_at=datetime.now().timestamp(),
            model=request.model,
            output=[response_output_message],
            usage=usage,
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
