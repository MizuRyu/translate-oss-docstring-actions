# Copyright (c) Microsoft. All rights reserved.

"""Agent Frameworkのエグゼキューター実装。"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from agent_framework import AgentProtocol

from ._conversations import ConversationStore, InMemoryConversationStore
from ._discovery import EntityDiscovery
from ._mapper import MessageMapper
from ._tracing import capture_traces
from .models import AgentFrameworkRequest, OpenAIResponse
from .models._discovery_models import EntityInfo

logger = logging.getLogger(__name__)


class EntityNotFoundError(Exception):
    """エンティティが見つからない場合に発生する例外。"""

    pass


class AgentFrameworkExecutor:
    """Agent Frameworkエンティティ（エージェントとワークフロー）のエグゼキューター。"""

    def __init__(
        self,
        entity_discovery: EntityDiscovery,
        message_mapper: MessageMapper,
        conversation_store: ConversationStore | None = None,
    ):
        """Agent Frameworkエグゼキューターを初期化する。

        Args:
            entity_discovery: エンティティディスカバリーのインスタンス
            message_mapper: メッセージマッパーのインスタンス
            conversation_store: Optionalな会話ストア（デフォルトはインメモリ）

        """
        self.entity_discovery = entity_discovery
        self.message_mapper = message_mapper
        self._setup_tracing_provider()
        self._setup_agent_framework_tracing()

        # 提供された会話ストアを使用するか、デフォルトでインメモリを使用する
        self.conversation_store = conversation_store or InMemoryConversationStore()

    def _setup_tracing_provider(self) -> None:
        """独自のTracerProviderをセットアップしてプロセッサを追加できるようにする。"""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            # まだプロバイダーが存在しない場合のみセットアップする
            if not hasattr(trace, "_TRACER_PROVIDER") or trace._TRACER_PROVIDER is None:
                resource = Resource.create({
                    "service.name": "agent-framework-server",
                    "service.version": "1.0.0",
                })
                provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(provider)
                logger.info("Set up TracerProvider for server tracing")
            else:
                logger.debug("TracerProvider already exists")

        except ImportError:
            logger.debug("OpenTelemetry not available")
        except Exception as e:
            logger.warning(f"Failed to setup TracerProvider: {e}")

    def _setup_agent_framework_tracing(self) -> None:
        """Agent Frameworkの組み込みトレーシングをセットアップする。"""
        # ENABLE_OTELが設定されている場合のみAgent Frameworkのトレーシングを構成する
        if os.environ.get("ENABLE_OTEL"):
            try:
                from agent_framework.observability import setup_observability

                setup_observability(enable_sensitive_data=True)
                logger.info("Enabled Agent Framework observability")
            except Exception as e:
                logger.warning(f"Failed to enable Agent Framework observability: {e}")
        else:
            logger.debug("ENABLE_OTEL not set, skipping observability setup")

    async def discover_entities(self) -> list[EntityInfo]:
        """利用可能なすべてのエンティティをディスカバーする。

        Returns:
            発見されたエンティティのリスト

        """
        return await self.entity_discovery.discover_entities()

    def get_entity_info(self, entity_id: str) -> EntityInfo:
        """エンティティ情報を取得する。

        Args:
            entity_id: エンティティ識別子

        Returns:
            エンティティ情報

        Raises:
            EntityNotFoundError: エンティティが見つからない場合

        """
        entity_info = self.entity_discovery.get_entity_info(entity_id)
        if entity_info is None:
            raise EntityNotFoundError(f"Entity '{entity_id}' not found")
        return entity_info

    async def execute_streaming(self, request: AgentFrameworkRequest) -> AsyncGenerator[Any, None]:
        """リクエストを実行し、OpenAI形式で結果をストリームする。

        Args:
            request: 実行するリクエスト

        Yields:
            OpenAIレスポンスのストリームイベント

        """
        try:
            entity_id = request.get_entity_id()
            if not entity_id:
                logger.error("No entity_id specified in request")
                return

            # エンティティが存在するか検証する
            if not self.entity_discovery.get_entity_info(entity_id):
                logger.error(f"Entity '{entity_id}' not found")
                return

            # エンティティを実行しイベントを変換する
            async for raw_event in self.execute_entity(entity_id, request):
                openai_events = await self.message_mapper.convert_event(raw_event, request)
                for event in openai_events:
                    yield event

        except Exception as e:
            logger.exception(f"Error in streaming execution: {e}")
            # ここでエラーイベントをyieldする可能性がある

    async def execute_sync(self, request: AgentFrameworkRequest) -> OpenAIResponse:
        """リクエストを同期的に実行し、完全なレスポンスを返す。

        Args:
            request: 実行するリクエスト

        Returns:
            最終的に集約されたOpenAIレスポンス

        """
        # すべてのストリーミングイベントを収集する
        events = [event async for event in self.execute_streaming(request)]

        # 最終レスポンスに集約する
        return await self.message_mapper.aggregate_to_response(events, request)

    async def execute_entity(self, entity_id: str, request: AgentFrameworkRequest) -> AsyncGenerator[Any, None]:
        """エンティティを実行し、生のAgent Frameworkイベントとトレースイベントをyieldする。

        Args:
            entity_id: 実行するエンティティのID
            request: 実行するリクエスト

        Yields:
            生のAgent Frameworkイベントとトレースイベント

        """
        try:
            # エンティティ情報を取得する
            entity_info = self.get_entity_info(entity_id)

            # 遅延読み込みをトリガーする（すでに読み込まれていればキャッシュから返す）
            entity_obj = await self.entity_discovery.load_entity(entity_id)

            if not entity_obj:
                raise EntityNotFoundError(f"Entity object for '{entity_id}' not found")

            logger.info(f"Executing {entity_info.type}: {entity_id}")

            # トレースコンテキストのためにリクエストからsession_idを抽出する
            session_id = getattr(request.extra_body, "session_id", None) if request.extra_body else None

            # 簡易化されたトレースキャプチャを使用する
            with capture_traces(session_id=session_id, entity_id=entity_id) as trace_collector:
                if entity_info.type == "agent":
                    async for event in self._execute_agent(entity_obj, request, trace_collector):
                        yield event
                elif entity_info.type == "workflow":
                    async for event in self._execute_workflow(entity_obj, request, trace_collector):
                        yield event
                else:
                    raise ValueError(f"Unsupported entity type: {entity_info.type}")

                # 実行完了後に残っているトレースイベントをyieldする
                for trace_event in trace_collector.get_pending_events():
                    yield trace_event

        except Exception as e:
            logger.exception(f"Error executing entity {entity_id}: {e}")
            # エラーイベントをyieldする
            yield {"type": "error", "message": str(e), "entity_id": entity_id}

    async def _execute_agent(
        self, agent: AgentProtocol, request: AgentFrameworkRequest, trace_collector: Any
    ) -> AsyncGenerator[Any, None]:
        """トレース収集とオプションのThreadサポート付きでAgent Frameworkエージェントを実行する。

        Args:
            agent: 実行するエージェントオブジェクト
            request: 実行するリクエスト
            trace_collector: イベントを取得するトレースコレクター

        Yields:
            エージェントの更新イベントとトレースイベント

        """
        try:
            # エージェントのライフサイクル開始イベントを発行する
            from .models._openai_custom import AgentStartedEvent

            yield AgentStartedEvent()

            # 入力を適切なChatMessageまたは文字列に変換する
            user_message = self._convert_input_to_chat_message(request.input)

            # 会話パラメーターからThreadを取得する（OpenAI標準！）
            thread = None
            conversation_id = request.get_conversation_id()
            if conversation_id:
                thread = self.conversation_store.get_thread(conversation_id)
                if thread:
                    logger.debug(f"Using existing conversation: {conversation_id}")
                else:
                    logger.warning(f"Conversation {conversation_id} not found, proceeding without thread")

            if isinstance(user_message, str):
                logger.debug(f"Executing agent with text input: {user_message[:100]}...")
            else:
                logger.debug(f"Executing agent with multimodal ChatMessage: {type(user_message)}")
            # エージェントがストリーミングをサポートしているかチェックする
            if hasattr(agent, "run_stream") and callable(agent.run_stream):
                # Agent FrameworkのネイティブストリーミングをオプションのThread付きで使用する
                if thread:
                    async for update in agent.run_stream(user_message, thread=thread):
                        for trace_event in trace_collector.get_pending_events():
                            yield trace_event

                        yield update
                else:
                    async for update in agent.run_stream(user_message):
                        for trace_event in trace_collector.get_pending_events():
                            yield trace_event

                        yield update
            elif hasattr(agent, "run") and callable(agent.run):
                # 非ストリーミングAgent - run()を使用し、完全なレスポンスをyieldする
                logger.info("Agent lacks run_stream(), using run() method (non-streaming)")
                if thread:
                    response = await agent.run(user_message, thread=thread)
                else:
                    response = await agent.run(user_message)

                # レスポンスの前にtraceイベントをyieldする
                for trace_event in trace_collector.get_pending_events():
                    yield trace_event

                # 完全なレスポンスをyieldする（mapperがストリーミングイベントに変換します）
                yield response
            else:
                raise ValueError("Agent must implement either run() or run_stream() method")

            # Agentのライフサイクル完了イベントを発行する
            from .models._openai_custom import AgentCompletedEvent

            yield AgentCompletedEvent()

        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            # Agentのライフサイクル失敗イベントを発行する
            from .models._openai_custom import AgentFailedEvent

            yield AgentFailedEvent(error=e)

            # 後方互換性のためにエラーもyieldし続ける
            yield {"type": "error", "message": f"Agent execution error: {e!s}"}

    async def _execute_workflow(
        self, workflow: Any, request: AgentFrameworkRequest, trace_collector: Any
    ) -> AsyncGenerator[Any, None]:
        """トレース収集付きでAgent Frameworkのワークフローを実行します。

        Args:
            workflow: 実行するWorkflowオブジェクト
            request: 実行するRequest
            trace_collector: イベントを取得するためのTrace collector

        Yields:
            Workflowイベントとtraceイベント

        """
        try:
            # request.inputフィールドから直接入力データを取得する
            input_data = request.input
            logger.debug(f"Using input field: {type(input_data)}")

            # workflowの期待される入力タイプに基づいて入力を解析する
            parsed_input = await self._parse_workflow_input(workflow, input_data)

            logger.debug(f"Executing workflow with parsed input type: {type(parsed_input)}")

            # Agent Framework workflowのネイティブストリーミングを使用する
            async for event in workflow.run_stream(parsed_input):
                # 保留中のtraceイベントを最初にyieldする
                for trace_event in trace_collector.get_pending_events():
                    yield trace_event

                # 次にworkflowイベントをyieldする
                yield event

        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            yield {"type": "error", "message": f"Workflow execution error: {e!s}"}

    def _convert_input_to_chat_message(self, input_data: Any) -> Any:
        """OpenAI Responses APIの入力をAgent FrameworkのChatMessageまたは文字列に変換します。

        テキスト、画像、ファイル、マルチモーダルコンテンツなど様々な入力形式に対応します。
        単純なケースでは文字列抽出にフォールバックします。

        Args:
            input_data: OpenAI ResponseInputParam (List[ResponseInputItemParam])

        Returns:
            マルチモーダルコンテンツの場合はChatMessage、単純なテキストの場合は文字列

        """
        # Agent Frameworkの型をインポートする
        try:
            from agent_framework import ChatMessage, DataContent, Role, TextContent
        except ImportError:
            # Agent Frameworkが利用できない場合は文字列抽出にフォールバックする
            return self._extract_user_message_fallback(input_data)

        # 単純な文字列入力を処理する（後方互換性）
        if isinstance(input_data, str):
            return input_data

        # OpenAI ResponseInputParam (List[ResponseInputItemParam])を処理する
        if isinstance(input_data, list):
            return self._convert_openai_input_to_chat_message(input_data, ChatMessage, TextContent, DataContent, Role)

        # その他の形式に対するフォールバック
        return self._extract_user_message_fallback(input_data)

    def _convert_openai_input_to_chat_message(
        self, input_items: list[Any], ChatMessage: Any, TextContent: Any, DataContent: Any, Role: Any
    ) -> Any:
        """OpenAI ResponseInputParamをAgent FrameworkのChatMessageに変換します。

        OpenAI形式のテキスト、画像、ファイル、その他のコンテンツタイプを
        適切なコンテンツオブジェクトを持つAgent FrameworkのChatMessageに変換します。

        Args:
            input_items: OpenAI ResponseInputItemParamオブジェクトのリスト（辞書またはオブジェクト）
            ChatMessage: チャットメッセージ作成用のChatMessageクラス
            TextContent: テキストコンテンツ用のTextContentクラス
            DataContent: データ/メディアコンテンツ用のDataContentクラス
            Role: メッセージの役割用のRole列挙型

        Returns:
            変換されたコンテンツを持つChatMessage

        """
        contents = []

        # 各入力アイテムを処理する
        for item in input_items:
            # 辞書形式（JSONから）を処理する
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "message":
                    # OpenAIメッセージからコンテンツを抽出する
                    message_content = item.get("content", [])

                    # 文字列コンテンツとリストコンテンツの両方を処理する
                    if isinstance(message_content, str):
                        contents.append(TextContent(text=message_content))
                    elif isinstance(message_content, list):
                        for content_item in message_content:
                            # 辞書形式のコンテンツアイテムを処理する
                            if isinstance(content_item, dict):
                                content_type = content_item.get("type")

                                if content_type == "input_text":
                                    text = content_item.get("text", "")
                                    contents.append(TextContent(text=text))

                                elif content_type == "input_image":
                                    image_url = content_item.get("image_url", "")
                                    if image_url:
                                        # 可能であればdata URIからメディアタイプを抽出する data
                                        # URLからメディアタイプを解析し、フォールバックはimage/png
                                        if image_url.startswith("data:"):
                                            try:
                                                # data:image/jpeg;base64,...形式からメディアタイプを抽出する
                                                media_type = image_url.split(";")[0].split(":")[1]
                                            except (IndexError, AttributeError):
                                                logger.warning(
                                                    f"Failed to parse media type from data URL: {image_url[:30]}..."
                                                )
                                                media_type = "image/png"
                                        else:
                                            media_type = "image/png"
                                        contents.append(DataContent(uri=image_url, media_type=media_type))

                                elif content_type == "input_file":
                                    # ファイル入力を処理する
                                    file_data = content_item.get("file_data")
                                    file_url = content_item.get("file_url")
                                    filename = content_item.get("filename", "")

                                    # ファイル名からメディアタイプを判定する
                                    media_type = "application/octet-stream"  # デフォルト
                                    if filename:
                                        if filename.lower().endswith(".pdf"):
                                            media_type = "application/pdf"
                                        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                                            media_type = f"image/{filename.split('.')[-1].lower()}"
                                        elif filename.lower().endswith((
                                            ".wav",
                                            ".mp3",
                                            ".m4a",
                                            ".ogg",
                                            ".flac",
                                            ".aac",
                                        )):
                                            ext = filename.split(".")[-1].lower()
                                            # 拡張子を正規化して音声MIMEタイプに合わせる
                                            media_type = "audio/mp4" if ext == "m4a" else f"audio/{ext}"

                                    # file_dataまたはfile_urlを使用する
                                    if file_data:
                                        # file_dataはbase64と仮定し、data URIを作成する
                                        data_uri = f"data:{media_type};base64,{file_data}"
                                        contents.append(DataContent(uri=data_uri, media_type=media_type))
                                    elif file_url:
                                        contents.append(DataContent(uri=file_url, media_type=media_type))

                                elif content_type == "function_approval_response":
                                    # 関数承認レスポンスを処理する（DevUI拡張）
                                    try:
                                        from agent_framework import FunctionApprovalResponseContent, FunctionCallContent

                                        request_id = content_item.get("request_id", "")
                                        approved = content_item.get("approved", False)
                                        function_call_data = content_item.get("function_call", {})

                                        # function_callデータからFunctionCallContentを作成する
                                        function_call = FunctionCallContent(
                                            call_id=function_call_data.get("id", ""),
                                            name=function_call_data.get("name", ""),
                                            arguments=function_call_data.get("arguments", {}),
                                        )

                                        # 正しい署名でFunctionApprovalResponseContentを作成する
                                        approval_response = FunctionApprovalResponseContent(
                                            approved,  # positional argument
                                            id=request_id,  # keyword argument 'id', NOT 'request_id'
                                            function_call=function_call,  # FunctionCallContent object
                                        )
                                        contents.append(approval_response)
                                        logger.info(
                                            f"Added FunctionApprovalResponseContent: id={request_id}, "
                                            f"approved={approved}, call_id={function_call.call_id}"
                                        )
                                    except ImportError:
                                        logger.warning(
                                            "FunctionApprovalResponseContent not available in agent_framework"
                                        )
                                    except Exception as e:
                                        logger.error(f"Failed to create FunctionApprovalResponseContent: {e}")

            # 必要に応じて他のOpenAI入力アイテムタイプを処理する （ツール呼び出し、関数結果など）
            # コンテンツが見つからない場合は単純なテキストメッセージを作成する
        if not contents:
            contents.append(TextContent(text=""))

        chat_message = ChatMessage(role=Role.USER, contents=contents)

        logger.info(f"Created ChatMessage with {len(contents)} contents:")
        for idx, content in enumerate(contents):
            content_type = content.__class__.__name__
            if hasattr(content, "media_type"):
                logger.info(f"  [{idx}] {content_type} - media_type: {content.media_type}")
            else:
                logger.info(f"  [{idx}] {content_type}")

        return chat_message

    def _extract_user_message_fallback(self, input_data: Any) -> str:
        """ユーザーメッセージを文字列として抽出するフォールバックメソッド。

        Args:
            input_data: 様々な形式の入力データ

        Returns:
            抽出されたユーザーメッセージの文字列

        """
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, dict):
            # 一般的なフィールド名を試す
            for field in ["message", "text", "input", "content", "query"]:
                if field in input_data:
                    return str(input_data[field])
            # JSON文字列にフォールバックする
            return json.dumps(input_data)
        return str(input_data)

    async def _parse_workflow_input(self, workflow: Any, raw_input: Any) -> Any:
        """workflowの期待される入力タイプに基づいて入力を解析する。

        Args:
            workflow: Workflowオブジェクト
            raw_input: 生の入力データ

        Returns:
            workflowに適した解析済み入力

        """
        try:
            # 構造化入力を処理する
            if isinstance(raw_input, dict):
                return self._parse_structured_workflow_input(workflow, raw_input)
            return self._parse_raw_workflow_input(workflow, str(raw_input))

        except Exception as e:
            logger.warning(f"Error parsing workflow input: {e}")
            return raw_input

    def _get_start_executor_message_types(self, workflow: Any) -> tuple[Any | None, list[Any]]:
        """開始executorとその宣言された入力タイプを返す。"""
        try:
            start_executor = workflow.get_start_executor()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug(f"Unable to access workflow start executor: {exc}")
            return None, []

        if not start_executor:
            return None, []

        message_types: list[Any] = []

        try:
            input_types = getattr(start_executor, "input_types", None)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug(f"Failed to read executor input_types: {exc}")
        else:
            if input_types:
                message_types = list(input_types)

        if not message_types and hasattr(start_executor, "_handlers"):
            try:
                handlers = start_executor._handlers
                if isinstance(handlers, dict):
                    message_types = list(handlers.keys())
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.debug(f"Failed to read executor handlers: {exc}")

        return start_executor, message_types

    def _parse_structured_workflow_input(self, workflow: Any, input_data: dict[str, Any]) -> Any:
        """workflow実行のための構造化入力データを解析する。

        Args:
            workflow: Workflowオブジェクト
            input_data: 構造化入力データ

        Returns:
            workflow用に解析された入力

        """
        try:
            from ._utils import parse_input_for_type

            # 開始executorとその入力タイプを取得する
            start_executor, message_types = self._get_start_executor_message_types(workflow)
            if not start_executor:
                logger.debug("Cannot determine input type for workflow - using raw dict")
                return input_data

            if not message_types:
                logger.debug("No message types found for start executor - using raw dict")
                return input_data

            # 最初の（主要な）入力タイプを取得する
            from ._utils import select_primary_input_type

            input_type = select_primary_input_type(message_types)
            if input_type is None:
                logger.debug("Could not select primary input type for workflow - using raw dict")
                return input_data

            # _utilsの統合された解析ロジックを使用する
            return parse_input_for_type(input_data, input_type)

        except Exception as e:
            logger.warning(f"Error parsing structured workflow input: {e}")
            return input_data

    def _parse_raw_workflow_input(self, workflow: Any, raw_input: str) -> Any:
        """workflowの期待される入力タイプに基づいて生の入力文字列を解析する。

        Args:
            workflow: Workflowオブジェクト
            raw_input: 生の入力文字列

        Returns:
            workflow用に解析された入力

        """
        try:
            from ._utils import parse_input_for_type

            # 開始executorとその入力タイプを取得する
            start_executor, message_types = self._get_start_executor_message_types(workflow)
            if not start_executor:
                logger.debug("Cannot determine input type for workflow - using raw string")
                return raw_input

            if not message_types:
                logger.debug("No message types found for start executor - using raw string")
                return raw_input

            # 最初の（主要な）入力タイプを取得する
            from ._utils import select_primary_input_type

            input_type = select_primary_input_type(message_types)
            if input_type is None:
                logger.debug("Could not select primary input type for workflow - using raw string")
                return raw_input

            # _utilsの統合された解析ロジックを使用する
            return parse_input_for_type(raw_input, input_type)

        except Exception as e:
            logger.debug(f"Error parsing workflow input: {e}")
            return raw_input
