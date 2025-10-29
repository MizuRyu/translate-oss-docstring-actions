# Copyright (c) Microsoft. All rights reserved.

"""FastAPIサーバーの実装。"""

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ._discovery import EntityDiscovery
from ._executor import AgentFrameworkExecutor
from ._mapper import MessageMapper
from .models import AgentFrameworkRequest, OpenAIError
from .models._discovery_models import DiscoveryResponse, EntityInfo

logger = logging.getLogger(__name__)


class DevServer:
    """開発サーバー - Agentのデバッグ用OpenAI互換APIサーバー。"""

    def __init__(
        self,
        entities_dir: str | None = None,
        port: int = 8080,
        host: str = "127.0.0.1",
        cors_origins: list[str] | None = None,
        ui_enabled: bool = True,
    ) -> None:
        """開発サーバーを初期化する。

        Args:
            entities_dir: エンティティをスキャンするディレクトリ
            port: サーバーを起動するポート
            host: サーバーをバインドするホスト
            cors_origins: 許可するCORSオリジンのリスト
            ui_enabled: UIを有効にするかどうか

        """
        self.entities_dir = entities_dir
        self.port = port
        self.host = host
        self.cors_origins = cors_origins or ["*"]
        self.ui_enabled = ui_enabled
        self.executor: AgentFrameworkExecutor | None = None
        self._app: FastAPI | None = None
        self._pending_entities: list[Any] | None = None

    async def _ensure_executor(self) -> AgentFrameworkExecutor:
        """executorが初期化されていることを保証する。"""
        if self.executor is None:
            logger.info("Initializing Agent Framework executor...")

            # コンポーネントを直接作成する
            entity_discovery = EntityDiscovery(self.entities_dir)
            message_mapper = MessageMapper()
            self.executor = AgentFrameworkExecutor(entity_discovery, message_mapper)

            # ディレクトリからエンティティを発見する
            discovered_entities = await self.executor.discover_entities()
            logger.info(f"Discovered {len(discovered_entities)} entities from directory")

            # 保留中のインメモリエンティティを登録する
            if self._pending_entities:
                discovery = self.executor.entity_discovery
                for entity in self._pending_entities:
                    try:
                        entity_info = await discovery.create_entity_info_from_object(entity, source="in-memory")
                        discovery.register_entity(entity_info.id, entity_info, entity)
                        logger.info(f"Registered in-memory entity: {entity_info.id}")
                    except Exception as e:
                        logger.error(f"Failed to register in-memory entity: {e}")
                self._pending_entities = None  # 登録後にクリアする

            # すべての登録後の最終的なエンティティ数を取得する
            all_entities = self.executor.entity_discovery.list_entities()
            logger.info(f"Total entities available: {len(all_entities)}")

        return self.executor

    async def _cleanup_entities(self) -> None:
        """エンティティリソースをクリーンアップする（クライアント、MCPツール、資格情報などを閉じる）。"""
        if not self.executor:
            return

        logger.info("Cleaning up entity resources...")
        entities = self.executor.entity_discovery.list_entities()
        closed_count = 0
        mcp_tools_closed = 0
        credentials_closed = 0

        for entity_info in entities:
            try:
                entity_obj = self.executor.entity_discovery.get_entity_object(entity_info.id)

                # チャットクライアントとその資格情報を閉じる
                if entity_obj and hasattr(entity_obj, "chat_client"):
                    client = entity_obj.chat_client

                    # チャットクライアント自体を閉じる
                    if hasattr(client, "close") and callable(client.close):
                        if inspect.iscoroutinefunction(client.close):
                            await client.close()
                        else:
                            client.close()
                        closed_count += 1
                        logger.debug(f"Closed client for entity: {entity_info.id}")

                    # チャットクライアントに付随する資格情報を閉じる（例：AzureCliCredential）
                    credential_attrs = ["credential", "async_credential", "_credential", "_async_credential"]
                    for attr in credential_attrs:
                        if hasattr(client, attr):
                            cred = getattr(client, attr)
                            if cred and hasattr(cred, "close") and callable(cred.close):
                                try:
                                    if inspect.iscoroutinefunction(cred.close):
                                        await cred.close()
                                    else:
                                        cred.close()
                                    credentials_closed += 1
                                    logger.debug(f"Closed credential for entity: {entity_info.id}")
                                except Exception as e:
                                    logger.warning(f"Error closing credential for {entity_info.id}: {e}")

                # MCPツールを閉じる（フレームワークは_local_mcp_toolsで追跡）
                if entity_obj and hasattr(entity_obj, "_local_mcp_tools"):
                    for mcp_tool in entity_obj._local_mcp_tools:
                        if hasattr(mcp_tool, "close") and callable(mcp_tool.close):
                            try:
                                if inspect.iscoroutinefunction(mcp_tool.close):
                                    await mcp_tool.close()
                                else:
                                    mcp_tool.close()
                                mcp_tools_closed += 1
                                tool_name = getattr(mcp_tool, "name", "unknown")
                                logger.debug(f"Closed MCP tool '{tool_name}' for entity: {entity_info.id}")
                            except Exception as e:
                                logger.warning(f"Error closing MCP tool for {entity_info.id}: {e}")

            except Exception as e:
                logger.warning(f"Error closing entity {entity_info.id}: {e}")

        if closed_count > 0:
            logger.info(f"Closed {closed_count} entity client(s)")
        if credentials_closed > 0:
            logger.info(f"Closed {credentials_closed} credential(s)")
        if mcp_tools_closed > 0:
            logger.info(f"Closed {mcp_tools_closed} MCP tool(s)")

    def create_app(self) -> FastAPI:
        """FastAPIアプリケーションを作成する。"""

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            # 起動処理
            logger.info("Starting Agent Framework Server")
            await self._ensure_executor()
            yield
            # シャットダウン処理
            logger.info("Shutting down Agent Framework Server")

            # エンティティリソースをクリーンアップする（例：資格情報、クライアントを閉じる）
            if self.executor:
                await self._cleanup_entities()

        app = FastAPI(
            title="Agent Framework Server",
            description="OpenAI-compatible API server for Agent Framework and other AI frameworks",
            version="1.0.0",
            lifespan=lifespan,
        )

        # CORSミドルウェアを追加する
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_routes(app)
        self._mount_ui(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """APIルートを登録する。"""

        @app.get("/health")
        async def health_check() -> dict[str, Any]:
            """ヘルスチェックエンドポイント。"""
            executor = await self._ensure_executor()
            # list_entities()を使用してエンティティの再発見と再登録を避ける
            entities = executor.entity_discovery.list_entities()

            return {"status": "healthy", "entities_count": len(entities), "framework": "agent_framework"}

        @app.get("/v1/entities", response_model=DiscoveryResponse)
        async def discover_entities() -> DiscoveryResponse:
            """登録されているすべてのエンティティをリストする。"""
            try:
                executor = await self._ensure_executor()
                # discover_entities()の代わりにlist_entities()を使用して既に登録されたエンティティを取得する
                entities = executor.entity_discovery.list_entities()
                return DiscoveryResponse(entities=entities)
            except Exception as e:
                logger.error(f"Error listing entities: {e}")
                raise HTTPException(status_code=500, detail=f"Entity listing failed: {e!s}") from e

        @app.get("/v1/entities/{entity_id}/info", response_model=EntityInfo)
        async def get_entity_info(entity_id: str) -> EntityInfo:
            """特定のエンティティの詳細情報を取得する（遅延ロードをトリガー）。"""
            try:
                executor = await self._ensure_executor()
                entity_info = executor.get_entity_info(entity_id)

                if not entity_info:
                    raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

                # エンティティがまだロードされていない場合に遅延ロードをトリガーする これによりモジュールがインポートされ、メタデータが充実する
                entity_obj = await executor.entity_discovery.load_entity(entity_id)

                # 更新されたエンティティ情報を取得する（ロード中に充実されている可能性あり）
                entity_info = executor.get_entity_info(entity_id) or entity_info

                # ワークフローの場合、追加の詳細情報を補完する
                if entity_info.type == "workflow" and entity_obj:
                    # エンティティオブジェクトは上記のload_entity()で既にロード済み ワークフロー構造を取得する
                    workflow_dump = None
                    if hasattr(entity_obj, "to_dict") and callable(getattr(entity_obj, "to_dict", None)):
                        try:
                            workflow_dump = entity_obj.to_dict()  # type: ignore[attr-defined]
                        except Exception:
                            workflow_dump = None
                    elif hasattr(entity_obj, "to_json") and callable(getattr(entity_obj, "to_json", None)):
                        try:
                            raw_dump = entity_obj.to_json()  # type: ignore[attr-defined]
                        except Exception:
                            workflow_dump = None
                        else:
                            if isinstance(raw_dump, (bytes, bytearray)):
                                try:
                                    raw_dump = raw_dump.decode()
                                except Exception:
                                    raw_dump = raw_dump.decode(errors="replace")
                            if isinstance(raw_dump, str):
                                try:
                                    parsed_dump = json.loads(raw_dump)
                                except Exception:
                                    workflow_dump = raw_dump
                                else:
                                    workflow_dump = parsed_dump if isinstance(parsed_dump, dict) else raw_dump
                            else:
                                workflow_dump = raw_dump
                    elif hasattr(entity_obj, "__dict__"):
                        workflow_dump = {k: v for k, v in entity_obj.__dict__.items() if not k.startswith("_")}

                    # 入力スキーマ情報を取得する
                    input_schema = {}
                    input_type_name = "Unknown"
                    start_executor_id = ""

                    try:
                        from ._utils import (
                            extract_executor_message_types,
                            generate_input_schema,
                            select_primary_input_type,
                        )

                        start_executor = entity_obj.get_start_executor()
                    except Exception as e:
                        logger.debug(f"Could not extract input info for workflow {entity_id}: {e}")
                    else:
                        if start_executor:
                            start_executor_id = getattr(start_executor, "executor_id", "") or getattr(
                                start_executor, "id", ""
                            )

                            message_types = extract_executor_message_types(start_executor)
                            input_type = select_primary_input_type(message_types)

                            if input_type:
                                input_type_name = getattr(input_type, "__name__", str(input_type))

                                # 包括的なスキーマ生成を用いてスキーマを生成する
                                input_schema = generate_input_schema(input_type)

                    if not input_schema:
                        input_schema = {"type": "string"}
                        if input_type_name == "Unknown":
                            input_type_name = "string"

                    # executorリストを取得する
                    executor_list = []
                    if hasattr(entity_obj, "executors") and entity_obj.executors:
                        executor_list = [getattr(ex, "executor_id", str(ex)) for ex in entity_obj.executors]

                    # エンティティ情報のコピーを作成し、ワークフロー固有のフィールドを補完する
                    update_payload: dict[str, Any] = {
                        "workflow_dump": workflow_dump,
                        "input_schema": input_schema,
                        "input_type_name": input_type_name,
                        "start_executor_id": start_executor_id,
                    }
                    if executor_list:
                        update_payload["executors"] = executor_list
                    return entity_info.model_copy(update=update_payload)

                # ワークフロー以外のエンティティはそのまま返す
                return entity_info

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting entity info for {entity_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get entity info: {e!s}") from e

        @app.post("/v1/entities/{entity_id}/reload")
        async def reload_entity(entity_id: str) -> dict[str, Any]:
            """エンティティをホットリロードする（キャッシュをクリアし、次回アクセス時に再インポートされる）。

            これは開発中のホットリロードを可能にします - エンティティコードを編集し、このエンドポイントを呼び出すと、
            次の実行でサーバー再起動なしに更新されたコードが使用されます。

            """
            try:
                executor = await self._ensure_executor()

                # エンティティが存在するか確認する
                entity_info = executor.get_entity_info(entity_id)
                if not entity_info:
                    raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

                # キャッシュを無効化する
                executor.entity_discovery.invalidate_entity(entity_id)

                return {
                    "success": True,
                    "message": f"Entity '{entity_id}' cache cleared. Will reload on next access.",
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error reloading entity {entity_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to reload entity: {e!s}") from e

        @app.post("/v1/responses")
        async def create_response(request: AgentFrameworkRequest, raw_request: Request) -> Any:
            """OpenAI Responses APIエンドポイント。"""
            try:
                raw_body = await raw_request.body()
                logger.info(f"Raw request body: {raw_body.decode()}")
                logger.info(f"Parsed request: model={request.model}, extra_body={request.extra_body}")

                # 新しい方法でentity_idを取得する
                entity_id = request.get_entity_id()
                logger.info(f"Extracted entity_id: {entity_id}")

                if not entity_id:
                    error = OpenAIError.create(f"Missing entity_id. Request extra_body: {request.extra_body}")
                    return JSONResponse(status_code=400, content=error.to_dict())

                # executorを取得し、エンティティの存在を検証する
                executor = await self._ensure_executor()
                try:
                    entity_info = executor.get_entity_info(entity_id)
                    logger.info(f"Found entity: {entity_info.name} ({entity_info.type})")
                except Exception:
                    error = OpenAIError.create(f"Entity not found: {entity_id}")
                    return JSONResponse(status_code=404, content=error.to_dict())

                # リクエストを実行する
                if request.stream:
                    return StreamingResponse(
                        self._stream_execution(executor, request),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Access-Control-Allow-Origin": "*",
                        },
                    )
                return await executor.execute_sync(request)

            except Exception as e:
                logger.error(f"Error executing request: {e}")
                error = OpenAIError.create(f"Execution failed: {e!s}")
                return JSONResponse(status_code=500, content=error.to_dict())

        # ======================================== OpenAI Conversations API (Standard)
        # ========================================

        @app.post("/v1/conversations")
        async def create_conversation(request_data: dict[str, Any]) -> dict[str, Any]:
            """新しい会話を作成する - OpenAI標準。"""
            try:
                metadata = request_data.get("metadata")
                executor = await self._ensure_executor()
                conversation = executor.conversation_store.create_conversation(metadata=metadata)
                return conversation.model_dump()
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creating conversation: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create conversation: {e!s}") from e

        @app.get("/v1/conversations")
        async def list_conversations(agent_id: str | None = None) -> dict[str, Any]:
            """会話をリストし、オプションでagent_idでフィルタリングする。"""
            try:
                executor = await self._ensure_executor()

                if agent_id:
                    # agent_idメタデータでフィルタリングする
                    conversations = executor.conversation_store.list_conversations_by_metadata({"agent_id": agent_id})
                else:
                    # すべての会話を返す（InMemoryStoreの場合はすべてをリスト） 注意:
                    # これはlist_conversations_by_metadata({})がすべてを返すことを前提としています
                    conversations = executor.conversation_store.list_conversations_by_metadata({})

                return {
                    "object": "list",
                    "data": [conv.model_dump() for conv in conversations],
                    "has_more": False,
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error listing conversations: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list conversations: {e!s}") from e

        @app.get("/v1/conversations/{conversation_id}")
        async def retrieve_conversation(conversation_id: str) -> dict[str, Any]:
            """会話を取得する - OpenAI標準。"""
            try:
                executor = await self._ensure_executor()
                conversation = executor.conversation_store.get_conversation(conversation_id)
                if not conversation:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                return conversation.model_dump()
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting conversation {conversation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get conversation: {e!s}") from e

        @app.post("/v1/conversations/{conversation_id}")
        async def update_conversation(conversation_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
            """会話のメタデータを更新する - OpenAI標準。"""
            try:
                executor = await self._ensure_executor()
                metadata = request_data.get("metadata", {})
                conversation = executor.conversation_store.update_conversation(conversation_id, metadata=metadata)
                return conversation.model_dump()
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating conversation {conversation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update conversation: {e!s}") from e

        @app.delete("/v1/conversations/{conversation_id}")
        async def delete_conversation(conversation_id: str) -> dict[str, Any]:
            """会話を削除する - OpenAI標準。"""
            try:
                executor = await self._ensure_executor()
                result = executor.conversation_store.delete_conversation(conversation_id)
                return result.model_dump()
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting conversation {conversation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {e!s}") from e

        @app.post("/v1/conversations/{conversation_id}/items")
        async def create_conversation_items(conversation_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
            """会話にアイテムを追加する - OpenAI標準。"""
            try:
                executor = await self._ensure_executor()
                items = request_data.get("items", [])
                conv_items = await executor.conversation_store.add_items(conversation_id, items=items)
                return {"object": "list", "data": [item.model_dump() for item in conv_items]}
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error adding items to conversation {conversation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add items: {e!s}") from e

        @app.get("/v1/conversations/{conversation_id}/items")
        async def list_conversation_items(
            conversation_id: str, limit: int = 100, after: str | None = None, order: str = "asc"
        ) -> dict[str, Any]:
            """会話アイテムをリストする - OpenAI標準。"""
            try:
                executor = await self._ensure_executor()
                items, has_more = await executor.conversation_store.list_items(
                    conversation_id, limit=limit, after=after, order=order
                )
                return {
                    "object": "list",
                    "data": [item.model_dump() for item in items],
                    "has_more": has_more,
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error listing items for conversation {conversation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list items: {e!s}") from e

        @app.get("/v1/conversations/{conversation_id}/items/{item_id}")
        async def retrieve_conversation_item(conversation_id: str, item_id: str) -> dict[str, Any]:
            """特定の会話アイテムを取得する - OpenAI標準。"""
            try:
                executor = await self._ensure_executor()
                item = executor.conversation_store.get_item(conversation_id, item_id)
                if not item:
                    raise HTTPException(status_code=404, detail="Item not found")
                return item.model_dump()
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting item {item_id} from conversation {conversation_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get item: {e!s}") from e

    async def _stream_execution(
        self, executor: AgentFrameworkExecutor, request: AgentFrameworkRequest
    ) -> AsyncGenerator[str, None]:
        """executorを通じて直接ストリーム実行する。"""
        try:
            # 最終的なresponse.completedイベントのためにイベントを収集する
            events = []

            # すべてのイベントをストリームする
            async for event in executor.execute_streaming(request):
                events.append(event)

                # 重要: model_dump_jsonを最初にチェックする。to_json()は改行を含む可能性があり（整形表示）、
                # SSEフォーマットを破壊する。model_dump_json()は単一行のJSONを返す。
                if hasattr(event, "model_dump_json"):
                    payload = event.model_dump_json()  # type: ignore[attr-defined]
                elif hasattr(event, "to_json") and callable(getattr(event, "to_json", None)):
                    payload = event.to_json()  # type: ignore[attr-defined]
                    # SSE互換性のために整形表示されたJSONから改行を除去する
                    payload = payload.replace("\n", "").replace("\r", "")
                elif isinstance(event, dict):
                    # プレーンなdictイベントを処理する（例：executorからのエラーイベント）
                    payload = json.dumps(event)
                elif hasattr(event, "to_dict") and callable(getattr(event, "to_dict", None)):
                    payload = json.dumps(event.to_dict())  # type: ignore[attr-defined]
                else:
                    payload = json.dumps(str(event))
                yield f"data: {payload}\n\n"

            # 最終応答に集約し、response.completedイベントを発行する（OpenAI標準）
            from .models import ResponseCompletedEvent

            final_response = await executor.message_mapper.aggregate_to_response(events, request)
            completed_event = ResponseCompletedEvent(
                type="response.completed",
                response=final_response,
                sequence_number=len(events),
            )
            yield f"data: {completed_event.model_dump_json()}\n\n"

            # 最終のdoneイベントを送信する
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming execution: {e}")
            error_event = {"id": "error", "object": "error", "error": {"message": str(e), "type": "execution_error"}}
            yield f"data: {json.dumps(error_event)}\n\n"

    def _mount_ui(self, app: FastAPI) -> None:
        """UIを静的ファイルとしてマウントする。"""
        from pathlib import Path

        ui_dir = Path(__file__).parent / "ui"
        if ui_dir.exists() and ui_dir.is_dir() and self.ui_enabled:
            app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    def register_entities(self, entities: list[Any]) -> None:
        """サーバー起動時に発見されるエンティティを登録する。

        Args:
            entities: 登録するエンティティオブジェクトのリスト

        """
        if self._pending_entities is None:
            self._pending_entities = []
        self._pending_entities.extend(entities)

    def get_app(self) -> FastAPI:
        """FastAPIアプリケーションインスタンスを取得する。"""
        if self._app is None:
            self._app = self.create_app()
        return self._app
