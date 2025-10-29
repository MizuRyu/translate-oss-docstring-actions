# Copyright (c) Microsoft. All rights reserved.

import base64
import json
import re
import uuid
from collections.abc import AsyncIterable, Sequence
from typing import Any, cast

import httpx
from a2a.client import Client, ClientConfig, ClientFactory, minimal_agent_card
from a2a.client.auth.interceptor import AuthInterceptor
from a2a.types import (
    AgentCard,
    Artifact,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Task,
    TaskState,
    TextPart,
    TransportProtocol,
)
from a2a.types import Message as A2AMessage
from a2a.types import Part as A2APart
from a2a.types import Role as A2ARole
from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Contents,
    DataContent,
    Role,
    TextContent,
    UriContent,
    prepend_agent_framework_to_user_agent,
)

__all__ = ["A2AAgent"]

URI_PATTERN = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<base64_data>[A-Za-z0-9+/=]+)$")
TERMINAL_TASK_STATES = [
    TaskState.completed,
    TaskState.failed,
    TaskState.canceled,
    TaskState.rejected,
]


def _get_uri_data(uri: str) -> str:
    match = URI_PATTERN.match(uri)
    if not match:
        raise ValueError(f"Invalid data URI format: {uri}")

    return match.group("base64_data")


class A2AAgent(BaseAgent):
    """Agent2Agent (A2A) プロトコルの実装です。

    A2A ClientをラップしてAgent Frameworkを外部のA2A準拠エージェントとHTTP/JSON-RPC経由で接続します。
    送信時にフレームワークのChatMessagesをA2A Messagesに変換し、
    A2Aのレスポンス（Messages/Tasks）をフレームワークの型に戻します。
    BaseAgentの機能を継承しつつ、基盤となるA2Aプロトコル通信を管理します。

    URL、AgentCard、または既存のA2A Clientインスタンスで初期化可能です。

    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        description: str | None = None,
        agent_card: AgentCard | None = None,
        url: str | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        auth_interceptor: AuthInterceptor | None = None,
        **kwargs: Any,
    ) -> None:
        """A2AAgentを初期化します。

        キーワード引数:
            name: エージェントの名前。
            id: エージェントの一意識別子。指定しない場合は自動生成されます。
            description: エージェントの目的の簡単な説明。
            agent_card: エージェントのエージェントカード。
            url: A2AサーバーのURL。
            client: エージェントのA2Aクライアント。
            http_client: 使用するOptionalなhttpx.AsyncClient。
            auth_interceptor: セキュアなエンドポイント用のOptionalな認証インターセプター。
            kwargs: BaseAgentに渡される追加のプロパティ。

        """
        super().__init__(id=id, name=name, description=description, **kwargs)
        self._http_client: httpx.AsyncClient | None = http_client
        if client is not None:
            self.client = client
            self._close_http_client = True
            return
        if agent_card is None:
            if url is None:
                raise ValueError("Either agent_card or url must be provided")
            # URLから最小限のエージェントカードを作成します。
            agent_card = minimal_agent_card(url, [TransportProtocol.jsonrpc])

        # 提供されたhttpxクライアントを作成または使用します。
        if http_client is None:
            timeout = httpx.Timeout(
                connect=10.0,  # 10 seconds to establish connection
                read=60.0,  # 60 seconds to read response (A2A operations can take time)
                write=10.0,  # 10 seconds to send request
                pool=5.0,  # 5 seconds to get connection from pool
            )
            headers = prepend_agent_framework_to_user_agent()
            http_client = httpx.AsyncClient(timeout=timeout, headers=headers)
            self._http_client = http_client  # クリーンアップ用に保存します。
            self._close_http_client = True

        # ファクトリーを使ってA2Aクライアントを作成します。
        config = ClientConfig(
            httpx_client=http_client,
            supported_transports=[TransportProtocol.jsonrpc],
        )
        factory = ClientFactory(config)
        interceptors = [auth_interceptor] if auth_interceptor is not None else None
        self.client = factory.create(agent_card, interceptors=interceptors)  # type: ignore

    async def __aenter__(self) -> "A2AAgent":
        """非同期コンテキストマネージャのエントリー。"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """非同期コンテキストマネージャの終了時にhttpxクライアントをクリーンアップします。"""
        # 自分で作成したhttpxクライアントを閉じます。
        if self._http_client is not None and self._close_http_client:
            await self._http_client.aclose()

    async def run(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        """エージェントからレスポンスを取得します。

        このメソッドはエージェントの実行の最終結果を単一のAgentRunResponseオブジェクトとして返します。
        呼び出し元は最終結果が利用可能になるまでブロックされます。

        引数:
            messages: エージェントに送信するメッセージ。

        キーワード引数:
            thread: メッセージに関連付けられた会話スレッド。
            kwargs: 追加のキーワード引数。

        戻り値:
            エージェントのレスポンスアイテム。

        """
        # すべての更新を収集し、フレームワークを使って更新をレスポンスに統合します。
        updates = [update async for update in self.run_stream(messages, thread=thread, **kwargs)]
        return AgentRunResponse.from_agent_run_response_updates(updates)

    async def run_stream(
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        """エージェントをストリームとして実行します。

        このメソッドはエージェントの実行の中間ステップと最終結果を
        AgentRunResponseUpdateオブジェクトのストリームとして呼び出し元に返します。

        引数:
            messages: エージェントに送信するメッセージ。

        キーワード引数:
            thread: メッセージに関連付けられた会話スレッド。
            kwargs: 追加のキーワード引数。

        yield:
            エージェントのレスポンスアイテム。

        """
        messages = self._normalize_messages(messages)
        a2a_message = self._chat_message_to_a2a_message(messages[-1])

        response_stream = self.client.send_message(a2a_message)

        async for item in response_stream:
            if isinstance(item, Message):
                # A2A Messageを処理します。
                contents = self._a2a_parts_to_contents(item.parts)
                yield AgentRunResponseUpdate(
                    contents=contents,
                    role=Role.ASSISTANT if item.role == A2ARole.agent else Role.USER,
                    response_id=str(getattr(item, "message_id", uuid.uuid4())),
                    raw_representation=item,
                )
            elif isinstance(item, tuple) and len(item) == 2:  # ClientEvent = (Task, UpdateEvent)
                task, _update_event = item
                if isinstance(task, Task) and task.status.state in TERMINAL_TASK_STATES:
                    # TaskのアーティファクトをChatMessagesに変換し、個別の更新としてyieldします。
                    task_messages = self._task_to_chat_messages(task)
                    if task_messages:
                        for message in task_messages:
                            # raw_representationのアーティファクトのIDをmessage_idとして使用し、一意に識別します。
                            artifact_id = getattr(message.raw_representation, "artifact_id", None)
                            yield AgentRunResponseUpdate(
                                contents=message.contents,
                                role=message.role,
                                response_id=task.id,
                                message_id=artifact_id,
                                raw_representation=task,
                            )
                    else:
                        # 空のタスクです。
                        yield AgentRunResponseUpdate(
                            contents=[],
                            role=Role.ASSISTANT,
                            response_id=task.id,
                            raw_representation=task,
                        )
            else:
                # 不明なレスポンスのタイプです。
                msg = f"Only Message and Task responses are supported from A2A agents. Received: {type(item)}"
                raise NotImplementedError(msg)

    def _chat_message_to_a2a_message(self, message: ChatMessage) -> A2AMessage:
        """ChatMessageをA2A Messageに変換します。

        Agent FrameworkのChatMessageオブジェクトをA2AプロトコルのMessageに変換します。
        - すべてのメッセージ内容を適切なA2A Partタイプに変換
        - テキスト内容をTextPartオブジェクトにマッピング
        - ファイル参照（URI/data/hosted_file）をFilePartオブジェクトに変換
        - 元のメッセージのメタデータや追加プロパティを保持
        - フレームワークのメッセージはユーザー入力として扱うためroleを'user'に設定

        """
        parts: list[A2APart] = []
        if not message.contents:
            raise ValueError("ChatMessage.contents is empty; cannot convert to A2AMessage.")

        # すべての内容を処理します。
        for content in message.contents:
            match content.type:
                case "text":
                    parts.append(
                        A2APart(
                            root=TextPart(
                                text=content.text,
                                metadata=content.additional_properties,
                            )
                        )
                    )
                case "error":
                    parts.append(
                        A2APart(
                            root=TextPart(
                                text=content.message or "An error occurred.",
                                metadata=content.additional_properties,
                            )
                        )
                    )
                case "uri":
                    parts.append(
                        A2APart(
                            root=FilePart(
                                file=FileWithUri(
                                    uri=content.uri,
                                    mime_type=content.media_type,
                                ),
                                metadata=content.additional_properties,
                            )
                        )
                    )
                case "data":
                    parts.append(
                        A2APart(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=_get_uri_data(content.uri),
                                    mime_type=content.media_type,
                                ),
                                metadata=content.additional_properties,
                            )
                        )
                    )
                case "hosted_file":
                    parts.append(
                        A2APart(
                            root=FilePart(
                                file=FileWithUri(
                                    uri=content.file_id,
                                    mime_type=None,  # HostedFileContent doesn't specify media_type
                                ),
                                metadata=content.additional_properties,
                            )
                        )
                    )
                case _:
                    raise ValueError(f"Unknown content type: {content.type}")

        return A2AMessage(
            role=A2ARole("user"),
            parts=parts,
            message_id=message.message_id or uuid.uuid4().hex,
            metadata=cast(dict[str, Any], message.additional_properties),
        )

    def _a2a_parts_to_contents(self, parts: Sequence[A2APart]) -> list[Contents]:
        """A2A PartsをAgent FrameworkのContentsに変換します。

        A2AプロトコルのPartsをフレームワークネイティブのContentオブジェクトに変換し、
        テキスト、ファイル（URI/バイト）、データパーツをメタデータを保持しつつ処理します。

        """
        contents: list[Contents] = []
        for part in parts:
            inner_part = part.root
            match inner_part.kind:
                case "text":
                    contents.append(
                        TextContent(
                            text=inner_part.text,
                            additional_properties=inner_part.metadata,
                            raw_representation=inner_part,
                        )
                    )
                case "file":
                    if isinstance(inner_part.file, FileWithUri):
                        contents.append(
                            UriContent(
                                uri=inner_part.file.uri,
                                media_type=inner_part.file.mime_type or "",
                                additional_properties=inner_part.metadata,
                                raw_representation=inner_part,
                            )
                        )
                    elif isinstance(inner_part.file, FileWithBytes):
                        contents.append(
                            DataContent(
                                data=base64.b64decode(inner_part.file.bytes),
                                media_type=inner_part.file.mime_type or "",
                                additional_properties=inner_part.metadata,
                                raw_representation=inner_part,
                            )
                        )
                case "data":
                    contents.append(
                        TextContent(
                            text=json.dumps(inner_part.data),
                            additional_properties=inner_part.metadata,
                            raw_representation=inner_part,
                        )
                    )
                case _:
                    raise ValueError(f"Unknown Part kind: {inner_part.kind}")
        return contents

    def _task_to_chat_messages(self, task: Task) -> list[ChatMessage]:
        """A2A TaskのアーティファクトをASSISTANTロールのChatMessagesに変換します。"""
        messages: list[ChatMessage] = []

        if task.artifacts is not None:
            for artifact in task.artifacts:
                messages.append(self._artifact_to_chat_message(artifact))

        return messages

    def _artifact_to_chat_message(self, artifact: Artifact) -> ChatMessage:
        """A2A Artifactをパーツの内容を使ってChatMessageに変換します。"""
        contents = self._a2a_parts_to_contents(artifact.parts)
        return ChatMessage(
            role=Role.ASSISTANT,
            contents=contents,
            raw_representation=artifact,
        )
