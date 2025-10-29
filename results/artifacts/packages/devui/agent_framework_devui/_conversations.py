# Copyright (c) Microsoft. All rights reserved.

"""OpenAI Conversations APIのための会話ストレージ抽象化。

このモジュールは、AgentFrameworkのAgentThreadを内部でラップしながら、
会話管理のためのクリーンな抽象レイヤーを提供する。
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, cast

from agent_framework import AgentThread, ChatMessage
from openai.types.conversations import Conversation, ConversationDeletedResource
from openai.types.conversations.conversation_item import ConversationItem
from openai.types.conversations.message import Message
from openai.types.conversations.text_content import TextContent
from openai.types.responses import (
    ResponseFunctionToolCallItem,
    ResponseFunctionToolCallOutputItem,
    ResponseInputFile,
    ResponseInputImage,
)

# OpenAI Messageのroleリテラルの型エイリアス
MessageRole = Literal["unknown", "user", "assistant", "system", "critic", "discriminator", "developer", "tool"]


class ConversationStore(ABC):
    """会話ストレージの抽象基底クラス。

    AgentThreadインスタンスを内部で管理しながら
    OpenAI Conversations APIインターフェースを提供する。

    """

    @abstractmethod
    def create_conversation(self, metadata: dict[str, str] | None = None) -> Conversation:
        """新しい会話を作成する（AgentThreadの作成をラップ）。

        Args:
            metadata: オプションのメタデータ辞書（例：{"agent_id": "weather_agent"}）

        Returns:
            生成されたIDを持つConversationオブジェクト

        """
        pass

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """会話のメタデータを取得する。

        Args:
            conversation_id: 会話ID

        Returns:
            Conversationオブジェクトまたは見つからなければNone

        """
        pass

    @abstractmethod
    def update_conversation(self, conversation_id: str, metadata: dict[str, str]) -> Conversation:
        """会話のメタデータを更新します。

        Args:
            conversation_id: 会話のID
            metadata: 新しいメタデータの辞書

        Returns:
            更新されたConversationオブジェクト

        Raises:
            ValueError: 会話が見つからない場合

        """
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        """会話を削除します（AgentThreadを含む）。

        Args:
            conversation_id: 会話のID

        Returns:
            ConversationDeletedResourceオブジェクト

        Raises:
            ValueError: 会話が見つからない場合

        """
        pass

    @abstractmethod
    async def add_items(self, conversation_id: str, items: list[dict[str, Any]]) -> list[ConversationItem]:
        """会話にアイテムを追加します（AgentThread.message_storeと同期）。

        Args:
            conversation_id: 会話のID
            items: 追加する会話アイテムのリスト

        Returns:
            追加されたConversationItemオブジェクトのリスト

        Raises:
            ValueError: 会話が見つからない場合

        """
        pass

    @abstractmethod
    async def list_items(
        self, conversation_id: str, limit: int = 100, after: str | None = None, order: str = "asc"
    ) -> tuple[list[ConversationItem], bool]:
        """AgentThread.message_storeから会話アイテムを一覧表示します。

        Args:
            conversation_id: 会話のID
            limit: 返すアイテムの最大数
            after: ページネーション用カーソル（item_id）
            order: ソート順（"asc" または "desc"）

        Returns:
            (アイテムリスト, has_moreのbool値)のタプル

        Raises:
            ValueError: 会話が見つからない場合

        """
        pass

    @abstractmethod
    def get_item(self, conversation_id: str, item_id: str) -> ConversationItem | None:
        """特定の会話アイテムを取得します。

        Args:
            conversation_id: 会話のID
            item_id: アイテムのID

        Returns:
            ConversationItemまたは見つからない場合はNone

        """
        pass

    @abstractmethod
    def get_thread(self, conversation_id: str) -> AgentThread | None:
        """実行のための基盤となるAgentThreadを取得します（内部使用）。

        これは、executorが会話コンテキストでAgentを実行するためのAgentThreadを取得するための重要なメソッドです。

        Args:
            conversation_id: 会話のID

        Returns:
            AgentThreadオブジェクトまたは見つからない場合はNone

        """
        pass

    @abstractmethod
    def list_conversations_by_metadata(self, metadata_filter: dict[str, str]) -> list[Conversation]:
        """メタデータ（例：agent_id）で会話をフィルタリングします。

        Args:
            metadata_filter: 一致させるメタデータのキーと値のペア

        Returns:
            一致するConversationオブジェクトのリスト

        """
        pass


class InMemoryConversationStore(ConversationStore):
    """AgentThreadをラップしたインメモリの会話ストレージ。

    この実装は、会話とその基盤となるAgentThreadインスタンスをメモリ内に保存し、実行に使用します。

    """

    def __init__(self) -> None:
        """インメモリの会話ストレージを初期化します。

        ストレージ構造は、会話IDを基盤となるAgentThread、メタデータ、およびキャッシュされたConversationItemsを含む会話データにマッピングします。

        """
        self._conversations: dict[str, dict[str, Any]] = {}

        # O(1)検索のためのアイテムインデックス: {conversation_id: {item_id: ConversationItem}}
        self._item_index: dict[str, dict[str, ConversationItem]] = {}

    def create_conversation(self, metadata: dict[str, str] | None = None) -> Conversation:
        """基盤となるAgentThreadを持つ新しい会話を作成します。"""
        conv_id = f"conv_{uuid.uuid4().hex}"
        created_at = int(time.time())

        # デフォルトのChatMessageStoreを持つAgentThreadを作成します。
        thread = AgentThread()

        self._conversations[conv_id] = {
            "id": conv_id,
            "thread": thread,
            "metadata": metadata or {},
            "created_at": created_at,
            "items": [],
        }

        # この会話のためのアイテムインデックスを初期化します。
        self._item_index[conv_id] = {}

        return Conversation(id=conv_id, object="conversation", created_at=created_at, metadata=metadata)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """会話のメタデータを取得します。"""
        conv_data = self._conversations.get(conversation_id)
        if not conv_data:
            return None

        return Conversation(
            id=conv_data["id"],
            object="conversation",
            created_at=conv_data["created_at"],
            metadata=conv_data.get("metadata"),
        )

    def update_conversation(self, conversation_id: str, metadata: dict[str, str]) -> Conversation:
        """会話のメタデータを更新します。"""
        conv_data = self._conversations.get(conversation_id)
        if not conv_data:
            raise ValueError(f"Conversation {conversation_id} not found")

        conv_data["metadata"] = metadata

        return Conversation(
            id=conv_data["id"],
            object="conversation",
            created_at=conv_data["created_at"],
            metadata=metadata,
        )

    def delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        """会話とそのAgentThreadを削除します。"""
        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        del self._conversations[conversation_id]
        # アイテムインデックスをクリーンアップします。
        self._item_index.pop(conversation_id, None)

        return ConversationDeletedResource(id=conversation_id, object="conversation.deleted", deleted=True)

    async def add_items(self, conversation_id: str, items: list[dict[str, Any]]) -> list[ConversationItem]:
        """会話にアイテムを追加し、AgentThreadと同期します。"""
        conv_data = self._conversations.get(conversation_id)
        if not conv_data:
            raise ValueError(f"Conversation {conversation_id} not found")

        thread: AgentThread = conv_data["thread"]

        # アイテムをChatMessagesに変換し、スレッドに追加します。
        chat_messages = []
        for item in items:
            # 単純な変換 - 現時点ではテキストコンテンツを想定しています。
            role = item.get("role", "user")
            content = item.get("content", [])
            text = content[0].get("text", "") if content else ""

            chat_msg = ChatMessage(role=role, contents=[{"type": "text", "text": text}])
            chat_messages.append(chat_msg)

        # AgentThreadにメッセージを追加します。
        await thread.on_new_messages(chat_messages)

        # Messageオブジェクトを作成します（ConversationItemはUnion型なので具体的なMessageタイプを使用）。
        conv_items: list[ConversationItem] = []
        for msg in chat_messages:
            item_id = f"item_{uuid.uuid4().hex}"

            # 役割を抽出します - 文字列とenumの両方に対応。
            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            role = cast(MessageRole, role_str)  # 安全: Agent Frameworkの役割はOpenAIの役割と一致します。

            # ChatMessageの内容をOpenAIのTextContent形式に変換します。
            message_content = []
            for content_item in msg.contents:
                if hasattr(content_item, "type") and content_item.type == "text":
                    # TextContentオブジェクトからテキストを抽出します。
                    text_value = getattr(content_item, "text", "")
                    message_content.append(TextContent(type="text", text=text_value))

            # Messageオブジェクトを作成します（ConversationItemのUnionから具体的なタイプ）。
            message = Message(
                id=item_id,
                type="message",  # Required discriminator for union
                role=role,
                content=message_content,
                status="completed",  # Required field
            )
            conv_items.append(message)

        # アイテムをキャッシュします。
        conv_data["items"].extend(conv_items)

        # O(1)検索のためにアイテムインデックスを更新します。
        if conversation_id not in self._item_index:
            self._item_index[conversation_id] = {}

        for conv_item in conv_items:
            if conv_item.id:  # Guard against None
                self._item_index[conversation_id][conv_item.id] = conv_item

        return conv_items

    async def list_items(
        self, conversation_id: str, limit: int = 100, after: str | None = None, order: str = "asc"
    ) -> tuple[list[ConversationItem], bool]:
        """AgentThreadのメッセージストアから会話アイテムを一覧表示します。

        AgentFrameworkのChatMessagesを適切なOpenAIのConversationItemタイプに変換します:
        - テキスト/画像/ファイルを含むメッセージ → Message
        - 関数呼び出し → ResponseFunctionToolCallItem
        - 関数結果 → ResponseFunctionToolCallOutputItem

        """
        conv_data = self._conversations.get(conversation_id)
        if not conv_data:
            raise ValueError(f"Conversation {conversation_id} not found")

        thread: AgentThread = conv_data["thread"]

        # スレッドのメッセージストアからメッセージを取得します。
        items: list[ConversationItem] = []
        if thread.message_store:
            af_messages = await thread.message_store.list_messages()

            # 各AgentFrameworkのChatMessageを適切なConversationItemタイプに変換します。
            for i, msg in enumerate(af_messages):
                item_id = f"item_{i}"
                role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                role = cast(MessageRole, role_str)  # 安全: Agent Frameworkの役割はOpenAIの役割と一致します。

                # メッセージ内の各コンテンツアイテムを処理します。
                # 単一のChatMessageは複数のConversationItemsを生成する場合があります
                # （例：テキストと関数呼び出しの両方を含むメッセージ）。
                message_contents: list[TextContent | ResponseInputImage | ResponseInputFile] = []
                function_calls = []
                function_results = []

                for content in msg.contents:
                    content_type = getattr(content, "type", None)

                    if content_type == "text":
                        # Message用のテキストコンテンツ。
                        text_value = getattr(content, "text", "")
                        message_contents.append(TextContent(type="text", text=text_value))

                    elif content_type == "data":
                        # データコンテンツ（画像、ファイル、PDF）。
                        uri = getattr(content, "uri", "")
                        media_type = getattr(content, "media_type", None)

                        if media_type and media_type.startswith("image/"):
                            # ResponseInputImageに変換します。
                            message_contents.append(
                                ResponseInputImage(type="input_image", image_url=uri, detail="auto")
                            )
                        else:
                            # ResponseInputFileに変換します 可能であればURIからファイル名を抽出します。
                            filename = None
                            if media_type == "application/pdf":
                                filename = "document.pdf"

                            message_contents.append(
                                ResponseInputFile(type="input_file", file_url=uri, filename=filename)
                            )

                    elif content_type == "function_call":
                        # 関数呼び出し - 別のConversationItemを作成します。
                        call_id = getattr(content, "call_id", None)
                        name = getattr(content, "name", "")
                        arguments = getattr(content, "arguments", "")

                        if call_id and name:
                            function_calls.append(
                                ResponseFunctionToolCallItem(
                                    id=f"{item_id}_call_{call_id}",
                                    call_id=call_id,
                                    name=name,
                                    arguments=arguments,
                                    type="function_call",
                                    status="completed",
                                )
                            )

                    elif content_type == "function_result":
                        # 関数結果 - 別のConversationItemを作成します。
                        call_id = getattr(content, "call_id", None)
                        # 出力はadditional_propertiesに保存されます。
                        output = ""
                        if hasattr(content, "additional_properties"):
                            output = content.additional_properties.get("output", "")

                        if call_id:
                            function_results.append(
                                ResponseFunctionToolCallOutputItem(
                                    id=f"{item_id}_result_{call_id}",
                                    call_id=call_id,
                                    output=output,
                                    type="function_call_output",
                                    status="completed",
                                )
                            )

                # 検出した内容に基づいてConversationItemsを作成します
                # メッセージにテキスト/画像/ファイルがあれば、Messageアイテムを作成します。
                if message_contents:
                    message = Message(
                        id=item_id,
                        type="message",
                        role=role,  # type: ignore
                        content=message_contents,  # type: ignore
                        status="completed",
                    )
                    items.append(message)

                # 関数呼び出しアイテムを追加します。
                items.extend(function_calls)

                # 関数結果アイテムを追加します。
                items.extend(function_results)

        # ページネーションを適用します。
        if order == "desc":
            items = items[::-1]

        start_idx = 0
        if after:
            # カーソルの後のインデックスを見つけます。
            for i, item in enumerate(items):
                if item.id == after:
                    start_idx = i + 1
                    break

        paginated_items = items[start_idx : start_idx + limit]
        has_more = len(items) > start_idx + limit

        return paginated_items, has_more

    def get_item(self, conversation_id: str, item_id: str) -> ConversationItem | None:
        """特定の会話アイテムを取得します - インデックスによるO(1)検索。"""
        # 線形検索の代わりにインデックスを使用してO(1)検索を行います。
        conv_items = self._item_index.get(conversation_id)
        if not conv_items:
            return None

        return conv_items.get(item_id)

    def get_thread(self, conversation_id: str) -> AgentThread | None:
        """実行のためのAgentThreadを取得します - agent.run_stream()にとって重要です。"""
        conv_data = self._conversations.get(conversation_id)
        return conv_data["thread"] if conv_data else None

    def list_conversations_by_metadata(self, metadata_filter: dict[str, str]) -> list[Conversation]:
        """メタデータ（例：agent_id）で会話をフィルタリングします。"""
        results = []
        for conv_data in self._conversations.values():
            conv_meta = conv_data.get("metadata", {})
            # すべてのフィルタ項目が一致するか確認します。
            if all(conv_meta.get(k) == v for k, v in metadata_filter.items()):
                results.append(
                    Conversation(
                        id=conv_data["id"],
                        object="conversation",
                        created_at=conv_data["created_at"],
                        metadata=conv_meta,
                    )
                )
        return results
