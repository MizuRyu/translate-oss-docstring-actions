# Copyright (c) Microsoft. All rights reserved.

"""conversation store実装のテスト。"""

from typing import cast

import pytest
from openai.types.conversations import InputFileContent, InputImageContent, InputTextContent

from agent_framework_devui._conversations import InMemoryConversationStore


@pytest.mark.asyncio
async def test_create_conversation():
    """会話の作成をテスト。"""
    store = InMemoryConversationStore()

    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    assert conversation.id.startswith("conv_")
    assert conversation.object == "conversation"
    assert conversation.metadata == {"agent_id": "test_agent"}


@pytest.mark.asyncio
async def test_get_conversation():
    """会話の取得をテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    created = store.create_conversation(metadata={"agent_id": "test_agent"})

    # それを取得
    retrieved = store.get_conversation(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.metadata == {"agent_id": "test_agent"}


@pytest.mark.asyncio
async def test_get_conversation_not_found():
    """存在しない会話の取得をテスト。"""
    store = InMemoryConversationStore()

    conversation = store.get_conversation("conv_nonexistent")

    assert conversation is None


@pytest.mark.asyncio
async def test_update_conversation():
    """会話メタデータの更新をテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    created = store.create_conversation(metadata={"agent_id": "test_agent"})

    # メタデータを更新
    updated = store.update_conversation(created.id, metadata={"agent_id": "new_agent", "session_id": "sess_123"})

    assert updated.id == created.id
    assert updated.metadata == {"agent_id": "new_agent", "session_id": "sess_123"}


@pytest.mark.asyncio
async def test_delete_conversation():
    """会話の削除をテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    created = store.create_conversation(metadata={"agent_id": "test_agent"})

    # それを削除
    result = store.delete_conversation(created.id)

    assert result.id == created.id
    assert result.deleted is True
    assert result.object == "conversation.deleted"

    # 削除されたことを確認
    assert store.get_conversation(created.id) is None


@pytest.mark.asyncio
async def test_get_thread():
    """基盤となるAgentThreadの取得をテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    # スレッドを取得
    thread = store.get_thread(conversation.id)

    assert thread is not None
    # AgentThreadはmessage_storeを持つべき
    assert hasattr(thread, "message_store")


@pytest.mark.asyncio
async def test_get_thread_not_found():
    """存在しない会話のスレッド取得をテスト。"""
    store = InMemoryConversationStore()

    thread = store.get_thread("conv_nonexistent")

    assert thread is None


@pytest.mark.asyncio
async def test_list_conversations_by_metadata():
    """メタデータによる会話のフィルタリングをテスト。"""
    store = InMemoryConversationStore()

    # 複数の会話を作成
    _conv1 = store.create_conversation(metadata={"agent_id": "agent1"})
    _conv2 = store.create_conversation(metadata={"agent_id": "agent2"})
    conv3 = store.create_conversation(metadata={"agent_id": "agent1", "session_id": "sess_1"})

    # agent_idでフィルタリング
    results = store.list_conversations_by_metadata({"agent_id": "agent1"})

    assert len(results) == 2
    assert all(cast(dict[str, str], c.metadata).get("agent_id") == "agent1" for c in results if c.metadata)

    # agent_idとsession_idでフィルタリング
    results = store.list_conversations_by_metadata({"agent_id": "agent1", "session_id": "sess_1"})

    assert len(results) == 1
    assert results[0].id == conv3.id


@pytest.mark.asyncio
async def test_add_items():
    """会話にアイテムを追加するテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    # アイテムを追加
    items = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    conv_items = await store.add_items(conversation.id, items=items)

    assert len(conv_items) == 1
    # メッセージはConversationItemタイプ - 標準のOpenAIフィールドをチェック
    assert conv_items[0].type == "message"
    assert conv_items[0].role == "user"
    assert conv_items[0].status == "completed"
    assert len(conv_items[0].content) == 1
    assert conv_items[0].content[0].type == "text"
    text_content = cast(InputTextContent, conv_items[0].content[0])
    assert text_content.text == "Hello"


@pytest.mark.asyncio
async def test_list_items():
    """会話のアイテム一覧をテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    # アイテムを追加
    items = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
    ]
    await store.add_items(conversation.id, items=items)

    # アイテムを一覧表示
    retrieved_items, has_more = await store.list_items(conversation.id)

    assert len(retrieved_items) >= 2  # 少なくとも追加したアイテムが含まれていること
    assert has_more is False


@pytest.mark.asyncio
async def test_list_items_pagination():
    """アイテム一覧のページネーションをテスト。"""
    store = InMemoryConversationStore()

    # 会話を作成
    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    # 複数のアイテムを追加
    items = [{"role": "user", "content": [{"type": "text", "text": f"Message {i}"}]} for i in range(5)]
    await store.add_items(conversation.id, items=items)

    # limit付きで一覧表示
    retrieved_items, has_more = await store.list_items(conversation.id, limit=3)

    assert len(retrieved_items) == 3
    assert has_more is True


@pytest.mark.asyncio
async def test_list_items_converts_function_calls():
    """list_itemsが関数呼び出しをResponseFunctionToolCallItemに正しく変換することをテスト。"""
    from agent_framework import ChatMessage, ChatMessageStore, Role

    store = InMemoryConversationStore()

    # 会話を作成
    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    # 基盤となるスレッドを取得し、message storeをセットアップ
    thread = store.get_thread(conversation.id)
    assert thread is not None

    # 存在しない場合はmessage storeを初期化
    if thread.message_store is None:
        thread.message_store = ChatMessageStore()

    # 関数呼び出しを伴うエージェント実行からのメッセージをシミュレート
    messages = [
        ChatMessage(role=Role.USER, contents=[{"type": "text", "text": "What's the weather in SF?"}]),
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city": "San Francisco"}',
                    "call_id": "call_test123",
                }
            ],
        ),
        ChatMessage(
            role=Role.TOOL,
            contents=[
                {
                    "type": "function_result",
                    "call_id": "call_test123",
                    "output": '{"temperature": 65, "condition": "sunny"}',
                }
            ],
        ),
        ChatMessage(role=Role.ASSISTANT, contents=[{"type": "text", "text": "The weather is sunny, 65°F"}]),
    ]

    # スレッドにメッセージを追加
    await thread.on_new_messages(messages)

    # 会話のアイテムを一覧表示
    items, has_more = await store.list_items(conversation.id)

    # 正しい数とタイプのアイテムが取得できたことを検証
    assert len(items) == 4, f"Expected 4 items, got {len(items)}"
    assert has_more is False

    # アイテムタイプをチェック
    assert items[0].type == "message", "First item should be a message"
    assert items[0].role == "user"
    assert len(items[0].content) == 1
    text_content_0 = cast(InputTextContent, items[0].content[0])
    assert text_content_0.text == "What's the weather in SF?"

    assert items[1].type == "function_call", "Second item should be a function_call"
    assert items[1].call_id == "call_test123"
    assert items[1].name == "get_weather"
    assert items[1].arguments == '{"city": "San Francisco"}'
    assert items[1].status == "completed"

    assert items[2].type == "function_call_output", "Third item should be a function_call_output"
    assert items[2].call_id == "call_test123"
    assert items[2].output == '{"temperature": 65, "condition": "sunny"}'
    assert items[2].status == "completed"

    assert items[3].type == "message", "Fourth item should be a message"
    assert items[3].role == "assistant"
    assert len(items[3].content) == 1
    text_content_3 = cast(InputTextContent, items[3].content[0])
    assert text_content_3.text == "The weather is sunny, 65°F"

    # CRITICAL: 空のメッセージアイテムがないことを保証
    for item in items:
        if item.type == "message":
            assert len(item.content) > 0, f"Message item {item.id} has empty content!"


@pytest.mark.asyncio
async def test_list_items_handles_images_and_files():
    """list_itemsがデータコンテンツ（画像/ファイル）をOpenAIタイプに正しく変換することをテスト。"""
    from agent_framework import ChatMessage, ChatMessageStore, Role

    store = InMemoryConversationStore()

    # 会話を作成
    conversation = store.create_conversation(metadata={"agent_id": "test_agent"})

    # 基盤となるスレッドを取得
    thread = store.get_thread(conversation.id)
    assert thread is not None

    if thread.message_store is None:
        thread.message_store = ChatMessageStore()

    # 画像とファイルを含むメッセージをシミュレート
    messages = [
        ChatMessage(
            role=Role.USER,
            contents=[
                {"type": "text", "text": "Check this image and PDF"},
                {"type": "data", "uri": "data:image/png;base64,iVBORw0KGgo=", "media_type": "image/png"},
                {"type": "data", "uri": "data:application/pdf;base64,JVBERi0=", "media_type": "application/pdf"},
            ],
        ),
    ]

    await thread.on_new_messages(messages)

    # アイテムを一覧表示
    items, has_more = await store.list_items(conversation.id)

    assert len(items) == 1
    assert items[0].type == "message"
    assert items[0].role == "user"
    assert len(items[0].content) == 3

    # コンテンツタイプをチェック
    assert items[0].content[0].type == "text"
    text_content = cast(InputTextContent, items[0].content[0])
    assert text_content.text == "Check this image and PDF"

    assert items[0].content[1].type == "input_image"
    image_content = cast(InputImageContent, items[0].content[1])
    assert image_content.image_url == "data:image/png;base64,iVBORw0KGgo="
    assert image_content.detail == "auto"

    assert items[0].content[2].type == "input_file"
    file_content = cast(InputFileContent, items[0].content[2])
    assert file_content.file_url == "data:application/pdf;base64,JVBERi0="
