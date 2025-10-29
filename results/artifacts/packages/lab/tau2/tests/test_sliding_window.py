# Copyright (c) Microsoft. All rights reserved.

"""スライディングウィンドウメッセージリストのテスト。"""

from unittest.mock import patch

from agent_framework._types import ChatMessage, FunctionCallContent, FunctionResultContent, Role, TextContent
from agent_framework_lab_tau2._sliding_window import SlidingWindowChatMessageStore


def test_initialization_empty():
    """メッセージなしでの初期化テスト。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)

    assert sliding_window.max_tokens == 1000
    assert sliding_window.system_message is None
    assert sliding_window.tool_definitions is None
    assert len(sliding_window.messages) == 0
    assert len(sliding_window.truncated_messages) == 0


def test_initialization_with_parameters():
    """システムメッセージとツール定義での初期化テスト。"""
    system_msg = "You are a helpful assistant"
    tool_defs = [{"name": "test_tool", "description": "A test tool"}]

    sliding_window = SlidingWindowChatMessageStore(
        max_tokens=2000, system_message=system_msg, tool_definitions=tool_defs
    )

    assert sliding_window.max_tokens == 2000
    assert sliding_window.system_message == system_msg
    assert sliding_window.tool_definitions == tool_defs


def test_initialization_with_messages():
    """既存メッセージでの初期化テスト。"""
    messages = [
        ChatMessage(role=Role.USER, contents=[TextContent(text="Hello")]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Hi there!")]),
    ]

    sliding_window = SlidingWindowChatMessageStore(messages=messages, max_tokens=1000)

    assert len(sliding_window.messages) == 2
    assert len(sliding_window.truncated_messages) == 2


async def test_add_messages_simple():
    """切り詰めなしでメッセージを追加するテスト。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=10000)  # 大きな制限値

    new_messages = [
        ChatMessage(role=Role.USER, contents=[TextContent(text="What's the weather?")]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="I can help with that.")]),
    ]

    await sliding_window.add_messages(new_messages)

    messages = await sliding_window.list_messages()
    assert len(messages) == 2
    assert messages[0].text == "What's the weather?"
    assert messages[1].text == "I can help with that."


async def test_list_all_messages_vs_list_messages():
    """list_all_messagesとlist_messagesの違いのテスト。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=50)  # 切り詰めを強制する小さな制限値

    # 切り詰めを引き起こすために多くのメッセージを追加
    messages = [
        ChatMessage(role=Role.USER, contents=[TextContent(text=f"Message {i} with some content")]) for i in range(10)
    ]

    await sliding_window.add_messages(messages)

    truncated_messages = await sliding_window.list_messages()
    all_messages = await sliding_window.list_all_messages()

    # すべてのメッセージはすべての内容を含むべきです
    assert len(all_messages) == 10

    # トークン制限により切り詰められたメッセージは少なくなるべきです
    assert len(truncated_messages) < len(all_messages)


def test_get_token_count_basic():
    """基本的なトークンカウントのテスト。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)
    sliding_window.truncated_messages = [ChatMessage(role=Role.USER, contents=[TextContent(text="Hello")])]

    token_count = sliding_window.get_token_count()

    # 0より多いはずです（正確な数はエンコーディングによる）
    assert token_count > 0


def test_get_token_count_with_system_message():
    """システムメッセージを含むトークンカウントのテスト。"""
    system_msg = "You are a helpful assistant"
    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000, system_message=system_msg)

    # メッセージなしの場合
    token_count_empty = sliding_window.get_token_count()

    # メッセージを追加
    sliding_window.truncated_messages = [ChatMessage(role=Role.USER, contents=[TextContent(text="Hello")])]
    token_count_with_message = sliding_window.get_token_count()

    # メッセージありの方がトークン数が多いはず
    assert token_count_with_message > token_count_empty
    assert token_count_empty > 0  # システムメッセージはトークンに寄与する


def test_get_token_count_function_call():
    """関数呼び出しを含むトークンカウントのテスト。"""
    function_call = FunctionCallContent(call_id="call_123", name="test_function", arguments={"param": "value"})

    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)
    sliding_window.truncated_messages = [ChatMessage(role=Role.ASSISTANT, contents=[function_call])]

    token_count = sliding_window.get_token_count()
    assert token_count > 0


def test_get_token_count_function_result():
    """関数結果を含むトークンカウントのテスト。"""
    function_result = FunctionResultContent(call_id="call_123", result={"success": True, "data": "result"})

    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)
    sliding_window.truncated_messages = [ChatMessage(role=Role.TOOL, contents=[function_result])]

    token_count = sliding_window.get_token_count()
    assert token_count > 0


@patch("agent_framework_lab_tau2._sliding_window.logger")
def test_truncate_messages_removes_old_messages(mock_logger):
    """トークン制限を超えた場合に古いメッセージが切り詰められることのテスト。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=20)  # 非常に小さな制限値

    # 制限を超えるメッセージを作成
    messages = [
        ChatMessage(
            role=Role.USER,
            contents=[TextContent(text="This is a very long message that should exceed the token limit")],
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="This is another very long message that should also exceed the token limit")],
        ),
        ChatMessage(role=Role.USER, contents=[TextContent(text="Short msg")]),
    ]

    sliding_window.truncated_messages = messages.copy()
    sliding_window.truncate_messages()

    # 切り詰め後はメッセージ数が少なくなるはず
    assert len(sliding_window.truncated_messages) < len(messages)

    # 警告がログに記録されるべきです
    assert mock_logger.warning.called


@patch("agent_framework_lab_tau2._sliding_window.logger")
def test_truncate_messages_removes_leading_tool_messages(mock_logger):
    """トランケーションが先頭のtoolメッセージを削除することをテストします。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=10000)  # 大きな制限

    # toolメッセージで始まるメッセージを作成する
    tool_message = ChatMessage(role=Role.TOOL, contents=[FunctionResultContent(call_id="call_123", result="result")])
    user_message = ChatMessage(role=Role.USER, contents=[TextContent(text="Hello")])

    sliding_window.truncated_messages = [tool_message, user_message]
    sliding_window.truncate_messages()

    # toolメッセージは先頭から削除されるべきです
    assert len(sliding_window.truncated_messages) == 1
    assert sliding_window.truncated_messages[0].role == Role.USER

    # toolメッセージを削除したことに関する警告がログに記録されているはずです
    mock_logger.warning.assert_called()


def test_estimate_any_object_token_count_dict():
    """辞書オブジェクトのトークンカウントをテストします。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)

    test_dict = {"key": "value", "number": 42}
    token_count = sliding_window.estimate_any_object_token_count(test_dict)

    assert token_count > 0


def test_estimate_any_object_token_count_string():
    """文字列オブジェクトのトークンカウントをテストします。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)

    test_string = "This is a test string"
    token_count = sliding_window.estimate_any_object_token_count(test_string)

    assert token_count > 0


def test_estimate_any_object_token_count_non_serializable():
    """JSONシリアライズ不可能なオブジェクトのトークンカウントをテストします。"""
    sliding_window = SlidingWindowChatMessageStore(max_tokens=1000)

    # JSONシリアライズできないオブジェクトを作成する
    class CustomObject:
        def __str__(self):
            return "CustomObject instance"

    custom_obj = CustomObject()
    token_count = sliding_window.estimate_any_object_token_count(custom_obj)

    # 文字列表現にフォールバックするはずです
    assert token_count > 0


async def test_real_world_scenario():
    """現実的な会話シナリオをテストします。"""
    sliding_window = SlidingWindowChatMessageStore(
        max_tokens=30,
        system_message="You are a helpful assistant",  # Moderate limit
    )

    # 会話をシミュレートする
    conversation = [
        ChatMessage(role=Role.USER, contents=[TextContent(text="Hello, how are you?")]),
        ChatMessage(
            role=Role.ASSISTANT, contents=[TextContent(text="I'm doing well, thank you! How can I help you today?")]
        ),
        ChatMessage(role=Role.USER, contents=[TextContent(text="Can you tell me about the weather?")]),
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[
                TextContent(
                    text="I'd be happy to help with weather information, "
                    "but I don't have access to current weather data."
                )
            ],
        ),
        ChatMessage(role=Role.USER, contents=[TextContent(text="What about telling me a joke instead?")]),
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="Sure! Why don't scientists trust atoms? Because they make up everything!")],
        ),
    ]

    await sliding_window.add_messages(conversation)

    current_messages = await sliding_window.list_messages()
    all_messages = await sliding_window.list_all_messages()

    # すべてのメッセージは保持されるべきです
    assert len(all_messages) == 6

    # 現在のメッセージはトランケーションされる可能性があります
    assert len(current_messages) <= 6

    # トークン数は制限内または制限に近いはずです
    token_count = sliding_window.get_token_count()
    # トランケーションは制限を超えた時に発生するため、ある程度のマージンを許容します
    assert token_count <= sliding_window.max_tokens * 1.1
