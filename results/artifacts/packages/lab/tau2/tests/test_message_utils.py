# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import patch

from agent_framework._types import ChatMessage, FunctionCallContent, FunctionResultContent, Role, TextContent
from agent_framework_lab_tau2._message_utils import flip_messages, log_messages


def test_flip_messages_user_to_assistant():
    """ユーザーメッセージをアシスタントに反転するテスト。"""
    messages = [
        ChatMessage(
            role=Role.USER, contents=[TextContent(text="Hello assistant")], author_name="User1", message_id="msg_001"
        )
    ]

    flipped = flip_messages(messages)

    assert len(flipped) == 1
    assert flipped[0].role == Role.ASSISTANT
    assert flipped[0].text == "Hello assistant"
    assert flipped[0].author_name == "User1"
    assert flipped[0].message_id == "msg_001"


def test_flip_messages_assistant_to_user():
    """アシスタントメッセージをユーザーに反転するテスト。"""
    messages = [
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="Hello user")],
            author_name="Assistant1",
            message_id="msg_002",
        )
    ]

    flipped = flip_messages(messages)

    assert len(flipped) == 1
    assert flipped[0].role == Role.USER
    assert flipped[0].text == "Hello user"
    assert flipped[0].author_name == "Assistant1"
    assert flipped[0].message_id == "msg_002"


def test_flip_messages_assistant_with_function_calls_filtered():
    """アシスタントからユーザーに反転する際に関数呼び出しがフィルタリングされることのテスト。"""
    function_call = FunctionCallContent(call_id="call_123", name="test_function", arguments={"param": "value"})

    messages = [
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="I'll call a function"), function_call, TextContent(text="After the call")],
            message_id="msg_003",
        )
    ]

    flipped = flip_messages(messages)

    assert len(flipped) == 1
    assert flipped[0].role == Role.USER
    # 関数呼び出しはフィルタリングされるべきです
    assert len(flipped[0].contents) == 2
    assert all(content.type == "text" for content in flipped[0].contents)
    assert "I'll call a function" in flipped[0].text
    assert "After the call" in flipped[0].text


def test_flip_messages_assistant_with_only_function_calls_skipped():
    """関数呼び出しのみのアシスタントメッセージがスキップされることのテスト。"""
    function_call = FunctionCallContent(call_id="call_456", name="another_function", arguments={"key": "value"})

    messages = [
        ChatMessage(role=Role.ASSISTANT, contents=[function_call], message_id="msg_004")  # Only function call, no text
    ]

    flipped = flip_messages(messages)

    # フィルタリング後にテキストコンテンツがないため空であるべきです
    assert len(flipped) == 0


def test_flip_messages_tool_messages_skipped():
    """ツールメッセージがスキップされることのテスト。"""
    function_result = FunctionResultContent(call_id="call_789", result={"success": True})

    messages = [ChatMessage(role=Role.TOOL, contents=[function_result])]

    flipped = flip_messages(messages)

    # ツールメッセージはスキップされるべきです
    assert len(flipped) == 0


def test_flip_messages_system_messages_preserved():
    """システムメッセージがそのまま保持されることのテスト。"""
    messages = [ChatMessage(role=Role.SYSTEM, contents=[TextContent(text="System instruction")], message_id="sys_001")]

    flipped = flip_messages(messages)

    assert len(flipped) == 1
    assert flipped[0].role == Role.SYSTEM
    assert flipped[0].text == "System instruction"
    assert flipped[0].message_id == "sys_001"


def test_flip_messages_mixed_conversation():
    """混合会話を反転するテスト。"""
    function_call = FunctionCallContent(call_id="call_mixed", name="mixed_function", arguments={})

    function_result = FunctionResultContent(call_id="call_mixed", result="function result")

    messages = [
        ChatMessage(role=Role.SYSTEM, contents=[TextContent(text="System prompt")]),
        ChatMessage(role=Role.USER, contents=[TextContent(text="User question")]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Assistant response"), function_call]),
        ChatMessage(role=Role.TOOL, contents=[function_result]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Final response")]),
    ]

    flipped = flip_messages(messages)

    # 以下を含むべきです: system（変更なし）、assistant（userから）、user（assistantから、フィルタリング済み）、
    # assistant（最終アシスタントから）
    assert len(flipped) == 4

    # 反転された各メッセージをチェック
    assert flipped[0].role == Role.SYSTEM
    assert flipped[0].text == "System prompt"

    assert flipped[1].role == Role.ASSISTANT
    assert flipped[1].text == "User question"

    assert flipped[2].role == Role.USER
    assert flipped[2].text == "Assistant response"  # 関数呼び出しはフィルタリングされました

    # ツールメッセージはスキップされました

    assert flipped[3].role == Role.USER
    assert flipped[3].text == "Final response"


def test_flip_messages_empty_list():
    """空のメッセージリストを反転するテスト。"""
    messages = []
    flipped = flip_messages(messages)
    assert len(flipped) == 0


def test_flip_messages_preserves_metadata():
    """反転中にメッセージのメタデータが保持されることのテスト。"""
    messages = [
        ChatMessage(
            role=Role.USER, contents=[TextContent(text="Test message")], author_name="TestUser", message_id="test_123"
        )
    ]

    flipped = flip_messages(messages)

    assert len(flipped) == 1
    assert flipped[0].author_name == "TestUser"
    assert flipped[0].message_id == "test_123"


@patch("agent_framework_lab_tau2._message_utils.logger")
def test_log_messages_text_content(mock_logger):
    """テキストコンテンツを含むメッセージのロギングテスト。"""
    messages = [
        ChatMessage(role=Role.USER, contents=[TextContent(text="Hello")]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Hi there!")]),
    ]

    log_messages(messages)

    # 各メッセージに対してlogger.infoが呼ばれるべきです
    assert mock_logger.opt.return_value.info.call_count == 2


@patch("agent_framework_lab_tau2._message_utils.logger")
def test_log_messages_function_call(mock_logger):
    """関数呼び出しを含むメッセージのロギングテスト。"""
    function_call = FunctionCallContent(call_id="call_log", name="log_function", arguments={"param": "value"})

    messages = [ChatMessage(role=Role.ASSISTANT, contents=[function_call])]

    log_messages(messages)

    # 関数呼び出しがログに記録されるべきです
    mock_logger.opt.return_value.info.assert_called()
    call_args = mock_logger.opt.return_value.info.call_args[0][0]
    assert "TOOL_CALL" in call_args
    assert "log_function" in call_args


@patch("agent_framework_lab_tau2._message_utils.logger")
def test_log_messages_function_result(mock_logger):
    """関数結果を含むメッセージのロギングテスト。"""
    function_result = FunctionResultContent(call_id="call_result", result="success")

    messages = [ChatMessage(role=Role.TOOL, contents=[function_result])]

    log_messages(messages)

    # 関数結果がログに記録されるべきです
    mock_logger.opt.return_value.info.assert_called()
    call_args = mock_logger.opt.return_value.info.call_args[0][0]
    assert "TOOL_RESULT" in call_args


@patch("agent_framework_lab_tau2._message_utils.logger")
def test_log_messages_different_roles(mock_logger):
    """異なる役割のメッセージが異なる色でログされることのテスト。"""
    messages = [
        ChatMessage(role=Role.SYSTEM, contents=[TextContent(text="System")]),
        ChatMessage(role=Role.USER, contents=[TextContent(text="User")]),
        ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="Assistant")]),
        ChatMessage(role=Role.TOOL, contents=[TextContent(text="Tool")]),
    ]

    log_messages(messages)

    # 各メッセージに対してloggerが呼ばれるべきです
    assert mock_logger.opt.return_value.info.call_count == 4

    # 異なるカラータグが使用されていることをチェック
    calls = mock_logger.opt.return_value.info.call_args_list
    system_call = calls[0][0][0]
    user_call = calls[1][0][0]
    assistant_call = calls[2][0][0]
    tool_call = calls[3][0][0]

    assert "cyan" in system_call or "SYSTEM" in system_call
    assert "green" in user_call or "USER" in user_call
    assert "blue" in assistant_call or "ASSISTANT" in assistant_call
    assert "yellow" in tool_call or "TOOL" in tool_call


@patch("agent_framework_lab_tau2._message_utils.logger")
def test_log_messages_escapes_html(mock_logger):
    """ログ出力でHTML風の文字が正しくエスケープされることのテスト。"""
    messages = [ChatMessage(role=Role.USER, contents=[TextContent(text="Message with <tag> content")])]

    log_messages(messages)

    mock_logger.opt.return_value.info.assert_called()
    call_args = mock_logger.opt.return_value.info.call_args[0][0]
    # <文字がエスケープされるべきです
    assert "\\<tag>" in call_args or "&lt;tag&gt;" in call_args


@patch("agent_framework_lab_tau2._message_utils.logger")
def test_log_messages_mixed_content_types(mock_logger):
    """混合コンテンツタイプのメッセージのロギングテスト。"""
    function_call = FunctionCallContent(call_id="mixed_call", name="mixed_function", arguments={"key": "value"})

    messages = [
        ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="I'll call a function"), function_call, TextContent(text="Done!")],
        )
    ]

    log_messages(messages)

    # 異なるコンテンツタイプごとに複数回ログされるべきです
    assert mock_logger.opt.return_value.info.call_count == 3
