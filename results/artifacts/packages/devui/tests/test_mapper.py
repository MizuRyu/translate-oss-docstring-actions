# Copyright (c) Microsoft. All rights reserved.

"""メッセージマッピング機能に特化したクリーンなテスト。"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

# 実際のタイプのためにメインのagent_frameworkパッケージを追加する
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "main"))

# Agent Frameworkのタイプをインポートする（常に利用可能と仮定）
from agent_framework._types import (
    AgentRunResponseUpdate,
    ErrorContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
)

from agent_framework_devui._mapper import MessageMapper
from agent_framework_devui.models._openai_custom import AgentFrameworkRequest


def create_test_content(content_type: str, **kwargs: Any) -> Any:
    """テスト用コンテンツオブジェクトを作成する。"""
    if content_type == "text":
        return TextContent(text=kwargs.get("text", "Hello, world!"))
    if content_type == "function_call":
        return FunctionCallContent(
            call_id=kwargs.get("call_id", "test_call_id"),
            name=kwargs.get("name", "test_func"),
            arguments=kwargs.get("arguments", {"param": "value"}),
        )
    if content_type == "error":
        return ErrorContent(message=kwargs.get("message", "Test error"), error_code=kwargs.get("code", "test_error"))
    raise ValueError(f"Unknown content type: {content_type}")


def create_test_agent_update(contents: list[Any]) -> Any:
    """テスト用AgentRunResponseUpdateを作成する - 偽の属性なし！"""
    return AgentRunResponseUpdate(
        contents=contents, role=Role.ASSISTANT, message_id="test_msg", response_id="test_resp"
    )


@pytest.fixture
def mapper() -> MessageMapper:
    return MessageMapper()


@pytest.fixture
def test_request() -> AgentFrameworkRequest:
    # 簡略化されたルーティングを使用：model = entity_id
    return AgentFrameworkRequest(
        model="test_agent",  # Model IS the entity_id
        input="Test input",
        stream=True,
    )


async def test_critical_isinstance_bug_detection(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """重大：isinstance と hasattr のバグを検出できたテスト。"""

    content = create_test_content("text", text="Bug detection test")
    update = create_test_agent_update([content])

    # バグを検出できた重要なアサーション
    assert hasattr(update, "contents")  # 実際の属性 ✅
    assert not hasattr(update, "response")  # 偽の属性は存在すべきでない ✅

    # 実際のタイプで isinstance が動作することをテストする
    assert isinstance(update, AgentRunResponseUpdate)

    # マッパー変換をテスト - "Unknown event" を生成してはならない
    events = await mapper.convert_event(update, test_request)

    assert len(events) > 0
    assert all(hasattr(event, "type") for event in events)
    # 適切なタイプで未知のイベントは決して発生しないはずである
    assert all(event.type != "unknown" for event in events)


async def test_text_content_mapping(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """適切なOpenAIイベント階層でのTextContentマッピングをテストする。"""
    content = create_test_content("text", text="Hello, clean test!")
    update = create_test_agent_update([content])

    events = await mapper.convert_event(update, test_request)

    # 適切なOpenAI階層では、3つのイベントを期待する： 1. response.output_item.added (message) 2.
    # response.content_part.added (text part) 3. response.output_text.delta (実際のテキスト)
    assert len(events) == 3

    # message output item をチェックする
    assert events[0].type == "response.output_item.added"
    assert events[0].item.type == "message"
    assert events[0].item.role == "assistant"

    # content part をチェックする
    assert events[1].type == "response.content_part.added"
    assert events[1].part.type == "output_text"

    # text delta をチェックする
    assert events[2].type == "response.output_text.delta"
    assert events[2].delta == "Hello, clean test!"


async def test_function_call_mapping(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """FunctionCallContent マッピングをテストする。"""
    content = create_test_content("function_call", name="test_func", arguments={"location": "TestCity"})
    update = create_test_agent_update([content])

    events = await mapper.convert_event(update, test_request)

    # 生成されるべき：response.output_item.added + response.function_call_arguments.delta
    assert len(events) >= 2
    assert events[0].type == "response.output_item.added"
    assert events[1].type == "response.function_call_arguments.delta"

    # JSONがdeltaイベントに含まれていることをチェックする
    delta_events = [e for e in events if e.type == "response.function_call_arguments.delta"]
    full_json = "".join(event.delta for event in delta_events)
    assert "TestCity" in full_json


async def test_function_result_content_with_string_result(
    mapper: MessageMapper, test_request: AgentFrameworkRequest
) -> None:
    """プレーンな文字列結果を持つFunctionResultContentをテストする（通常のツール）。"""
    content = FunctionResultContent(
        call_id="test_call_123",
        result="Hello, World!",  # Plain string like regular Python function tools
    )
    update = create_test_agent_update([content])

    events = await mapper.convert_event(update, test_request)

    # response.function_result.complete イベントを生成するべきである
    assert len(events) >= 1
    result_events = [e for e in events if e.type == "response.function_result.complete"]
    assert len(result_events) == 1
    assert result_events[0].output == "Hello, World!"
    assert result_events[0].call_id == "test_call_123"
    assert result_events[0].status == "completed"


async def test_function_result_content_with_nested_content_objects(
    mapper: MessageMapper, test_request: AgentFrameworkRequest
) -> None:
    """ネストされたContentオブジェクトを持つFunctionResultContentをテストする（MCPツールの場合）。

    GitHub #1476 の問題をテストする。MCPツールはFunctionResultContentを返し、
    ネストされたTextContentオブジェクトが正しくシリアライズされない問題がある。
    """
    # これがMCPツールが返すもので、結果はネストされたContentオブジェクトを含む
    content = FunctionResultContent(
        call_id="mcp_call_456",
        result=[TextContent(text="Hello from MCP!")],  # List containing TextContent object
    )
    update = create_test_agent_update([content])

    events = await mapper.convert_event(update, test_request)

    # ネストされたContentオブジェクトを正常にシリアライズできるべきである
    assert len(events) >= 1
    result_events = [e for e in events if e.type == "response.function_result.complete"]
    assert len(result_events) == 1

    # 出力はネストされたTextContentのテキストを含むべきであり、TypeErrorや空の出力はないべきである
    assert result_events[0].output != ""
    assert "Hello from MCP!" in result_events[0].output
    assert result_events[0].call_id == "mcp_call_456"


async def test_function_result_content_with_multiple_nested_content_objects(
    mapper: MessageMapper, test_request: AgentFrameworkRequest
) -> None:
    """複数のネストされたContentオブジェクトを持つFunctionResultContentをテストする。"""
    # MCPツールは複数のContentオブジェクトを返すことができる
    content = FunctionResultContent(
        call_id="mcp_call_789",
        result=[
            TextContent(text="First result"),
            TextContent(text="Second result"),
        ],
    )
    update = create_test_agent_update([content])

    events = await mapper.convert_event(update, test_request)

    assert len(events) >= 1
    result_events = [e for e in events if e.type == "response.function_result.complete"]
    assert len(result_events) == 1

    # すべてのネストされたContentオブジェクトをシリアライズするべきである
    output = result_events[0].output
    assert output != ""
    assert "First result" in output
    assert "Second result" in output


async def test_error_content_mapping(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """ErrorContent マッピングをテストする。"""
    content = create_test_content("error", message="Test error", code="test_code")
    update = create_test_agent_update([content])

    events = await mapper.convert_event(update, test_request)

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].message == "Test error"
    assert events[0].code == "test_code"


async def test_mixed_content_types(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """複数のコンテンツタイプを一緒にテストする。"""
    contents = [
        create_test_content("text", text="Starting..."),
        create_test_content("function_call", name="process", arguments={"data": "test"}),
        create_test_content("text", text="Done!"),
    ]
    update = create_test_agent_update(contents)

    events = await mapper.convert_event(update, test_request)

    assert len(events) >= 3

    # 両方のタイプのイベントが存在するべきである
    event_types = {event.type for event in events}
    assert "response.output_text.delta" in event_types
    assert "response.function_call_arguments.delta" in event_types


async def test_unknown_content_fallback(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """未知のコンテンツタイプの優雅な処理をテストする。"""
    # Pydanticのバリデーションにより無効なAgentRunResponseUpdateを作成できないため、
    # フォールバックパスを直接テストする。代わりにcontent mapperの未知コンテンツ処理をテストする。

    class MockUnknownContent:
        def __init__(self):
            self.__class__.__name__ = "WeirdUnknownContent"  # content_mappers に存在しない

    # content mapper を直接テストする
    context = mapper._get_or_create_context(test_request)
    unknown_content = MockUnknownContent()

    # これにより _convert_agent_update の未知コンテンツフォールバックがトリガーされるはずである
    event = await mapper._create_unknown_content_event(unknown_content, context)

    assert event.type == "response.output_text.delta"
    assert "Unknown content type" in event.delta
    assert "WeirdUnknownContent" in event.delta


async def test_agent_run_response_mapping(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """mapperが完全なAgentRunResponse（非ストリーミング）を処理できることをテストします。"""
    from agent_framework import AgentRunResponse, ChatMessage, Role, TextContent

    # agent.run()が返すような完全なレスポンスを作成します。
    message = ChatMessage(
        role=Role.ASSISTANT,
        contents=[TextContent(text="Complete response from run()")],
    )
    response = AgentRunResponse(messages=[message], response_id="test_resp_123")

    # Mapperはこれをストリーミングイベントに変換する必要があります。
    events = await mapper.convert_event(response, test_request)

    assert len(events) > 0
    # テキストデルタイベントを生成する必要があります。
    text_events = [e for e in events if e.type == "response.output_text.delta"]
    assert len(text_events) > 0
    assert text_events[0].delta == "Complete response from run()"


async def test_agent_lifecycle_events(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """agentのライフサイクルイベントがOpenAI形式に正しく変換されることをテストします。"""
    from agent_framework_devui.models._openai_custom import AgentCompletedEvent, AgentFailedEvent, AgentStartedEvent

    # AgentStartedEventをテストします。
    start_event = AgentStartedEvent()
    events = await mapper.convert_event(start_event, test_request)

    assert len(events) == 2  # response.createdとresponse.in_progressを発行する必要があります。
    assert events[0].type == "response.created"
    assert events[1].type == "response.in_progress"
    assert events[0].response.model == "test_agent"  # リクエストからモデルを使用する必要があります。
    assert events[0].response.status == "in_progress"

    # AgentCompletedEventをテストします。
    complete_event = AgentCompletedEvent()
    events = await mapper.convert_event(complete_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert events[0].response.status == "completed"

    # AgentFailedEventをテストします。
    error = Exception("Test error")
    failed_event = AgentFailedEvent(error=error)
    events = await mapper.convert_event(failed_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.failed"
    assert events[0].response.status == "failed"
    assert events[0].response.error.message == "Test error"
    assert events[0].response.error.code == "server_error"


@pytest.mark.skip(reason="Workflow events need real classes from agent_framework.workflows")
async def test_workflow_lifecycle_events(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """ワークフローのライフサイクルイベントがOpenAI形式に正しく変換されることをテストします。"""

    # モックのワークフローイベントを作成します（テストで実際のものにアクセスできないため）。
    class WorkflowStartedEvent:  # noqa: B903
        def __init__(self, workflow_id: str):
            self.workflow_id = workflow_id

    class WorkflowCompletedEvent:  # noqa: B903
        def __init__(self, workflow_id: str):
            self.workflow_id = workflow_id

    class WorkflowFailedEvent:  # noqa: B903
        def __init__(self, workflow_id: str, error_info: dict | None = None):
            self.workflow_id = workflow_id
            self.error_info = error_info

    # WorkflowStartedEventをテストします。
    start_event = WorkflowStartedEvent(workflow_id="test_workflow_123")
    events = await mapper.convert_event(start_event, test_request)

    assert len(events) == 2  # response.createdとresponse.in_progressを発行する必要があります。
    assert events[0].type == "response.created"
    assert events[1].type == "response.in_progress"
    assert events[0].response.model == "test_agent"  # リクエストからモデルを使用する必要があります。
    assert events[0].response.status == "in_progress"

    # WorkflowCompletedEventをテストします。
    complete_event = WorkflowCompletedEvent(workflow_id="test_workflow_123")
    events = await mapper.convert_event(complete_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert events[0].response.status == "completed"

    # エラー情報を含むWorkflowFailedEventをテストします。
    failed_event = WorkflowFailedEvent(workflow_id="test_workflow_123", error_info={"message": "Workflow failed"})
    events = await mapper.convert_event(failed_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.failed"
    assert events[0].response.status == "failed"
    assert events[0].response.error.message == "{'message': 'Workflow failed'}"
    assert events[0].response.error.code == "server_error"


@pytest.mark.skip(reason="Executor events need real classes from agent_framework.workflows")
async def test_executor_action_events(mapper: MessageMapper, test_request: AgentFrameworkRequest) -> None:
    """ワークフローのexecutorイベントがカスタムの出力アイテムイベントに正しく変換されることをテストします。"""

    # モックのexecutorイベントを作成します（テストで実際のものにアクセスできないため）。
    class ExecutorInvokedEvent:  # noqa: B903
        def __init__(self, executor_id: str, executor_type: str = "test"):
            self.executor_id = executor_id
            self.executor_type = executor_type

    class ExecutorCompletedEvent:  # noqa: B903
        def __init__(self, executor_id: str, result: Any = None):
            self.executor_id = executor_id
            self.result = result

    class ExecutorFailedEvent:  # noqa: B903
        def __init__(self, executor_id: str, error: Exception | None = None):
            self.executor_id = executor_id
            self.error = error

    # ExecutorInvokedEventをテストします。
    invoked_event = ExecutorInvokedEvent(executor_id="exec_123", executor_type="test_executor")
    events = await mapper.convert_event(invoked_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.output_item.added"
    assert events[0].item["type"] == "executor_action"
    assert events[0].item["executor_id"] == "exec_123"
    assert events[0].item["status"] == "in_progress"

    # ExecutorCompletedEventをテストします。
    complete_event = ExecutorCompletedEvent(executor_id="exec_123", result={"data": "success"})
    events = await mapper.convert_event(complete_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.output_item.done"
    assert events[0].item["type"] == "executor_action"
    assert events[0].item["executor_id"] == "exec_123"
    assert events[0].item["status"] == "completed"
    assert events[0].item["result"] == {"data": "success"}

    # ExecutorFailedEventをテストします。
    failed_event = ExecutorFailedEvent(executor_id="exec_123", error=Exception("Executor failed"))
    events = await mapper.convert_event(failed_event, test_request)

    assert len(events) == 1
    assert events[0].type == "response.output_item.done"
    assert events[0].item["type"] == "executor_action"
    assert events[0].item["executor_id"] == "exec_123"
    assert events[0].item["status"] == "failed"
    assert "Executor failed" in str(events[0].item["error"]["message"])


if __name__ == "__main__":
    # シンプルなテストランナー。
    async def run_all_tests() -> None:
        mapper = MessageMapper()
        test_request = AgentFrameworkRequest(
            model="test",
            input="Test",
            stream=True,
        )

        tests = [
            ("Critical isinstance bug detection", test_critical_isinstance_bug_detection),
            ("Text content mapping", test_text_content_mapping),
            ("Function call mapping", test_function_call_mapping),
            ("Error content mapping", test_error_content_mapping),
            ("Mixed content types", test_mixed_content_types),
            ("Unknown content fallback", test_unknown_content_fallback),
        ]

        passed = 0
        for _test_name, test_func in tests:
            try:
                await test_func(mapper, test_request)
                passed += 1
            except Exception:
                pass

    asyncio.run(run_all_tests())
