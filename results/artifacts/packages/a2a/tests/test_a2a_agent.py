# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from a2a.types import Artifact, DataPart, FilePart, FileWithUri, Message, Part, Task, TaskState, TaskStatus, TextPart
from a2a.types import Role as A2ARole
from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    ChatMessage,
    DataContent,
    ErrorContent,
    HostedFileContent,
    Role,
    TextContent,
    UriContent,
)
from agent_framework.a2a import A2AAgent
from pytest import fixture, raises

from agent_framework_a2a._agent import _get_uri_data  # type: ignore


class MockA2AClient:
    """テスト用のA2A Clientのモック実装です。"""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.responses: list[Any] = []

    def add_message_response(self, message_id: str, text: str, role: str = "agent") -> None:
        """モックのMessageレスポンスを追加します。"""

        # 実際のTextPartインスタンスを作成し、Partでラップします。
        text_part = Part(root=TextPart(text=text))

        # 実際のMessageインスタンスを作成します。
        message = Message(
            message_id=message_id, role=A2ARole.agent if role == "agent" else A2ARole.user, parts=[text_part]
        )
        self.responses.append(message)

    def add_task_response(self, task_id: str, artifacts: list[dict[str, Any]]) -> None:
        """モックのTaskレスポンスを追加します。"""
        # モックのアーティファクトを作成します。
        mock_artifacts = []
        for artifact_data in artifacts:
            # 実際のTextPartインスタンスを作成し、Partでラップします。
            text_part = Part(root=TextPart(text=artifact_data.get("content", "Test content")))

            artifact = Artifact(
                artifact_id=artifact_data.get("id", str(uuid4())),
                name=artifact_data.get("name", "test-artifact"),
                description=artifact_data.get("description", "Test artifact"),
                parts=[text_part],
            )
            mock_artifacts.append(artifact)

        # タスクのステータスを作成します。
        status = TaskStatus(state=TaskState.completed, message=None)

        # 実際のTaskインスタンスを作成します。
        task = Task(
            id=task_id, context_id="test-context", status=status, artifacts=mock_artifacts if mock_artifacts else None
        )

        # ClientEventのタプル形式をモックします。
        update_event = None  # 完了したタスクに特定の更新イベントはありません。
        client_event = (task, update_event)
        self.responses.append(client_event)

    async def send_message(self, message: Any) -> AsyncIterator[Any]:
        """レスポンスをyieldするsend_messageメソッドのモックです。"""
        self.call_count += 1

        if self.responses:
            response = self.responses.pop(0)
            yield response


@fixture
def mock_a2a_client() -> MockA2AClient:
    """モックのA2Aクライアントを提供するフィクスチャです。"""
    return MockA2AClient()


@fixture
def a2a_agent(mock_a2a_client: MockA2AClient) -> A2AAgent:
    """モッククライアントを持つA2AAgentを提供するフィクスチャです。"""
    return A2AAgent(name="Test Agent", id="test-agent", client=mock_a2a_client, http_client=None)


def test_a2a_agent_initialization_with_client(mock_a2a_client: MockA2AClient) -> None:
    """提供されたクライアントでのA2AAgent初期化のテストです。"""
    # mockオブジェクトのためにPydanticの検証を回避するためにmodel_constructを使用します。
    agent = A2AAgent(
        name="Test Agent", id="test-agent-123", description="A test agent", client=mock_a2a_client, http_client=None
    )

    assert agent.name == "Test Agent"
    assert agent.id == "test-agent-123"
    assert agent.description == "A test agent"
    assert agent.client == mock_a2a_client


def test_a2a_agent_initialization_without_client_raises_error() -> None:
    """クライアントまたはURLがない場合のA2AAgent初期化がValueErrorを発生させるテストです。"""
    with raises(ValueError, match="Either agent_card or url must be provided"):
        A2AAgent(name="Test Agent")


async def test_run_with_message_response(a2a_agent: A2AAgent, mock_a2a_client: MockA2AClient) -> None:
    """即時Messageレスポンスを持つrun()メソッドのテストです。"""
    mock_a2a_client.add_message_response("msg-123", "Hello from agent!", "agent")

    response = await a2a_agent.run("Hello agent")

    assert isinstance(response, AgentRunResponse)
    assert len(response.messages) == 1
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[0].text == "Hello from agent!"
    assert response.response_id == "msg-123"
    assert mock_a2a_client.call_count == 1


async def test_run_with_task_response_single_artifact(a2a_agent: A2AAgent, mock_a2a_client: MockA2AClient) -> None:
    """単一のアーティファクトを含むTaskレスポンスを持つrun()メソッドのテストです。"""
    artifacts = [{"id": "art-1", "content": "Generated report content"}]
    mock_a2a_client.add_task_response("task-456", artifacts)

    response = await a2a_agent.run("Generate a report")

    assert isinstance(response, AgentRunResponse)
    assert len(response.messages) == 1
    assert response.messages[0].role == Role.ASSISTANT
    assert response.messages[0].text == "Generated report content"
    assert response.response_id == "task-456"
    assert mock_a2a_client.call_count == 1


async def test_run_with_task_response_multiple_artifacts(a2a_agent: A2AAgent, mock_a2a_client: MockA2AClient) -> None:
    """複数のアーティファクトを含むTaskレスポンスを持つrun()メソッドのテストです。"""
    artifacts = [
        {"id": "art-1", "content": "First artifact content"},
        {"id": "art-2", "content": "Second artifact content"},
        {"id": "art-3", "content": "Third artifact content"},
    ]
    mock_a2a_client.add_task_response("task-789", artifacts)

    response = await a2a_agent.run("Generate multiple outputs")

    assert isinstance(response, AgentRunResponse)
    assert len(response.messages) == 3

    assert response.messages[0].text == "First artifact content"
    assert response.messages[1].text == "Second artifact content"
    assert response.messages[2].text == "Third artifact content"

    # すべてアシスタントメッセージであるべきです。
    for message in response.messages:
        assert message.role == Role.ASSISTANT

    assert response.response_id == "task-789"


async def test_run_with_task_response_no_artifacts(a2a_agent: A2AAgent, mock_a2a_client: MockA2AClient) -> None:
    """アーティファクトを含まないTaskレスポンスを持つrun()メソッドのテストです。"""
    mock_a2a_client.add_task_response("task-empty", [])

    response = await a2a_agent.run("Do something with no output")

    assert isinstance(response, AgentRunResponse)
    assert response.response_id == "task-empty"


async def test_run_with_unknown_response_type_raises_error(a2a_agent: A2AAgent, mock_a2a_client: MockA2AClient) -> None:
    """不明なレスポンスタイプでrun()メソッドがNotImplementedErrorを発生させるテストです。"""
    mock_a2a_client.responses.append("invalid_response")

    with raises(NotImplementedError, match="Only Message and Task responses are supported"):
        await a2a_agent.run("Test message")


def test_task_to_chat_messages_empty_artifacts(a2a_agent: A2AAgent) -> None:
    """アーティファクトを含まないタスクで_task_to_chat_messagesのテストです。"""
    task = MagicMock()
    task.artifacts = None

    result = a2a_agent._task_to_chat_messages(task)

    assert len(result) == 0


def test_task_to_chat_messages_with_artifacts(a2a_agent: A2AAgent) -> None:
    """アーティファクトを含むタスクで_task_to_chat_messagesのテストです。"""
    task = MagicMock()

    # モックのアーティファクトを作成します。
    artifact1 = MagicMock()
    artifact1.artifact_id = "art-1"
    text_part1 = MagicMock()
    text_part1.root = MagicMock()
    text_part1.root.kind = "text"
    text_part1.root.text = "Content 1"
    text_part1.root.metadata = None
    artifact1.parts = [text_part1]

    artifact2 = MagicMock()
    artifact2.artifact_id = "art-2"
    text_part2 = MagicMock()
    text_part2.root = MagicMock()
    text_part2.root.kind = "text"
    text_part2.root.text = "Content 2"
    text_part2.root.metadata = None
    artifact2.parts = [text_part2]

    task.artifacts = [artifact1, artifact2]

    result = a2a_agent._task_to_chat_messages(task)

    assert len(result) == 2
    assert result[0].text == "Content 1"
    assert result[1].text == "Content 2"
    assert all(msg.role == Role.ASSISTANT for msg in result)


def test_artifact_to_chat_message(a2a_agent: A2AAgent) -> None:
    """_artifact_to_chat_message変換のテストです。"""
    artifact = MagicMock()
    artifact.artifact_id = "test-artifact"

    text_part = MagicMock()
    text_part.root = MagicMock()
    text_part.root.kind = "text"
    text_part.root.text = "Artifact content"
    text_part.root.metadata = None

    artifact.parts = [text_part]

    result = a2a_agent._artifact_to_chat_message(artifact)

    assert isinstance(result, ChatMessage)
    assert result.role == Role.ASSISTANT
    assert result.text == "Artifact content"
    assert result.raw_representation == artifact


def test_get_uri_data_valid_uri() -> None:
    """有効なデータURIで_get_uri_dataのテストです。"""

    uri = "data:application/json;base64,eyJ0ZXN0IjoidmFsdWUifQ=="
    result = _get_uri_data(uri)
    assert result == "eyJ0ZXN0IjoidmFsdWUifQ=="


def test_get_uri_data_invalid_uri() -> None:
    """無効なURI形式で_get_uri_dataのテストです。"""

    with raises(ValueError, match="Invalid data URI format"):
        _get_uri_data("not-a-valid-data-uri")


def test_a2a_parts_to_contents_conversion(a2a_agent: A2AAgent) -> None:
    """A2Aパーツからcontentsへの変換のテストです。"""

    agent = A2AAgent(name="Test Agent", client=MockA2AClient(), _http_client=None)

    # A2Aパーツを作成します。
    parts = [Part(root=TextPart(text="First part")), Part(root=TextPart(text="Second part"))]

    # contentsに変換します。
    contents = agent._a2a_parts_to_contents(parts)

    # 変換を検証します。
    assert len(contents) == 2
    assert isinstance(contents[0], TextContent)
    assert isinstance(contents[1], TextContent)
    assert contents[0].text == "First part"
    assert contents[1].text == "Second part"


def test_chat_message_to_a2a_message_with_error_content(a2a_agent: A2AAgent) -> None:
    """ErrorContentを持つ_chat_message_to_a2a_messageのテストです。"""

    # ErrorContentを持つChatMessageを作成します。
    error_content = ErrorContent(message="Test error message")
    message = ChatMessage(role=Role.USER, contents=[error_content])

    # A2Aメッセージに変換します。
    a2a_message = a2a_agent._chat_message_to_a2a_message(message)

    # 変換を検証します。
    assert len(a2a_message.parts) == 1
    assert a2a_message.parts[0].root.text == "Test error message"


def test_chat_message_to_a2a_message_with_uri_content(a2a_agent: A2AAgent) -> None:
    """UriContentを持つ_chat_message_to_a2a_messageのテストです。"""

    # UriContentを持つChatMessageを作成します。
    uri_content = UriContent(uri="http://example.com/file.pdf", media_type="application/pdf")
    message = ChatMessage(role=Role.USER, contents=[uri_content])

    # A2Aメッセージに変換します。
    a2a_message = a2a_agent._chat_message_to_a2a_message(message)

    # 変換を検証します。
    assert len(a2a_message.parts) == 1
    assert a2a_message.parts[0].root.file.uri == "http://example.com/file.pdf"
    assert a2a_message.parts[0].root.file.mime_type == "application/pdf"


def test_chat_message_to_a2a_message_with_data_content(a2a_agent: A2AAgent) -> None:
    """DataContentを持つ_chat_message_to_a2a_messageのテストです。"""

    # DataContent（base64データURI）を持つChatMessageを作成します。
    data_content = DataContent(uri="data:text/plain;base64,SGVsbG8gV29ybGQ=", media_type="text/plain")
    message = ChatMessage(role=Role.USER, contents=[data_content])

    # A2Aメッセージに変換します。
    a2a_message = a2a_agent._chat_message_to_a2a_message(message)

    # 変換を検証します。
    assert len(a2a_message.parts) == 1
    assert a2a_message.parts[0].root.file.bytes == "SGVsbG8gV29ybGQ="
    assert a2a_message.parts[0].root.file.mime_type == "text/plain"


def test_chat_message_to_a2a_message_empty_contents_raises_error(a2a_agent: A2AAgent) -> None:
    """空のcontentsで_chat_message_to_a2a_messageがValueErrorを発生させるテストです。"""
    # contentsがないChatMessageを作成します。
    message = ChatMessage(role=Role.USER, contents=[])

    # 空のcontentsでValueErrorが発生するはずです。
    with raises(ValueError, match="ChatMessage.contents is empty"):
        a2a_agent._chat_message_to_a2a_message(message)


async def test_run_stream_with_message_response(a2a_agent: A2AAgent, mock_a2a_client: MockA2AClient) -> None:
    """即時Messageレスポンスを持つrun_stream()メソッドのテストです。"""
    mock_a2a_client.add_message_response("msg-stream-123", "Streaming response from agent!", "agent")

    # ストリーミング更新を収集します。
    updates: list[AgentRunResponseUpdate] = []
    async for update in a2a_agent.run_stream("Hello agent"):
        updates.append(update)

    # ストリーミングレスポンスを検証します。
    assert len(updates) == 1
    assert isinstance(updates[0], AgentRunResponseUpdate)
    assert updates[0].role == Role.ASSISTANT
    assert len(updates[0].contents) == 1

    content = updates[0].contents[0]
    assert isinstance(content, TextContent)
    assert content.text == "Streaming response from agent!"

    assert updates[0].response_id == "msg-stream-123"
    assert mock_a2a_client.call_count == 1


async def test_context_manager_cleanup() -> None:
    """httpクライアントのコンテキストマネージャクリーンアップのテストです。"""

    # aclose呼び出しを追跡するモックhttpクライアントを作成します。
    mock_http_client = AsyncMock()
    mock_a2a_client = MagicMock()

    agent = A2AAgent(client=mock_a2a_client)
    agent._http_client = mock_http_client

    # コンテキストマネージャのクリーンアップをテストする
    async with agent:
        pass

    # aclose が呼び出されたことを検証する
    mock_http_client.aclose.assert_called_once()


async def test_context_manager_no_cleanup_when_no_http_client() -> None:
    """_http_client が None の場合のコンテキストマネージャをテストする。"""

    mock_a2a_client = MagicMock()

    agent = A2AAgent(client=mock_a2a_client, _http_client=None)

    # これはエラーを発生させないはずです
    async with agent:
        pass


def test_chat_message_to_a2a_message_with_multiple_contents() -> None:
    """複数のコンテンツを持つ ChatMessage の変換をテストする。"""

    agent = A2AAgent(client=MagicMock(), _http_client=None)

    # 複数のコンテンツタイプを持つメッセージを作成する
    message = ChatMessage(
        role=Role.USER,
        contents=[
            TextContent(text="Here's the analysis:"),
            DataContent(data=b"binary data", media_type="application/octet-stream"),
            UriContent(uri="https://example.com/image.png", media_type="image/png"),
            TextContent(text='{"structured": "data"}'),
        ],
    )

    result = agent._chat_message_to_a2a_message(message)

    # すべての4つのコンテンツが parts に変換されているはずです
    assert len(result.parts) == 4

    # 各 part のタイプをチェックする
    assert result.parts[0].root.kind == "text"  # 通常のテキスト
    assert result.parts[1].root.kind == "file"  # バイナリデータ
    assert result.parts[2].root.kind == "file"  # URI コンテンツ
    assert result.parts[3].root.kind == "text"  # JSON テキストはテキストのまま（解析しない）


def test_a2a_parts_to_contents_with_data_part() -> None:
    """A2A DataPart の変換をテストする。"""

    agent = A2AAgent(client=MagicMock(), _http_client=None)

    # DataPart を作成する
    data_part = Part(root=DataPart(data={"key": "value", "number": 42}, metadata={"source": "test"}))

    contents = agent._a2a_parts_to_contents([data_part])

    assert len(contents) == 1

    assert isinstance(contents[0], TextContent)
    assert contents[0].text == '{"key": "value", "number": 42}'
    assert contents[0].additional_properties == {"source": "test"}


def test_a2a_parts_to_contents_unknown_part_kind() -> None:
    """不明な A2A part kind のエラー処理をテストする。"""
    agent = A2AAgent(client=MagicMock(), _http_client=None)

    # 不明な kind を持つモックパートを作成する
    mock_part = MagicMock()
    mock_part.root.kind = "unknown_kind"

    with raises(ValueError, match="Unknown Part kind: unknown_kind"):
        agent._a2a_parts_to_contents([mock_part])


def test_chat_message_to_a2a_message_with_hosted_file() -> None:
    """HostedFileContent を持つ ChatMessage の A2A メッセージへの変換をテストする。"""

    agent = A2AAgent(client=MagicMock(), _http_client=None)

    # ホストされたファイルコンテンツを持つメッセージを作成する
    message = ChatMessage(
        role=Role.USER,
        contents=[HostedFileContent(file_id="hosted://storage/document.pdf")],
    )

    result = agent._chat_message_to_a2a_message(message)  # noqa: SLF001

    # 変換を検証する
    assert len(result.parts) == 1
    part = result.parts[0]
    assert part.root.kind == "file"

    # FileWithUri を持つ FilePart であることを検証する

    assert isinstance(part.root, FilePart)
    assert isinstance(part.root.file, FileWithUri)
    assert part.root.file.uri == "hosted://storage/document.pdf"
    assert part.root.file.mime_type is None  # HostedFileContent は media_type を指定しない


def test_a2a_parts_to_contents_with_hosted_file_uri() -> None:
    """ホストされたファイル URI を持つ A2A FilePart を UriContent に戻す変換をテストする。"""

    agent = A2AAgent(client=MagicMock(), _http_client=None)

    # ホストされたファイル URI を持つ FilePart を作成する（A2A が返すものをシミュレート）
    file_part = Part(
        root=FilePart(
            file=FileWithUri(
                uri="hosted://storage/document.pdf",
                mime_type=None,
            )
        )
    )

    contents = agent._a2a_parts_to_contents([file_part])  # noqa: SLF001

    assert len(contents) == 1

    assert isinstance(contents[0], UriContent)
    assert contents[0].uri == "hosted://storage/document.pdf"
    assert contents[0].media_type == ""  # None を空文字列に変換した


def test_auth_interceptor_parameter() -> None:
    """auth_interceptor パラメータがエラーなく受け入れられることをテストする。"""
    # モックの auth interceptor を作成する
    mock_auth_interceptor = MagicMock()

    # auth_interceptor パラメータで A2AAgent を作成できることをテストする 簡単のため url パラメータを使用
    agent = A2AAgent(
        name="test-agent",
        url="https://test-agent.example.com",
        auth_interceptor=mock_auth_interceptor,
    )

    # エージェントが正常に作成されたことを検証する
    assert agent.name == "test-agent"
    assert agent.client is not None
