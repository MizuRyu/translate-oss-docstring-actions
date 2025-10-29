# Copyright (c) Microsoft. All rights reserved.

"""サーバー機能に特化したテスト。"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from agent_framework_devui import DevServer
from agent_framework_devui._utils import extract_executor_message_types, select_primary_input_type
from agent_framework_devui.models._openai_custom import AgentFrameworkRequest


class _StubExecutor:
    """ハンドラメタデータを公開するシンプルなexecutorスタブ。"""

    def __init__(self, *, input_types=None, handlers=None):
        if input_types is not None:
            self.input_types = list(input_types)
        if handlers is not None:
            self._handlers = dict(handlers)


@pytest.fixture
def test_entities_dir():
    """適切なエンティティ構造を持つsamplesディレクトリを使用します。"""
    # メインのpython samplesフォルダからsamplesディレクトリを取得します。
    current_dir = Path(__file__).parent
    # python/samples/getting_started/devuiに移動します。
    samples_dir = current_dir.parent.parent.parent / "samples" / "getting_started" / "devui"
    return str(samples_dir.resolve())


async def test_server_health_endpoint(test_entities_dir):
    """/healthエンドポイントをテストします。"""
    server = DevServer(entities_dir=test_entities_dir)
    executor = await server._ensure_executor()

    # エンティティ数をテストします。
    entities = await executor.discover_entities()
    assert len(entities) > 0
    # フレームワーク名は単一フレームワークに簡素化されたためハードコードされています。


@pytest.mark.skip("Skipping while we fix discovery")
async def test_server_entities_endpoint(test_entities_dir):
    """/v1/entitiesエンドポイントをテストします。"""
    server = DevServer(entities_dir=test_entities_dir)
    executor = await server._ensure_executor()

    entities = await executor.discover_entities()
    assert len(entities) >= 1
    # 少なくともweather agentを見つける必要があります。
    agent_entities = [e for e in entities if e.type == "agent"]
    assert len(agent_entities) >= 1
    agent_names = [e.name for e in agent_entities]
    assert "WeatherAgent" in agent_names


async def test_server_execution_sync(test_entities_dir):
    """同期実行エンドポイントをテストします。"""
    server = DevServer(entities_dir=test_entities_dir)
    executor = await server._ensure_executor()

    entities = await executor.discover_entities()
    agent_id = entities[0].id

    # モデルをentity_idとして使用します（新しい簡素化されたルーティング）。
    request = AgentFrameworkRequest(
        model=agent_id,  # model IS the entity_id now!
        input="San Francisco",
        stream=False,
    )

    response = await executor.execute_sync(request)
    assert response.model == agent_id  # モデル（entity_id）をエコーバックする必要があります。
    assert len(response.output) > 0


async def test_server_execution_streaming(test_entities_dir):
    """ストリーミング実行エンドポイントをテストします。"""
    server = DevServer(entities_dir=test_entities_dir)
    executor = await server._ensure_executor()

    entities = await executor.discover_entities()
    agent_id = entities[0].id

    # モデルをentity_idとして使用します（新しい簡素化されたルーティング）。
    request = AgentFrameworkRequest(
        model=agent_id,  # model IS the entity_id now!
        input="New York",
        stream=True,
    )

    event_count = 0
    async for _event in executor.execute_streaming(request):
        event_count += 1
        if event_count > 5:  # Limit for testing
            break

    assert event_count > 0


def test_configuration():
    """基本的な設定をテストします。"""
    server = DevServer(entities_dir="test", port=9000, host="localhost")
    assert server.port == 9000
    assert server.host == "localhost"
    assert server.entities_dir == "test"
    assert server.cors_origins == ["*"]
    assert server.ui_enabled


def test_extract_executor_message_types_prefers_input_types():
    """利用可能な場合はinput typesプロパティが使用されます。"""
    stub = _StubExecutor(input_types=[str, dict])

    types = extract_executor_message_types(stub)

    assert types == [str, dict]


def test_extract_executor_message_types_falls_back_to_handlers():
    """input_typesがない場合、ハンドラはメッセージメタデータを提供します。"""
    stub = _StubExecutor(handlers={str: object(), int: object()})

    types = extract_executor_message_types(stub)

    assert str in types
    assert int in types


def test_select_primary_input_type_prefers_string_and_dict():
    """プライマリタイプの選択はユーザーフレンドリーなプリミティブを優先します。"""
    string_first = select_primary_input_type([dict[str, str], str])
    dict_first = select_primary_input_type([dict[str, str]])
    fallback = select_primary_input_type([int, float])

    assert string_first is str
    assert dict_first is dict
    assert fallback is int


@pytest.mark.asyncio
async def test_credential_cleanup() -> None:
    """サーバークリーンアップ中に非同期資格情報が正しくクローズされることをテストします。"""
    from unittest.mock import AsyncMock, Mock

    from agent_framework import ChatAgent

    # 非同期closeを持つモック資格情報を作成します。
    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock()

    # 資格情報を持つモックチャットクライアントを作成します。
    mock_client = Mock()
    mock_client.async_credential = mock_credential
    mock_client.model_id = "test-model"

    # モッククライアントでagentを作成します。
    agent = ChatAgent(name="TestAgent", chat_client=mock_client, instructions="Test agent")

    # agentでDevUIサーバーを作成します。
    server = DevServer()
    server._pending_entities = [agent]
    await server._ensure_executor()

    # クリーンアップを実行します。
    await server._cleanup_entities()

    # credential.close()が呼び出されたことを検証します。
    assert mock_credential.close.called, "Async credential close should have been called"
    assert mock_credential.close.call_count == 1


@pytest.mark.asyncio
async def test_credential_cleanup_error_handling() -> None:
    """資格情報クリーンアップエラーが適切に処理されることをテストします。"""
    from unittest.mock import AsyncMock, Mock

    from agent_framework import ChatAgent

    # close時にエラーを発生させるモック資格情報を作成します。
    mock_credential = AsyncMock()
    mock_credential.close = AsyncMock(side_effect=Exception("Close failed"))

    # 資格情報を持つモックチャットクライアントを作成します。
    mock_client = Mock()
    mock_client.async_credential = mock_credential
    mock_client.model_id = "test-model"

    # モッククライアントでagentを作成します。
    agent = ChatAgent(name="TestAgent", chat_client=mock_client, instructions="Test agent")

    # agentでDevUIサーバーを作成します。
    server = DevServer()
    server._pending_entities = [agent]
    await server._ensure_executor()

    # クリーンアップを実行します - 資格情報エラーがあっても例外は発生しません。
    await server._cleanup_entities()

    # closeが試みられたことを検証します。
    assert mock_credential.close.called


@pytest.mark.asyncio
async def test_multiple_credential_attributes() -> None:
    """一般的な資格情報属性名をすべてチェックすることをテストします。"""
    from unittest.mock import AsyncMock, Mock

    from agent_framework import ChatAgent

    # モック資格情報を作成します。
    mock_cred1 = Mock()
    mock_cred1.close = Mock()
    mock_cred2 = AsyncMock()
    mock_cred2.close = AsyncMock()

    # 複数の資格情報属性を持つモックチャットクライアントを作成します。
    mock_client = Mock()
    mock_client.credential = mock_cred1
    mock_client.async_credential = mock_cred2
    mock_client.model_id = "test-model"

    # モッククライアントでagentを作成します。
    agent = ChatAgent(name="TestAgent", chat_client=mock_client, instructions="Test agent")

    # agentでDevUIサーバーを作成します。
    server = DevServer()
    server._pending_entities = [agent]
    await server._ensure_executor()

    # クリーンアップを実行します。
    await server._cleanup_entities()

    # 両方の資格情報がクローズされたことを検証します。
    assert mock_cred1.close.called, "Sync credential should be closed"
    assert mock_cred2.close.called, "Async credential should be closed"


if __name__ == "__main__":
    # シンプルなテストランナー。
    async def run_tests():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # テスト用agentを作成します。
            agent_file = temp_path / "weather_agent.py"
            agent_file.write_text("""
class WeatherAgent:
    name = "Weather Agent"
    description = "Gets weather information"

    def run_stream(self, input_str):
        return f"Weather in {input_str} is sunny"
""")

            server = DevServer(entities_dir=str(temp_path))
            executor = await server._ensure_executor()

            entities = await executor.discover_entities()

            if entities:
                request = AgentFrameworkRequest(
                    model=entities[0].id,  # model IS the entity_id now!
                    input="test location",
                    stream=False,
                )

                await executor.execute_sync(request)

    asyncio.run(run_tests())
