# Copyright (c) Microsoft. All rights reserved.

"""実行フロー機能に特化したテスト。"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from agent_framework_devui._discovery import EntityDiscovery
from agent_framework_devui._executor import AgentFrameworkExecutor, EntityNotFoundError
from agent_framework_devui._mapper import MessageMapper
from agent_framework_devui.models._openai_custom import AgentFrameworkRequest


class _DummyStartExecutor:
    """テスト用にハンドラメタデータを公開する最小限のexecutorスタブ。"""

    def __init__(self, *, input_types=None, handlers=None):
        if input_types is not None:
            self.input_types = list(input_types)
        if handlers is not None:
            self._handlers = dict(handlers)


class _DummyWorkflow:
    """設定された開始executorを返すシンプルなワークフロースタブ。"""

    def __init__(self, start_executor):
        self._start_executor = start_executor

    def get_start_executor(self):
        return self._start_executor


@pytest.fixture
def test_entities_dir():
    """適切なエンティティ構造を持つsamplesディレクトリを使用する。"""
    # メインのpython samplesフォルダからsamplesディレクトリを取得する
    current_dir = Path(__file__).parent
    # python/samples/getting_started/devui に移動する
    samples_dir = current_dir.parent.parent.parent / "samples" / "getting_started" / "devui"
    return str(samples_dir.resolve())


@pytest.fixture
async def executor(test_entities_dir):
    """設定されたexecutorを作成する。"""
    discovery = EntityDiscovery(test_entities_dir)
    mapper = MessageMapper()
    executor = AgentFrameworkExecutor(discovery, mapper)

    # エンティティをディスカバリする
    await executor.discover_entities()

    return executor


async def test_executor_entity_discovery(executor):
    """executorエンティティのディスカバリをテストする。"""
    entities = await executor.discover_entities()

    # samplesディレクトリからエンティティを見つけるべきである
    assert len(entities) > 0, "Should discover at least one entity"

    entity_types = [e.type for e in entities]
    assert "agent" in entity_types, "Should find at least one agent"
    assert "workflow" in entity_types, "Should find at least one workflow"

    # エンティティ構造をテストする
    for entity in entities:
        assert entity.id, "Entity should have an ID"
        assert entity.name, "Entity should have a name"
        # `__init__.py` ファイルのみを持つエンティティは、モジュールがレイジーロード中にインポートされるまでタイプを決定できない。これが
        # 'unknown' タイプが存在する理由である。
        assert entity.type in ["agent", "workflow", "unknown"], (
            "Entity should have valid type (unknown allowed during discovery phase)"
        )


async def test_executor_get_entity_info(executor):
    """IDによるエンティティ情報取得をテストする。"""
    entities = await executor.discover_entities()
    entity_id = entities[0].id

    entity_info = executor.get_entity_info(entity_id)
    assert entity_info is not None
    assert entity_info.id == entity_id
    assert entity_info.type in ["agent", "workflow", "unknown"]


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires OpenAI API key")
async def test_executor_sync_execution(executor):
    """同期実行をテストする。"""
    entities = await executor.discover_entities()
    # テスト用のAgentエンティティを見つける
    agents = [e for e in entities if e.type == "agent"]
    assert len(agents) > 0, "No agent entities found for testing"
    agent_id = agents[0].id

    # 簡略化されたルーティングを使用：model = entity_id
    request = AgentFrameworkRequest(
        model=agent_id,  # Model IS the entity_id
        input="test data",
        stream=False,
    )

    response = await executor.execute_sync(request)

    # 簡略化されたルーティングでは、response.model は実際のagent_idを反映する
    assert response.model == agent_id
    assert response.object == "response"
    assert len(response.output) > 0
    assert response.usage.total_tokens > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires OpenAI API key")
@pytest.mark.skip("Skipping while we fix discovery")
async def test_executor_streaming_execution(executor):
    """ストリーミング実行をテストする。"""
    entities = await executor.discover_entities()
    # テスト用のAgentエンティティを見つける
    agents = [e for e in entities if e.type == "agent"]
    assert len(agents) > 0, "No agent entities found for testing"
    agent_id = agents[0].id

    # 簡略化されたルーティングを使用：model = entity_id
    request = AgentFrameworkRequest(
        model=agent_id,  # Model IS the entity_id
        input="streaming test",
        stream=True,
    )

    event_count = 0
    text_events = []

    async for event in executor.execute_streaming(request):
        event_count += 1
        if hasattr(event, "type") and event.type == "response.output_text.delta":
            text_events.append(event.delta)

        if event_count > 10:  # Limit for testing
            break

    assert event_count > 0
    assert len(text_events) > 0


async def test_executor_invalid_entity_id(executor):
    """無効なエンティティIDでの実行をテストする。"""
    with pytest.raises(EntityNotFoundError):
        executor.get_entity_info("nonexistent_agent")


async def test_executor_missing_entity_id(executor):
    """get_entity_id が model フィールドを返すことをテストする（簡略化ルーティング）。"""
    request = AgentFrameworkRequest(
        model="my_agent",
        input="test",
        stream=False,
    )

    # 簡略化されたルーティングでは、model フィールドが entity_id である
    entity_id = request.get_entity_id()
    assert entity_id == "my_agent"


def test_executor_get_start_executor_message_types_uses_handlers():
    """input_types が欠落している場合にハンドラメタデータが表面化されることを保証する。"""
    executor = AgentFrameworkExecutor(EntityDiscovery(None), MessageMapper())
    start_executor = _DummyStartExecutor(handlers={str: lambda *_: None})
    workflow = _DummyWorkflow(start_executor)

    start, message_types = executor._get_start_executor_message_types(workflow)

    assert start is start_executor
    assert str in message_types


def test_executor_select_primary_input_prefers_string():
    """他のハンドラの後にディスカバリされた場合でも文字列入力を選択する。"""
    from agent_framework_devui._utils import select_primary_input_type

    placeholder_type = type("Placeholder", (), {})

    chosen = select_primary_input_type([placeholder_type, str])

    assert chosen is str


def test_executor_parse_structured_prefers_input_field():
    """構造化ペイロードは、Agentの開始がテキストを要求する場合に文字列にマップされる。"""
    executor = AgentFrameworkExecutor(EntityDiscovery(None), MessageMapper())
    start_executor = _DummyStartExecutor(handlers={type("Req", (), {}): None, str: lambda *_: None})
    workflow = _DummyWorkflow(start_executor)

    parsed = executor._parse_structured_workflow_input(workflow, {"input": "hello"})

    assert parsed == "hello"


def test_executor_parse_raw_falls_back_to_string():
    """開始executorがテキストを期待する場合、生の入力は変更されない。"""
    executor = AgentFrameworkExecutor(EntityDiscovery(None), MessageMapper())
    start_executor = _DummyStartExecutor(handlers={str: lambda *_: None})
    workflow = _DummyWorkflow(start_executor)

    parsed = executor._parse_raw_workflow_input(workflow, "hi there")

    assert parsed == "hi there"


async def test_executor_handles_non_streaming_agent():
    """run() メソッドのみを持つAgentをexecutorが処理できることをテストする（run_streamなし）。"""
    from agent_framework import AgentRunResponse, AgentThread, ChatMessage, Role, TextContent

    class NonStreamingAgent:
        """run() メソッドのみを持つAgent - 完全なAgentProtocolを満たさない。"""

        id = "non_streaming_test"
        name = "Non-Streaming Test Agent"
        description = "Test agent without run_stream()"

        @property
        def display_name(self):
            return self.name

        async def run(self, messages=None, *, thread=None, **kwargs):
            return AgentRunResponse(
                messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=f"Processed: {messages}")])],
                response_id="test_123",
            )

        def get_new_thread(self, **kwargs):
            return AgentThread()

    # executorを作成しAgentを登録する
    discovery = EntityDiscovery(None)
    mapper = MessageMapper()
    executor = AgentFrameworkExecutor(discovery, mapper)

    agent = NonStreamingAgent()
    entity_info = await discovery.create_entity_info_from_object(agent, source="test")
    discovery.register_entity(entity_info.id, entity_info, agent)

    # 非ストリーミングAgentを実行する（簡略化ルーティングを使用）
    request = AgentFrameworkRequest(
        model=entity_info.id,  # Model IS the entity_id
        input="hello",
        stream=True,  # DevUI always streams
    )

    events = []
    async for event in executor.execute_streaming(request):
        events.append(event)

    # Agentがストリームしなくてもイベントを受け取るべきである
    assert len(events) > 0
    text_events = [e for e in events if hasattr(e, "type") and e.type == "response.output_text.delta"]
    assert len(text_events) > 0
    assert "Processed: hello" in text_events[0].delta


if __name__ == "__main__":
    # シンプルなテストランナー
    async def run_tests():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # テスト用Agentを作成する
            agent_file = temp_path / "streaming_agent.py"
            agent_file.write_text("""
class StreamingAgent:
    name = "Streaming Test Agent"
    description = "Test agent for streaming"

    async def run_stream(self, input_str):
        for i, word in enumerate(f"Processing {input_str}".split()):
            yield f"word_{i}: {word} "
""")

            discovery = EntityDiscovery(str(temp_path))
            mapper = MessageMapper()
            executor = AgentFrameworkExecutor(discovery, mapper)

            # ディスカバリをテストする
            entities = await executor.discover_entities()

            if entities:
                # 同期実行をテストする（簡略化ルーティングを使用）
                request = AgentFrameworkRequest(
                    model=entities[0].id,  # Model IS the entity_id
                    input="test input",
                    stream=False,
                )

                await executor.execute_sync(request)

                # ストリーミング実行をテストする
                request.stream = True
                event_count = 0
                async for _event in executor.execute_streaming(request):
                    event_count += 1
                    if event_count > 5:  # Limit for testing
                        break

    asyncio.run(run_tests())
