# Copyright (c) Microsoft. All rights reserved.

"""エンティティ発見機能に特化したテスト。"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from agent_framework_devui._discovery import EntityDiscovery


@pytest.fixture
def test_entities_dir():
    """適切なエンティティ構造を持つsamplesディレクトリを使用。"""
    # メインのpython samplesフォルダからsamplesディレクトリを取得
    current_dir = Path(__file__).parent
    # python/samples/getting_started/devuiに移動
    samples_dir = current_dir.parent.parent.parent / "samples" / "getting_started" / "devui"
    return str(samples_dir.resolve())


@pytest.mark.skip("Skipping while we fix discovery")
async def test_discover_agents(test_entities_dir):
    """agentの発見が機能し、有効なagentエンティティを返すことをテスト。"""
    discovery = EntityDiscovery(test_entities_dir)
    entities = await discovery.discover_entities()

    agents = [e for e in entities if e.type == "agent"]

    # agentを発見できることをテスト（特定の数ではない）
    assert len(agents) > 0, "Should discover at least one agent"

    # agentの構造/プロパティをテスト
    for agent in agents:
        assert agent.id, "Agent should have an ID"
        assert agent.name, "Agent should have a name"
        assert agent.type == "agent", "Should be identified as agent type"
        assert hasattr(agent, "description"), "Agent should have description attribute"


async def test_discover_workflows(test_entities_dir):
    """workflowの発見が機能し、有効なworkflowエンティティを返すことをテスト。"""
    discovery = EntityDiscovery(test_entities_dir)
    entities = await discovery.discover_entities()

    workflows = [e for e in entities if e.type == "workflow"]

    # workflowを発見できることをテスト（特定の数ではない）
    assert len(workflows) > 0, "Should discover at least one workflow"

    # ワークフローの構造/プロパティをテストする
    for workflow in workflows:
        assert workflow.id, "Workflow should have an ID"
        assert workflow.name, "Workflow should have a name"
        assert workflow.type == "workflow", "Should be identified as workflow type"
        assert hasattr(workflow, "description"), "Workflow should have description attribute"


async def test_empty_directory():
    """空のディレクトリでのディスカバリをテストする。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        discovery = EntityDiscovery(temp_dir)
        entities = await discovery.discover_entities()

        assert len(entities) == 0


async def test_discovery_accepts_agents_with_only_run():
    """run() メソッドのみを持つAgentをディスカバリが受け入れることをテストする。

    レイジーローディングでは、__init__.py のみを持つエンティティがディスカバリされるが、
    ロードされるまで "unknown" タイプとしてマークされる。
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # run() メソッドのみを持つAgentを作成する
        agent_dir = temp_path / "non_streaming_agent"
        agent_dir.mkdir()

        init_file = agent_dir / "__init__.py"
        init_file.write_text("""
from agent_framework import AgentRunResponse, AgentThread, ChatMessage, Role, TextContent

class NonStreamingAgent:
    id = "non_streaming"
    name = "Non-Streaming Agent"
    description = "Agent without run_stream"

    @property
    def display_name(self):
        return self.name

    async def run(self, messages=None, *, thread=None, **kwargs):
        return AgentRunResponse(
            messages=[ChatMessage(
                role=Role.ASSISTANT,
                contents=[TextContent(text="response")]
            )],
            response_id="test"
        )

    def get_new_thread(self, **kwargs):
        return AgentThread()

agent = NonStreamingAgent()
""")

        discovery = EntityDiscovery(str(temp_path))
        entities = await discovery.discover_entities()

        # レイジーローディングでは、エンティティはディスカバリされるがタイプは "unknown" である （agent.py や workflow.py
        # がなくタイプ検出できない）
        assert len(entities) == 1
        entity = entities[0]
        assert entity.id == "non_streaming_agent"
        assert entity.type == "unknown"  # タイプはまだ決定されていない
        assert entity.tools == []  # スパースなメタデータ

        # 完全なメタデータを得るためにレイジーローディングをトリガーする
        agent_obj = await discovery.load_entity(entity.id)
        assert agent_obj is not None

        # ロード後の拡張されたメタデータを確認する
        enriched = discovery.get_entity_info(entity.id)
        assert enriched.type == "agent"  # 正しく識別された
        assert enriched.name == "Non-Streaming Agent"
        assert not enriched.metadata.get("has_run_stream")


async def test_lazy_loading():
    """エンティティがディスカバリ時ではなくオンデマンドでロードされることをテストする。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # テスト用ワークフローを作成する
        workflow_dir = temp_path / "test_workflow"
        workflow_dir.mkdir()
        (workflow_dir / "workflow.py").write_text("""
from agent_framework import WorkflowBuilder, FunctionExecutor

# Create a simple workflow with a start executor
def test_func(input: str) -> str:
    return f"Processed: {input}"

builder = WorkflowBuilder()
executor = FunctionExecutor(id="test_executor", func=test_func)
builder.set_start_executor(executor)
workflow = builder.build()
""")

        discovery = EntityDiscovery(str(temp_path))

        # ディスカバリはモジュールをインポートしてはならない
        entities = await discovery.discover_entities()
        assert len(entities) == 1
        assert entities[0].id == "test_workflow"
        assert entities[0].type == "workflow"  # ファイル名からタイプを検出する
        assert entities[0].tools == []  # スパースなメタデータ（まだロードされていない）

        # エンティティはまだ loaded_objects に存在してはならない
        assert discovery.get_entity_object("test_workflow") is None

        # レイジーロードをトリガーする
        workflow_obj = await discovery.load_entity("test_workflow")
        assert workflow_obj is not None

        # キャッシュに存在するようになった
        assert discovery.get_entity_object("test_workflow") is workflow_obj

        # 2回目のロードは即時（キャッシュから）
        workflow_obj2 = await discovery.load_entity("test_workflow")
        assert workflow_obj2 is workflow_obj  # 同じオブジェクト


async def test_type_detection():
    """エンティティタイプがファイル名から検出されることをテストする。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # workflow.py を持つワークフローを作成する
        workflow_dir = temp_path / "my_workflow"
        workflow_dir.mkdir()
        (workflow_dir / "workflow.py").write_text("""
from agent_framework import WorkflowBuilder, FunctionExecutor

def test_func(input: str) -> str:
    return f"Processed: {input}"

builder = WorkflowBuilder()
executor = FunctionExecutor(id="test_executor", func=test_func)
builder.set_start_executor(executor)
workflow = builder.build()
""")

        # agent.py を持つAgentを作成する
        agent_dir = temp_path / "my_agent"
        agent_dir.mkdir()
        (agent_dir / "agent.py").write_text("""
from agent_framework import AgentRunResponse, AgentThread, ChatMessage, Role, TextContent

class TestAgent:
    name = "Test Agent"

    async def run(self, messages=None, *, thread=None, **kwargs):
        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="test")])],
            response_id="test"
        )

    def get_new_thread(self, **kwargs):
        return AgentThread()

agent = TestAgent()
""")

        # __init__.py のみを持つ曖昧なエンティティを作成する
        unknown_dir = temp_path / "my_thing"
        unknown_dir.mkdir()
        (unknown_dir / "__init__.py").write_text("# thing")

        discovery = EntityDiscovery(str(temp_path))
        entities = await discovery.discover_entities()

        # タイプが正しく検出されていることを確認する
        by_id = {e.id: e for e in entities}

        assert by_id["my_workflow"].type == "workflow"
        assert by_id["my_agent"].type == "agent"
        assert by_id["my_thing"].type == "unknown"


async def test_hot_reload():
    """invalidate_entity() がホットリロードを有効にすることをテストする。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # ワークフローを作成する
        workflow_dir = temp_path / "test_workflow"
        workflow_dir.mkdir()
        workflow_file = workflow_dir / "workflow.py"
        workflow_file.write_text("""
from agent_framework import WorkflowBuilder, FunctionExecutor

def test_func(input: str) -> str:
    return "v1"

builder = WorkflowBuilder()
executor = FunctionExecutor(id="test_executor", func=test_func)
builder.set_start_executor(executor)
workflow = builder.build()
""")

        discovery = EntityDiscovery(str(temp_path))
        await discovery.discover_entities()

        # エンティティをロードする
        workflow1 = await discovery.load_entity("test_workflow")
        assert workflow1 is not None

        # ファイルを変更して異なるワークフローを作成する
        workflow_file.write_text("""
from agent_framework import WorkflowBuilder, FunctionExecutor

def test_func(input: str) -> str:
    return "v2"

def test_func2(input: str) -> str:
    return "v2_extra"

builder = WorkflowBuilder()
executor1 = FunctionExecutor(id="test_executor", func=test_func)
executor2 = FunctionExecutor(id="test_executor2", func=test_func2)
builder.set_start_executor(executor1)
builder.add_edge(executor1, executor2)
workflow = builder.build()
""")

        # 無効化しなければキャッシュされたバージョンを取得する
        workflow2 = await discovery.load_entity("test_workflow")
        assert workflow2 is workflow1  # 同じオブジェクト（キャッシュ）
        # 古いワークフローは1つのexecutorを持つ
        assert len(workflow2.get_executors_list()) == 1

        # キャッシュを無効化する
        discovery.invalidate_entity("test_workflow")

        # 今やディスクからリロードされる
        workflow3 = await discovery.load_entity("test_workflow")
        assert workflow3 is not workflow1  # 異なるオブジェクト
        # 新しいワークフローは2つのexecutorを持つ
        assert len(workflow3.get_executors_list()) == 2


async def test_in_memory_entities_bypass_lazy_loading():
    """インメモリエンティティが以前通り動作することをテストする（レイジーロード不要）。"""
    from agent_framework import FunctionExecutor, WorkflowBuilder

    # インメモリワークフローを作成する
    def test_func(input: str) -> str:
        return f"Processed: {input}"

    builder = WorkflowBuilder()
    executor = FunctionExecutor(id="test_executor", func=test_func)
    builder.set_start_executor(executor)
    workflow = builder.build()

    discovery = EntityDiscovery()

    # インメモリエンティティを登録する
    entity_info = await discovery.create_entity_info_from_object(workflow, entity_type="workflow", source="in_memory")
    discovery.register_entity(entity_info.id, entity_info, workflow)

    # 即座に利用可能であるべき（レイジーロード不要）
    loaded = discovery.get_entity_object(entity_info.id)
    assert loaded is workflow

    # load_entity() はキャッシュから即座に返すべき
    loaded2 = await discovery.load_entity(entity_info.id)
    assert loaded2 is workflow  # 同じオブジェクト（キャッシュヒット）


if __name__ == "__main__":
    # シンプルなテストランナー
    async def run_tests():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # テストファイルを作成する
            agent_file = temp_path / "test_agent.py"
            agent_file.write_text("""
class WeatherAgent:
    name = "Weather Agent"
    description = "Gets weather information"

    def run_stream(self, input_str):
        return f"Weather in {input_str}"
""")

            workflow_file = temp_path / "test_workflow.py"
            workflow_file.write_text("""
class DataWorkflow:
    name = "Data Processing Workflow"
    description = "Processes data"

    def run(self, data):
        return f"Processed {data}"
""")

            discovery = EntityDiscovery(str(temp_path))
            await discovery.discover_entities()

    asyncio.run(run_tests())
