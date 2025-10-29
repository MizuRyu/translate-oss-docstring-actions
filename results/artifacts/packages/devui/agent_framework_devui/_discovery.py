# Copyright (c) Microsoft. All rights reserved.

"""Agent Frameworkのエンティティ検出実装。"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models._discovery_models import EntityInfo

logger = logging.getLogger(__name__)


class EntityDiscovery:
    """Agent Frameworkエンティティ（エージェントとワークフロー）の検出。"""

    def __init__(self, entities_dir: str | None = None):
        """エンティティ検出を初期化します。

        Args:
            entities_dir: エンティティをスキャンするディレクトリ（オプション）

        """
        self.entities_dir = entities_dir
        self._entities: dict[str, EntityInfo] = {}
        self._loaded_objects: dict[str, Any] = {}

    async def discover_entities(self) -> list[EntityInfo]:
        """Agent Frameworkエンティティをスキャンします。

        Returns:
            検出されたエンティティのリスト

        """
        if not self.entities_dir:
            logger.info("No Agent Framework entities directory configured")
            return []

        entities_dir = Path(self.entities_dir).resolve()  # noqa: ASYNC240
        await self._scan_entities_directory(entities_dir)

        logger.info(f"Discovered {len(self._entities)} Agent Framework entities")
        return self.list_entities()

    def get_entity_info(self, entity_id: str) -> EntityInfo | None:
        """エンティティのメタデータを取得します。

        Args:
            entity_id: エンティティ識別子

        Returns:
            エンティティ情報または見つからない場合はNone

        """
        return self._entities.get(entity_id)

    def get_entity_object(self, entity_id: str) -> Any | None:
        """実際にロードされたエンティティオブジェクトを取得します。

        Args:
            entity_id: エンティティ識別子

        Returns:
            エンティティオブジェクトまたは見つからない場合はNone

        """
        return self._loaded_objects.get(entity_id)

    async def load_entity(self, entity_id: str) -> Any:
        """オンデマンドでエンティティをロードします（遅延ロード）。

        このメソッドは、必要なときにのみエンティティモジュールをインポートすることで遅延ロードを実装します。
        インメモリのエンティティはキャッシュから即座に返されます。

        Args:
            entity_id: エンティティ識別子

        Returns:
            ロードされたエンティティオブジェクト

        Raises:
            ValueError: エンティティが見つからないかロードできない場合

        """
        # すでにロードされているか確認します（インメモリエンティティを含む）。
        if entity_id in self._loaded_objects:
            logger.debug(f"Entity {entity_id} already loaded (cache hit)")
            return self._loaded_objects[entity_id]

        # エンティティのメタデータを取得します。
        entity_info = self._entities.get(entity_id)
        if not entity_info:
            raise ValueError(f"Entity {entity_id} not found in registry")

        # インメモリエンティティはここに到達すべきではありません（事前にロードされています）。
        if entity_info.source == "in_memory":
            raise ValueError(f"In-memory entity {entity_id} missing from loaded objects cache")

        logger.info(f"Lazy loading entity: {entity_id} (source: {entity_info.source})")

        # ソースに基づいてロードします - ディレクトリとインメモリのみサポート。
        if entity_info.source == "directory":
            entity_obj = await self._load_directory_entity(entity_id, entity_info)
        else:
            raise ValueError(
                f"Unsupported entity source: {entity_info.source}. "
                f"Only 'directory' and 'in_memory' sources are supported."
            )

        # 実際のエンティティデータでメタデータを強化します entity_typeが"unknown"の場合は渡さず、推論により実際のタイプを決定させます。
        enriched_info = await self.create_entity_info_from_object(
            entity_obj,
            entity_type=entity_info.type if entity_info.type != "unknown" else None,
            source=entity_info.source,
        )
        # 重要: 元のentity_idを保持します（強化により新しいIDが生成されます）。
        enriched_info.id = entity_id
        # スパースメタデータから元のパスを保持します。
        if "path" in entity_info.metadata:
            enriched_info.metadata["path"] = entity_info.metadata["path"]
        enriched_info.metadata["lazy_loaded"] = True
        self._entities[entity_id] = enriched_info

        # ロードしたオブジェクトをキャッシュします。
        self._loaded_objects[entity_id] = entity_obj
        logger.info(f"Successfully loaded entity: {entity_id} (type: {enriched_info.type})")

        return entity_obj

    async def _load_directory_entity(self, entity_id: str, entity_info: EntityInfo) -> Any:
        """ディレクトリからエンティティをロードします（モジュールをインポート）。

        Args:
            entity_id: エンティティ識別子
            entity_info: エンティティのメタデータ

        Returns:
            ロードされたエンティティオブジェクト

        """
        # メタデータからディレクトリパスを取得します。
        dir_path = Path(entity_info.metadata.get("path", ""))
        if not dir_path.exists():  # noqa: ASYNC240
            raise ValueError(f"Entity directory not found: {dir_path}")

        # .envファイルが存在すればロードします。
        if dir_path.is_dir():  # noqa: ASYNC240
            self._load_env_for_entity(dir_path)
        else:
            self._load_env_for_entity(dir_path.parent)

        # モジュールをインポートします。
        if dir_path.is_dir():  # noqa: ASYNC240
            # ディレクトリベースのエンティティ - 異なるインポートパターンを試みます。
            import_patterns = [
                entity_id,
                f"{entity_id}.agent",
                f"{entity_id}.workflow",
            ]

            for pattern in import_patterns:
                module = self._load_module_from_pattern(pattern)
                if module:
                    # モジュール内でエンティティを見つけます - entity_idを渡して正しいIDで登録します。
                    entity_obj = await self._find_entity_in_module(module, entity_id, str(dir_path))
                    if entity_obj:
                        return entity_obj

            raise ValueError(f"No valid entity found in {dir_path}")
        # ファイルベースのエンティティ。
        module = self._load_module_from_file(dir_path, entity_id)
        if module:
            entity_obj = await self._find_entity_in_module(module, entity_id, str(dir_path))
            if entity_obj:
                return entity_obj

        raise ValueError(f"No valid entity found in {dir_path}")

    def list_entities(self) -> list[EntityInfo]:
        """検出されたすべてのエンティティを一覧表示します。

        Returns:
            すべてのエンティティ情報のリスト

        """
        return list(self._entities.values())

    def invalidate_entity(self, entity_id: str) -> None:
        """エンティティのキャッシュを無効化（クリア）してホットリロードを可能にします。

        これはロード済みオブジェクトのキャッシュからエンティティを削除し、
        Pythonのsys.modulesキャッシュからそのモジュールをクリアします。
        エンティティのメタデータは残るため、次回アクセス時に再インポートされます。

        Args:
            entity_id: 無効化するエンティティ識別子

        """
        # ロード済みオブジェクトのキャッシュから削除します。
        if entity_id in self._loaded_objects:
            del self._loaded_objects[entity_id]
            logger.info(f"Cleared loaded object cache for: {entity_id}")

        # Pythonのモジュールキャッシュ（サブモジュールを含む）からクリアします。
        keys_to_delete = [
            module_name
            for module_name in sys.modules
            if module_name == entity_id or module_name.startswith(f"{entity_id}.")
        ]
        for key in keys_to_delete:
            del sys.modules[key]
            logger.debug(f"Cleared module cache: {key}")

        # メタデータのlazy_loadedフラグをリセットします。
        entity_info = self._entities.get(entity_id)
        if entity_info and "lazy_loaded" in entity_info.metadata:
            entity_info.metadata["lazy_loaded"] = False

        logger.info(f"Entity invalidated: {entity_id} (will reload on next access)")

    def invalidate_all(self) -> None:
        """すべてのキャッシュされたエンティティを無効化します。

        すべてのエンティティの完全なリロードを強制するのに有用です。

        """
        entity_ids = list(self._loaded_objects.keys())
        for entity_id in entity_ids:
            self.invalidate_entity(entity_id)
        logger.info(f"Invalidated {len(entity_ids)} entities")

    def register_entity(self, entity_id: str, entity_info: EntityInfo, entity_object: Any) -> None:
        """メタデータとオブジェクトの両方でエンティティを登録します。

        Args:
            entity_id: ユニークなエンティティ識別子
            entity_info: エンティティのメタデータ
            entity_object: 実行用の実際のエンティティオブジェクト

        """
        self._entities[entity_id] = entity_info
        self._loaded_objects[entity_id] = entity_object
        logger.debug(f"Registered entity: {entity_id} ({entity_info.type})")

    async def create_entity_info_from_object(
        self, entity_object: Any, entity_type: str | None = None, source: str = "in_memory"
    ) -> EntityInfo:
        """Agent FrameworkのエンティティオブジェクトからEntityInfoを作成します。

        Args:
            entity_object: Agent Frameworkのエンティティオブジェクト
            entity_type: オプションのエンティティタイプの上書き
            source: エンティティのソース（directory, in_memory, remote）

        Returns:
            Agent Framework固有のメタデータを持つEntityInfo

        """
        # 提供されていない場合はエンティティタイプを決定します。
        if entity_type is None:
            entity_type = "agent"
            # ワークフローかどうかを確認します。
            if hasattr(entity_object, "get_executors_list") or hasattr(entity_object, "executors"):
                entity_type = "workflow"

        # 改善されたフォールバック命名でメタデータを抽出します。
        name = getattr(entity_object, "name", None)
        if not name:
            # インメモリのエンティティ: UUIDよりもクラス名の方が読みやすいため使用する
            class_name = entity_object.__class__.__name__
            name = f"{entity_type.title()} {class_name}"
        description = getattr(entity_object, "description", "")

        # Agent Framework固有の命名を使用してエンティティIDを生成する
        entity_id = self._generate_entity_id(entity_object, entity_type, source)

        # Agent Framework固有のロジックを使ってツール/エグゼキューターを抽出する
        tools_list = await self._extract_tools_from_object(entity_object, entity_type)

        # エージェント固有のフィールドを抽出する（エージェントのみ）
        instructions = None
        model = None
        chat_client_type = None
        context_providers_list = None
        middleware_list = None

        if entity_type == "agent":
            from ._utils import extract_agent_metadata

            agent_meta = extract_agent_metadata(entity_object)
            instructions = agent_meta["instructions"]
            model = agent_meta["model"]
            chat_client_type = agent_meta["chat_client_type"]
            context_providers_list = agent_meta["context_providers"]
            middleware_list = agent_meta["middleware"]

        # EntityInfoを作成する前にエージェントの機能に関する有用な情報をログに記録する
        if entity_type == "agent":
            has_run_stream = hasattr(entity_object, "run_stream")
            has_run = hasattr(entity_object, "run")

            if not has_run_stream and has_run:
                logger.info(
                    f"Agent '{entity_id}' only has run() (non-streaming). "
                    "DevUI will automatically convert to streaming."
                )
            elif not has_run_stream and not has_run:
                logger.warning(f"Agent '{entity_id}' lacks both run() and run_stream() methods. May not work.")

        # Agent Framework固有の仕様でEntityInfoを作成する
        return EntityInfo(
            id=entity_id,
            name=name,
            description=description,
            type=entity_type,
            framework="agent_framework",
            tools=[str(tool) for tool in (tools_list or [])],
            instructions=instructions,
            model_id=model,
            chat_client_type=chat_client_type,
            context_providers=context_providers_list,
            middleware=middleware_list,
            executors=tools_list if entity_type == "workflow" else [],
            input_schema={"type": "string"},  # Default schema
            start_executor_id=tools_list[0] if tools_list and entity_type == "workflow" else None,
            metadata={
                "source": "agent_framework_object",
                "class_name": entity_object.__class__.__name__
                if hasattr(entity_object, "__class__")
                else str(type(entity_object)),
                "has_run_stream": hasattr(entity_object, "run_stream"),
            },
        )

    async def _scan_entities_directory(self, entities_dir: Path) -> None:
        """Agent Frameworkのエンティティを対象にentitiesディレクトリをスキャンする（遅延読み込み）。

        このメソッドはモジュールをインポートせずにファイルシステムをスキャンし、
        エンティティにアクセスされたときにオンデマンドで補完されるスパースなメタデータを作成します。

        Args:
            entities_dir: エンティティをスキャンするディレクトリ

        """
        if not entities_dir.exists():  # noqa: ASYNC240
            logger.warning(f"Entities directory not found: {entities_dir}")
            return

        logger.info(f"Scanning {entities_dir} for Agent Framework entities (lazy mode)...")

        # まだ存在しない場合はentitiesディレクトリをPythonパスに追加する
        entities_dir_str = str(entities_dir)
        if entities_dir_str not in sys.path:
            sys.path.insert(0, entities_dir_str)

        # インポートせずにディレクトリとPythonファイルをスキャンする
        for item in entities_dir.iterdir():  # noqa: ASYNC240
            if item.name.startswith(".") or item.name == "__pycache__":
                continue

            if item.is_dir() and self._looks_like_entity(item):
                # ディレクトリベースのエンティティ - スパースなメタデータを作成する
                self._register_sparse_entity(item)
            elif item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                # 単一ファイルエンティティ - スパースなメタデータを作成する
                self._register_sparse_file_entity(item)

    def _looks_like_entity(self, dir_path: Path) -> bool:
        """ディレクトリにエンティティが含まれているかどうかをインポートせずにチェックする。

        Args:
            dir_path: チェックするディレクトリ

        Returns:
            ディレクトリがエンティティを含んでいると思われる場合はTrue

        """
        return (
            (dir_path / "agent.py").exists()
            or (dir_path / "workflow.py").exists()
            or (dir_path / "__init__.py").exists()
        )

    def _detect_entity_type(self, dir_path: Path) -> str:
        """ディレクトリ構造からエンティティタイプをインポートせずに検出する。

        ファイル名の規約を使ってエンティティタイプを判別する:
        - workflow.py → "workflow"
        - agent.py → "agent"
        - 両方またはどちらもない → "unknown"

        Args:
            dir_path: 分析するディレクトリ

        Returns:
            エンティティタイプ: "workflow", "agent", または "unknown"

        """
        has_agent = (dir_path / "agent.py").exists()
        has_workflow = (dir_path / "workflow.py").exists()

        if has_agent and has_workflow:
            # 両方のファイルが存在するため曖昧であり、unknownとしてマークする
            return "unknown"
        if has_workflow:
            return "workflow"
        if has_agent:
            return "agent"
        # __init__.pyはあるが特定のファイルはない
        return "unknown"

    def _register_sparse_entity(self, dir_path: Path) -> None:
        """スパースなメタデータでエンティティを登録する（インポートなし）。

        Args:
            dir_path: エンティティのディレクトリ

        """
        entity_id = dir_path.name
        entity_type = self._detect_entity_type(dir_path)

        entity_info = EntityInfo(
            id=entity_id,
            name=entity_id.replace("_", " ").title(),
            type=entity_type,
            framework="agent_framework",
            tools=[],  # Sparse - will be populated on load
            description="",  # Sparse - will be populated on load
            source="directory",
            metadata={
                "path": str(dir_path),
                "discovered": True,
                "lazy_loaded": False,
            },
        )

        self._entities[entity_id] = entity_info
        logger.debug(f"Registered sparse entity: {entity_id} (type: {entity_type})")

    def _register_sparse_file_entity(self, file_path: Path) -> None:
        """スパースなメタデータでファイルベースのエンティティを登録する（インポートなし）。

        Args:
            file_path: エンティティのPythonファイル

        """
        entity_id = file_path.stem

        # ファイルベースのエンティティは通常エージェントだが、インポートしないと確実にはわからない
        entity_info = EntityInfo(
            id=entity_id,
            name=entity_id.replace("_", " ").title(),
            type="unknown",  # Will be determined on load
            framework="agent_framework",
            tools=[],
            description="",
            source="directory",
            metadata={
                "path": str(file_path),
                "discovered": True,
                "lazy_loaded": False,
            },
        )

        self._entities[entity_id] = entity_info
        logger.debug(f"Registered sparse file entity: {entity_id}")

    def _load_env_for_entity(self, entity_path: Path) -> bool:
        """エンティティの.envファイルを読み込む。

        Args:
            entity_path: エンティティディレクトリのパス

        Returns:
            .envが正常に読み込まれた場合はTrue

        """
        # まずエンティティフォルダ内の.envをチェックする
        env_file = entity_path / ".env"
        if self._load_env_file(env_file):
            return True

        # 安全のために1階層上（entitiesディレクトリ）もチェックする
        if self.entities_dir:
            entities_dir = Path(self.entities_dir).resolve()
            entities_env = entities_dir / ".env"
            if self._load_env_file(entities_env):
                return True

        return False

    def _load_env_file(self, env_path: Path) -> bool:
        """.envファイルから環境変数を読み込む。

        Args:
            env_path: .envファイルのパス

        Returns:
            ファイルが正常に読み込まれた場合はTrue

        """
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.debug(f"Loaded .env from {env_path}")
            return True
        return False

    def _load_module_from_pattern(self, pattern: str) -> Any | None:
        """インポートパターンを使ってモジュールを読み込む。

        Args:
            pattern: 試すインポートパターン

        Returns:
            読み込まれたモジュール、失敗した場合はNone

        """
        try:
            # まずモジュールが存在するかをチェックする
            spec = importlib.util.find_spec(pattern)
            if spec is None:
                return None

            module = importlib.import_module(pattern)
            logger.debug(f"Successfully imported {pattern}")
            return module

        except ModuleNotFoundError:
            logger.debug(f"Import pattern {pattern} not found")
            return None
        except Exception as e:
            logger.warning(f"Error importing {pattern}: {e}")
            return None

    def _load_module_from_file(self, file_path: Path, module_name: str) -> Any | None:
        """ファイルパスから直接モジュールを読み込む。

        Args:
            file_path: Pythonファイルのパス
            module_name: モジュールに割り当てる名前

        Returns:
            読み込まれたモジュール、失敗した場合はNone

        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # 適切なインポートのためにsys.modulesに追加する
            spec.loader.exec_module(module)

            logger.debug(f"Successfully loaded module from {file_path}")
            return module

        except Exception as e:
            logger.warning(f"Error loading module from {file_path}: {e}")
            return None

    async def _find_entity_in_module(self, module: Any, entity_id: str, module_path: str) -> Any:
        """読み込んだモジュールからエージェントまたはワークフローエンティティを探す。

        Args:
            module: 読み込んだPythonモジュール
            entity_id: 登録するための期待されるエンティティ識別子
            module_path: メタデータ用のモジュールパス

        Returns:
            読み込まれたエンティティオブジェクト、見つからなければNone

        """
        # まず明示的な変数名を探す
        candidates = [
            ("agent", getattr(module, "agent", None)),
            ("workflow", getattr(module, "workflow", None)),
        ]

        for obj_type, obj in candidates:
            if obj is None:
                continue

            if self._is_valid_entity(obj, obj_type):
                # 正しいentity_id（ディレクトリ名から）で登録する オブジェクトを直接_loaded_objectsに保存して返せるようにする
                self._loaded_objects[entity_id] = obj
                return obj

        return None

    def _is_valid_entity(self, obj: Any, expected_type: str) -> bool:
        """ダックタイピングを使ってオブジェクトが有効なエージェントまたはワークフローかどうかをチェックする。

        Args:
            obj: 検証するオブジェクト
            expected_type: 期待されるタイプ（"agent"または"workflow"）

        Returns:
            オブジェクトが期待されるタイプに有効であればTrue

        """
        if expected_type == "agent":
            return self._is_valid_agent(obj)
        if expected_type == "workflow":
            return self._is_valid_workflow(obj)
        return False

    def _is_valid_agent(self, obj: Any) -> bool:
        """オブジェクトが有効なAgent Frameworkのエージェントかどうかをチェックする。

        Args:
            obj: 検証するオブジェクト

        Returns:
            オブジェクトが有効なエージェントと思われる場合はTrue

        """
        try:
            # 適切な型チェックのためにAgentProtocolのインポートを試みる
            try:
                from agent_framework import AgentProtocol

                if isinstance(obj, AgentProtocol):
                    return True
            except ImportError:
                pass

            # エージェントプロトコルのダックタイピングによるフォールバック
            # Agentはrun_stream()またはrun()メソッドのいずれかとidおよびnameを持つ必要がある
            has_execution_method = hasattr(obj, "run_stream") or hasattr(obj, "run")
            if has_execution_method and hasattr(obj, "id") and hasattr(obj, "name"):
                return True

        except (TypeError, AttributeError):
            pass

        return False

    def _is_valid_workflow(self, obj: Any) -> bool:
        """オブジェクトが有効なAgent Frameworkのワークフローかどうかをチェックする。

        Args:
            obj: 検証するオブジェクト

        Returns:
            オブジェクトが有効なワークフローと思われる場合はTrue

        """
        # ワークフローのチェック - run_streamメソッドとexecutorsを持つ必要がある
        return hasattr(obj, "run_stream") and (hasattr(obj, "executors") or hasattr(obj, "get_executors_list"))

    async def _register_entity_from_object(
        self, obj: Any, obj_type: str, module_path: str, source: str = "directory"
    ) -> None:
        """ライブオブジェクトからエンティティを登録する。

        Args:
            obj: エンティティオブジェクト
            obj_type: エンティティのタイプ（"agent"または"workflow"）
            module_path: メタデータ用のモジュールパス
            source: エンティティのソース（directory, in_memory, remote）

        """
        try:
            # ソース情報を含むエンティティIDを生成する
            entity_id = self._generate_entity_id(obj, obj_type, source)

            # 改善されたフォールバック命名でライブオブジェクトからメタデータを抽出する
            name = getattr(obj, "name", None)
            if not name:
                # UUIDよりもクラス名の方が読みやすいため使用する
                class_name = obj.__class__.__name__
                name = f"{obj_type.title()} {class_name}"
            description = getattr(obj, "description", None)
            tools = await self._extract_tools_from_object(obj, obj_type)

            # EntityInfoを作成する
            tools_union: list[str | dict[str, Any]] | None = None
            if tools:
                tools_union = [tool for tool in tools]

            # エージェント固有のフィールドを抽出する（エージェントのみ）
            instructions = None
            model = None
            chat_client_type = None
            context_providers_list = None
            middleware_list = None

            if obj_type == "agent":
                from ._utils import extract_agent_metadata

                agent_meta = extract_agent_metadata(obj)
                instructions = agent_meta["instructions"]
                model = agent_meta["model"]
                chat_client_type = agent_meta["chat_client_type"]
                context_providers_list = agent_meta["context_providers"]
                middleware_list = agent_meta["middleware"]

            entity_info = EntityInfo(
                id=entity_id,
                type=obj_type,
                name=name,
                framework="agent_framework",
                description=description,
                tools=tools_union,
                instructions=instructions,
                model_id=model,
                chat_client_type=chat_client_type,
                context_providers=context_providers_list,
                middleware=middleware_list,
                metadata={
                    "module_path": module_path,
                    "entity_type": obj_type,
                    "source": source,
                    "has_run_stream": hasattr(obj, "run_stream"),
                    "class_name": obj.__class__.__name__ if hasattr(obj, "__class__") else str(type(obj)),
                },
            )

            # エンティティを登録する
            self.register_entity(entity_id, entity_info, obj)

        except Exception as e:
            logger.error(f"Error registering entity from {source}: {e}")

    async def _extract_tools_from_object(self, obj: Any, obj_type: str) -> list[str]:
        """ライブオブジェクトからツール/エグゼキューター名を抽出する。

        Args:
            obj: エンティティオブジェクト
            obj_type: エンティティのタイプ

        Returns:
            ツール/エグゼキューター名のリスト

        """
        tools = []

        try:
            if obj_type == "agent":
                # エージェントの場合はchat_options.toolsを最初にチェックする
                chat_options = getattr(obj, "chat_options", None)
                if chat_options and hasattr(chat_options, "tools"):
                    for tool in chat_options.tools:
                        if hasattr(tool, "__name__"):
                            tools.append(tool.__name__)
                        elif hasattr(tool, "name"):
                            tools.append(tool.name)
                        else:
                            tools.append(str(tool))
                else:
                    # 直接tools属性へのフォールバック
                    agent_tools = getattr(obj, "tools", None)
                    if agent_tools:
                        for tool in agent_tools:
                            if hasattr(tool, "__name__"):
                                tools.append(tool.__name__)
                            elif hasattr(tool, "name"):
                                tools.append(tool.name)
                            else:
                                tools.append(str(tool))

            elif obj_type == "workflow":
                # ワークフローの場合はエグゼキューター名を抽出する
                if hasattr(obj, "get_executors_list"):
                    executor_objects = obj.get_executors_list()
                    tools = [getattr(ex, "id", str(ex)) for ex in executor_objects]
                elif hasattr(obj, "executors"):
                    executors = obj.executors
                    if isinstance(executors, list):
                        tools = [getattr(ex, "id", str(ex)) for ex in executors]
                    elif isinstance(executors, dict):
                        tools = list(executors.keys())

        except Exception as e:
            logger.debug(f"Error extracting tools from {obj_type} {type(obj)}: {e}")

        return tools

    def _generate_entity_id(self, entity: Any, entity_type: str, source: str = "directory") -> str:
        """衝突回避のためUUIDサフィックス付きのユニークなエンティティIDを生成する。

        Args:
            entity: エンティティオブジェクト
            entity_type: エンティティのタイプ（agent, workflowなど）
            source: エンティティのソース（directory, in_memory, remote）

        Returns:
            フォーマット: {type}_{source}_{name}_{uuid} のユニークなエンティティID

        """
        import re

        # 優先順位付きでベース名を抽出: name -> id -> class_name
        if hasattr(entity, "name") and entity.name:
            base_name = str(entity.name).lower().replace(" ", "-").replace("_", "-")
        elif hasattr(entity, "id") and entity.id:
            base_name = str(entity.id).lower().replace(" ", "-").replace("_", "-")
        elif hasattr(entity, "__class__"):
            class_name = entity.__class__.__name__
            # CamelCaseをkebab-caseに変換する
            base_name = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", class_name).lower()
        else:
            base_name = "entity"

        # 保証された一意性のために完全なUUIDを生成する
        full_uuid = uuid.uuid4().hex

        return f"{entity_type}_{source}_{base_name}_{full_uuid}"
