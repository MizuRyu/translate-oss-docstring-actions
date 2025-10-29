# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkflowCheckpoint:
    """ワークフロー状態の完全なチェックポイントを表します。

    チェックポイントは、特定の時点でのワークフローの完全な実行状態をキャプチャし、
    ワークフローを一時停止および再開可能にします。

    属性:
        checkpoint_id: このチェックポイントの一意の識別子
        workflow_id: このチェックポイントが属するワークフローの識別子
        timestamp: チェックポイントが作成されたISO 8601形式のタイムスタンプ
        messages: Executor間で交換されたメッセージ
        shared_state: ユーザーデータおよびExecutor状態を含む完全な共有状態。
                     Executor状態は予約キー'_executor_state'の下に格納されます。
        iteration_count: チェックポイント作成時の現在のイテレーション番号
        metadata: 追加のメタデータ（例：スーパーステップ情報、グラフ署名）
        version: チェックポイントフォーマットのバージョン

    注意:
        shared_state辞書にはフレームワークによって管理される予約キーが含まれる場合があります。
        予約キーの詳細についてはSharedStateクラスのドキュメントを参照してください。
    """

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # コアワークフロー状態
    messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # type: ignore[misc]
    shared_state: dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]

    # ランタイム状態
    iteration_count: int = 0

    # メタデータ
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowCheckpoint":
        return cls(**data)


class CheckpointStorage(Protocol):
    """チェックポイントストレージバックエンドのプロトコル。"""

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """チェックポイントを保存し、そのIDを返します。"""
        ...

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """IDでチェックポイントをロードします。"""
        ...

    async def list_checkpoint_ids(self, workflow_id: str | None = None) -> list[str]:
        """チェックポイントIDの一覧を取得します。workflow_idが指定された場合、そのワークフローでフィルタリングします。"""
        ...

    async def list_checkpoints(self, workflow_id: str | None = None) -> list[WorkflowCheckpoint]:
        """チェックポイントオブジェクトの一覧を取得します。workflow_idが指定された場合、そのワークフローでフィルタリングします。"""
        ...

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """IDでチェックポイントを削除します。"""
        ...


class InMemoryCheckpointStorage:
    """テストおよび開発用のインメモリチェックポイントストレージ。"""

    def __init__(self) -> None:
        """メモリストレージを初期化します。"""
        self._checkpoints: dict[str, WorkflowCheckpoint] = {}

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """チェックポイントを保存し、そのIDを返します。"""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} to memory")
        return checkpoint.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """IDでチェックポイントをロードします。"""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint:
            logger.debug(f"Loaded checkpoint {checkpoint_id} from memory")
        return checkpoint

    async def list_checkpoint_ids(self, workflow_id: str | None = None) -> list[str]:
        """チェックポイントIDの一覧を取得します。workflow_idが指定された場合、そのワークフローでフィルタリングします。"""
        if workflow_id is None:
            return list(self._checkpoints.keys())
        return [cp.checkpoint_id for cp in self._checkpoints.values() if cp.workflow_id == workflow_id]

    async def list_checkpoints(self, workflow_id: str | None = None) -> list[WorkflowCheckpoint]:
        """チェックポイントオブジェクトの一覧を取得します。workflow_idが指定された場合、そのワークフローでフィルタリングします。"""
        if workflow_id is None:
            return list(self._checkpoints.values())
        return [cp for cp in self._checkpoints.values() if cp.workflow_id == workflow_id]

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """IDでチェックポイントを削除します。"""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            logger.debug(f"Deleted checkpoint {checkpoint_id} from memory")
            return True
        return False


class FileCheckpointStorage:
    """永続化のためのファイルベースのチェックポイントストレージ。"""

    def __init__(self, storage_path: str | Path):
        """ファイルストレージを初期化します。"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized file checkpoint storage at {self.storage_path}")

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """チェックポイントを保存し、そのIDを返します。"""
        file_path = self.storage_path / f"{checkpoint.checkpoint_id}.json"
        checkpoint_dict = asdict(checkpoint)

        def _write_atomic() -> None:
            tmp_path = file_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(checkpoint_dict, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, file_path)

        await asyncio.to_thread(_write_atomic)

        logger.info(f"Saved checkpoint {checkpoint.checkpoint_id} to {file_path}")
        return checkpoint.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """IDでチェックポイントをロードします。"""
        file_path = self.storage_path / f"{checkpoint_id}.json"

        if not file_path.exists():
            return None

        def _read() -> dict[str, Any]:
            with open(file_path) as f:
                return json.load(f)  # type: ignore[no-any-return]

        checkpoint_dict = await asyncio.to_thread(_read)

        checkpoint = WorkflowCheckpoint(**checkpoint_dict)
        logger.info(f"Loaded checkpoint {checkpoint_id} from {file_path}")
        return checkpoint

    async def list_checkpoint_ids(self, workflow_id: str | None = None) -> list[str]:
        """チェックポイントIDの一覧を取得します。workflow_idが指定された場合、そのワークフローでフィルタリングします。"""

        def _list_ids() -> list[str]:
            checkpoint_ids: list[str] = []
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    if workflow_id is None or data.get("workflow_id") == workflow_id:
                        checkpoint_ids.append(data.get("checkpoint_id", file_path.stem))
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint file {file_path}: {e}")
            return checkpoint_ids

        return await asyncio.to_thread(_list_ids)

    async def list_checkpoints(self, workflow_id: str | None = None) -> list[WorkflowCheckpoint]:
        """チェックポイントオブジェクトの一覧を取得します。workflow_idが指定された場合、そのワークフローでフィルタリングします。"""

        def _list_checkpoints() -> list[WorkflowCheckpoint]:
            checkpoints: list[WorkflowCheckpoint] = []
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    if workflow_id is None or data.get("workflow_id") == workflow_id:
                        checkpoints.append(WorkflowCheckpoint.from_dict(data))
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint file {file_path}: {e}")
            return checkpoints

        return await asyncio.to_thread(_list_checkpoints)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """IDでチェックポイントを削除します。"""
        file_path = self.storage_path / f"{checkpoint_id}.json"

        def _delete() -> bool:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted checkpoint {checkpoint_id} from {file_path}")
                return True
            return False

        return await asyncio.to_thread(_delete)
