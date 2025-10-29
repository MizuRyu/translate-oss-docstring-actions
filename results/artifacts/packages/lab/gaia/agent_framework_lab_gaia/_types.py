# Copyright (c) Microsoft. All rights reserved.

"""agent評価のための共通タイプ。"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Evaluation",
    "Evaluator",
    "Prediction",
    "Task",
    "TaskResult",
    "TaskRunner",
]


@dataclass
class Task:
    """評価されるタスクを表します。"""

    task_id: str
    question: str
    answer: str | None = None
    level: int | None = None
    file_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Prediction:
    """タスクに対してagentが行った予測を表します。"""

    prediction: str
    messages: list[Any] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.messages is None:
            self.messages = []


@dataclass
class Evaluation:
    """予測の評価結果を表します。"""

    is_correct: bool
    score: float
    details: dict[str, Any] | None = None


@dataclass
class TaskResult:
    """単一タスク評価の完全な結果。"""

    task_id: str
    task: Task
    prediction: Prediction
    evaluation: Evaluation
    runtime_seconds: float | None = None
    error: str | None = None


@runtime_checkable
class TaskRunner(Protocol):
    """タスクを実行するためのプロトコル。"""

    async def __call__(self, task: Task) -> Prediction:
        """単一タスクを実行し、予測を返します。"""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """予測を評価するためのプロトコル。"""

    async def __call__(self, task: Task, prediction: Prediction) -> Evaluation:
        """指定されたタスクに対する予測を評価します。"""
        ...
