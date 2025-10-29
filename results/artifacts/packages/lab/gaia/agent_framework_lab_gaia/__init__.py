# Copyright (c) Microsoft. All rights reserved.

"""Agent Framework用のGAIAベンチマークモジュール。"""

import importlib.metadata

from ._types import Evaluation, Evaluator, Prediction, Task, TaskResult, TaskRunner
from .gaia import GAIA, GAIATelemetryConfig, gaia_scorer, viewer_main

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # 開発モード用のフォールバック。

__all__ = [
    "GAIA",
    "Evaluation",
    "Evaluator",
    "GAIATelemetryConfig",
    "Prediction",
    "Task",
    "TaskResult",
    "TaskRunner",
    "gaia_scorer",
    "viewer_main",
]
