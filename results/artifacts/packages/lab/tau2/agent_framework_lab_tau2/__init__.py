# Copyright (c) Microsoft. All rights reserved.

"""Agent FrameworkのTau2ベンチマーク。"""

import importlib.metadata

from ._tau2_utils import patch_env_set_state, unpatch_env_set_state
from .runner import ASSISTANT_AGENT_ID, ORCHESTRATOR_ID, USER_SIMULATOR_ID, TaskRunner

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # 開発モードのフォールバック

__all__ = [
    "ASSISTANT_AGENT_ID",
    "ORCHESTRATOR_ID",
    "USER_SIMULATOR_ID",
    "TaskRunner",
    "patch_env_set_state",
    "unpatch_env_set_state",
]
