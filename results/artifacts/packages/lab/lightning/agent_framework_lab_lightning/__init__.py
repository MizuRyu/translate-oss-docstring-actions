# Copyright (c) Microsoft. All rights reserved.

"""Microsoft Agent FrameworkのRLモジュール。"""

import importlib.metadata

from agent_framework.observability import OBSERVABILITY_SETTINGS
from agentlightning import AgentOpsTracer  # type: ignore

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # 開発モード用のフォールバック


class AgentFrameworkTracer(AgentOpsTracer):  # type: ignore
    """Agent-framework用トレーサー。

    Agent-frameworkのOpenTelemetry観測性を有効にするトレーサーで、
    トレースがAgent-lightningに表示されるようにします。

    """

    def init(self) -> None:
        """agent-framework-lab-lightningのトレーニングを初期化する。"""
        OBSERVABILITY_SETTINGS.enable_otel = True
        super().init()

    def teardown(self) -> None:
        """agent-framework-lab-lightningのトレーニングを終了する。"""
        super().teardown()
        OBSERVABILITY_SETTINGS.enable_otel = False


__all__: list[str] = ["AgentFrameworkTracer"]
