# Copyright (c) Microsoft. All rights reserved.

import contextlib
import json
import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Generator, Mapping
from enum import Enum
from functools import wraps
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.semconv_ai import GenAISystem, Meters, SpanAttributes
from pydantic import BaseModel, PrivateAttr

from . import __version__ as version_info
from ._logging import get_logger
from ._pydantic import AFBaseSettings
from .exceptions import AgentInitializationError, ChatClientInitializationError

if TYPE_CHECKING:  # pragma: no cover
    from azure.core.credentials import TokenCredential
    from opentelemetry.sdk._logs._internal.export import LogExporter
    from opentelemetry.sdk.metrics.export import MetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import Tracer
    from opentelemetry.util._decorator import _AgnosticContextManager  # type: ignore[reportPrivateUsage]

    from ._agents import AgentProtocol
    from ._clients import ChatClientProtocol
    from ._threads import AgentThread
    from ._tools import AIFunction
    from ._types import (
        AgentRunResponse,
        AgentRunResponseUpdate,
        ChatMessage,
        ChatResponse,
        ChatResponseUpdate,
        Contents,
        FinishReason,
    )

__all__ = [
    "OBSERVABILITY_SETTINGS",
    "OtelAttr",
    "get_meter",
    "get_tracer",
    "setup_observability",
    "use_agent_observability",
    "use_observability",
]


TAgent = TypeVar("TAgent", bound="AgentProtocol")
TChatClient = TypeVar("TChatClient", bound="ChatClientProtocol")


logger = get_logger()


OTEL_METRICS: Final[str] = "__otel_metrics__"
OPEN_TELEMETRY_CHAT_CLIENT_MARKER: Final[str] = "__open_telemetry_chat_client__"
OPEN_TELEMETRY_AGENT_MARKER: Final[str] = "__open_telemetry_agent__"
TOKEN_USAGE_BUCKET_BOUNDARIES: Final[tuple[float, ...]] = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
OPERATION_DURATION_BUCKET_BOUNDARIES: Final[tuple[float, ...]] = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)


# チャット履歴の複数のイベントを記録していますが、その一部は数百ナノ秒以内に発生しています。デフォルトのタイムスタンプ解像度では各メッセージの一意なタイムスタンプを保証できません。また、Azure
# Monitorは解像度をマイクロ秒に切り捨て、他のバックエンドはミリ秒に切り捨てます。
# しかし、ユーザーにチャットメッセージの順序を復元する方法を提供する必要があるため、各メッセージのタイムスタンプを1マイクロ秒ずつ増加させています。
# これは暫定的な対策であり、より一般的で良い解決策を検討中です。詳細はhttps://github.com/open-telemetry/semantic-conventions/issues/1701を参照してください。
class ChatMessageListTimestampFilter(logging.Filter):
    """INFOログのタイムスタンプを1マイクロ秒増加させるフィルターです。"""

    INDEX_KEY: ClassVar[str] = "chat_message_index"

    def filter(self, record: logging.LogRecord) -> bool:
        """INFOログのタイムスタンプを1マイクロ秒増加させます。"""
        if hasattr(record, self.INDEX_KEY):
            idx = getattr(record, self.INDEX_KEY)
            record.created += idx * 1e-6
        return True


logger.addFilter(ChatMessageListTimestampFilter())


class OtelAttr(str, Enum):
    """Generative AIで使用されるOpenTelemetryの属性をキャプチャするEnumです。

    ベースは https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/ と https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/ です。

    """

    OPERATION = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"
    ERROR_TYPE = "error.type"
    PORT = "server.port"
    ADDRESS = "server.address"
    SPAN_ID = "SpanId"
    TRACE_ID = "TraceId"
    # Request属性
    SEED = "gen_ai.request.seed"
    ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    TOP_K = "gen_ai.request.top_k"
    CHOICE_COUNT = "gen_ai.request.choice.count"
    # Response属性
    FINISH_REASONS = "gen_ai.response.finish_reasons"
    RESPONSE_ID = "gen_ai.response.id"
    # Usage属性
    INPUT_TOKENS = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # Tool属性
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_RESULT = "gen_ai.tool.call.result"
    # Agent属性
    AGENT_ID = "gen_ai.agent.id"
    # Client属性 TOKENはTに置き換えられています。なぜならruffとbanditの両方がTOKENを潜在的なSecretとして警告するためです。
    T_UNIT = "tokens"
    T_TYPE = "gen_ai.token.type"
    T_TYPE_INPUT = "input"
    T_TYPE_OUTPUT = "output"
    DURATION_UNIT = "s"
    # Agent属性
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    CONVERSATION_ID = "gen_ai.conversation.id"
    DATA_SOURCE_ID = "gen_ai.data_source.id"
    OUTPUT_TYPE = "gen_ai.output.type"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

    # Workflow属性
    WORKFLOW_ID = "workflow.id"
    WORKFLOW_NAME = "workflow.name"
    WORKFLOW_DESCRIPTION = "workflow.description"
    WORKFLOW_DEFINITION = "workflow.definition"
    WORKFLOW_BUILD_SPAN = "workflow.build"
    WORKFLOW_RUN_SPAN = "workflow.run"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_ERROR = "workflow.error"
    # Workflow Build属性
    BUILD_STARTED = "build.started"
    BUILD_VALIDATION_COMPLETED = "build.validation_completed"
    BUILD_COMPLETED = "build.completed"
    BUILD_ERROR = "build.error"
    BUILD_ERROR_MESSAGE = "build.error.message"
    BUILD_ERROR_TYPE = "build.error.type"
    # Workflow executor属性
    EXECUTOR_PROCESS_SPAN = "executor.process"
    EXECUTOR_ID = "executor.id"
    EXECUTOR_TYPE = "executor.type"
    # Edge group属性
    EDGE_GROUP_PROCESS_SPAN = "edge_group.process"
    EDGE_GROUP_TYPE = "edge_group.type"
    EDGE_GROUP_ID = "edge_group.id"
    EDGE_GROUP_DELIVERED = "edge_group.delivered"
    EDGE_GROUP_DELIVERY_STATUS = "edge_group.delivery_status"
    # Message属性
    MESSAGE_SEND_SPAN = "message.send"
    MESSAGE_SOURCE_ID = "message.source_id"
    MESSAGE_TARGET_ID = "message.target_id"
    MESSAGE_TYPE = "message.type"
    MESSAGE_DESTINATION_EXECUTOR_ID = "message.destination_executor_id"

    # Activityイベント
    EVENT_NAME = "event.name"
    SYSTEM_MESSAGE = "gen_ai.system.message"
    USER_MESSAGE = "gen_ai.user.message"
    ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    TOOL_MESSAGE = "gen_ai.tool.message"
    CHOICE = "gen_ai.choice"

    # Operation名
    CHAT_COMPLETION_OPERATION = "chat"
    TOOL_EXECUTION_OPERATION = "execute_tool"
    # GenAI Agentの作成を記述し、通常はリモートAgentサービスを扱う際に適用されます。
    AGENT_CREATE_OPERATION = "create_agent"
    AGENT_INVOKE_OPERATION = "invoke_agent"

    # Agent Framework固有の属性
    MEASUREMENT_FUNCTION_TAG_NAME = "agent_framework.function.name"
    MEASUREMENT_FUNCTION_INVOCATION_DURATION = "agent_framework.function.invocation.duration"
    AGENT_FRAMEWORK_GEN_AI_SYSTEM = "microsoft.agent_framework"

    def __repr__(self) -> str:
        """Enumメンバーの文字列表現を返します。"""
        return self.value

    def __str__(self) -> str:
        """Enumメンバーの文字列表現を返します。"""
        return self.value


ROLE_EVENT_MAP = {
    "system": OtelAttr.SYSTEM_MESSAGE,
    "user": OtelAttr.USER_MESSAGE,
    "assistant": OtelAttr.ASSISTANT_MESSAGE,
    "tool": OtelAttr.TOOL_MESSAGE,
}
FINISH_REASON_MAP = {
    "stop": "stop",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
    "length": "length",
}


# region Telemetryユーティリティ


def _get_otlp_exporters(endpoints: list[str]) -> list["LogExporter | SpanExporter | MetricExporter"]:
    """指定されたエンドポイントに対して標準のOTLP Exporterを作成します。"""
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    exporters: list["LogExporter | SpanExporter | MetricExporter"] = []

    for endpoint in endpoints:
        exporters.append(OTLPLogExporter(endpoint=endpoint))
        exporters.append(OTLPSpanExporter(endpoint=endpoint))
        exporters.append(OTLPMetricExporter(endpoint=endpoint))
    return exporters


def _get_azure_monitor_exporters(
    connection_strings: list[str],
    credential: "TokenCredential | None" = None,
) -> list["LogExporter | SpanExporter | MetricExporter"]:
    """接続文字列とオプションでcredentialに基づいてAzure Monitor Exporterを作成します。"""
    from azure.monitor.opentelemetry.exporter import (
        AzureMonitorLogExporter,
        AzureMonitorMetricExporter,
        AzureMonitorTraceExporter,
    )

    exporters: list["LogExporter | SpanExporter | MetricExporter"] = []
    for conn_string in connection_strings:
        exporters.append(AzureMonitorLogExporter(connection_string=conn_string, credential=credential))
        exporters.append(AzureMonitorTraceExporter(connection_string=conn_string, credential=credential))
        exporters.append(AzureMonitorMetricExporter(connection_string=conn_string, credential=credential))
    return exporters


def get_exporters(
    otlp_endpoints: list[str] | None = None,
    connection_strings: list[str] | None = None,
    credential: "TokenCredential | None" = None,
) -> list["LogExporter | SpanExporter | MetricExporter"]:
    """既存の設定に追加のExporterを追加します。

    Exporterを指定した場合、それらは該当するプロバイダーに直接追加されます。
    エンドポイントまたは接続文字列を指定した場合、新しいExporterが作成され追加されます。
    OTLP_endpointsは`OTLPLogExporter`、`OTLPMetricExporter`、`OTLPSpanExporter`の作成に使用されます。
    Connection_stringsはAzureMonitorExporterの作成に使用されます。

    環境変数で既に設定されているエンドポイントまたは接続文字列はスキップされます。
    同じ追加のエンドポイントまたは接続文字列でこのメソッドを2回呼び出すと、2回追加されます。

    Args:
        otlp_endpoints: OpenTelemetry Protocol (OTLP) エンドポイントのリスト。デフォルトはNone。
        connection_strings: Azure Monitor接続文字列のリスト。デフォルトはNone。
        credential: Azure Monitor Entra ID認証に使用するcredential。デフォルトはNone。

    """
    new_exporters: list["LogExporter | SpanExporter | MetricExporter"] = []
    if otlp_endpoints:
        new_exporters.extend(_get_otlp_exporters(endpoints=otlp_endpoints))

    if connection_strings:
        new_exporters.extend(
            _get_azure_monitor_exporters(
                connection_strings=connection_strings,
                credential=credential,
            )
        )
    return new_exporters


def _create_resource() -> "Resource":
    import os

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.attributes import service_attributes

    service_name = os.getenv("OTEL_SERVICE_NAME", "agent_framework")

    return Resource.create({service_attributes.SERVICE_NAME: service_name})


class ObservabilitySettings(AFBaseSettings):
    """Agent Framework Observabilityの設定です。

    環境変数が見つからない場合、設定はutf-8エンコーディングの.envファイルから読み込まれます。
    .envファイルにも設定が見つからない場合、設定は無視されますが、検証は失敗し設定が不足していることを通知します。

    警告:
        センシティブなイベントはテストおよび開発環境でのみ有効にしてください。

    キーワード引数:
        enable_otel: OpenTelemetry診断を有効にします。デフォルトはFalse。
            環境変数ENABLE_OTELで設定可能です。
        enable_sensitive_data: OpenTelemetryのセンシティブイベントを有効にします。デフォルトはFalse。
            環境変数ENABLE_SENSITIVE_DATAで設定可能です。
        applicationinsights_connection_string: Azure Monitor接続文字列。デフォルトはNone。
            環境変数APPLICATIONINSIGHTS_CONNECTION_STRINGで設定可能です。
        otlp_endpoint: OpenTelemetry Protocol (OTLP) エンドポイント。デフォルトはNone。
            環境変数OTLP_ENDPOINTで設定可能です。
        vs_code_extension_port: AI ToolkitまたはAzure AI FoundryのVS Code拡張機能がリッスンしているポート。
            デフォルトはNone。
            環境変数VS_CODE_EXTENSION_PORTで設定可能です。

    Examples:
        .. code-block:: python

            from agent_framework import ObservabilitySettings

            # 環境変数を使用する場合
            # ENABLE_OTEL=trueを設定
            # APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...を設定
            settings = ObservabilitySettings()

            # またはパラメータを直接渡す場合
            settings = ObservabilitySettings(
                enable_otel=True, applicationinsights_connection_string="InstrumentationKey=..."
            )

    """

    env_prefix: ClassVar[str] = ""

    enable_otel: bool = False
    enable_sensitive_data: bool = False
    applicationinsights_connection_string: str | list[str] | None = None
    otlp_endpoint: str | list[str] | None = None
    vs_code_extension_port: int | None = None
    _resource: "Resource" = PrivateAttr(default_factory=_create_resource)
    _executed_setup: bool = PrivateAttr(default=False)

    @property
    def ENABLED(self) -> bool:
        """モデル診断が有効かどうかをチェックします。

        モデル診断は、診断が有効かセンシティブイベントを含む診断が有効な場合に有効となります。

        """
        return self.enable_otel or self.enable_sensitive_data

    @property
    def SENSITIVE_DATA_ENABLED(self) -> bool:
        """センシティブイベントが有効かどうかをチェックします。

        センシティブイベントは、センシティブイベントを含む診断が有効な場合に有効となります。

        """
        return self.enable_sensitive_data

    @property
    def is_setup(self) -> bool:
        """セットアップが実行されたかどうかをチェックします。"""
        return self._executed_setup

    @property
    def resource(self) -> "Resource":
        """リソースを取得します。"""
        return self._resource

    @resource.setter
    def resource(self, value: "Resource") -> None:
        """リソースを設定します。"""
        self._resource = value

    def _configure(
        self,
        credential: "TokenCredential | None" = None,
        additional_exporters: list["LogExporter | SpanExporter | MetricExporter"] | None = None,
    ) -> None:
        """設定に基づいてアプリケーション全体のObservabilityを構成します。

        このメソッドはログ、トレース、メトリックプロバイダーを作成するヘルパーメソッドです。
        アプリケーション起動時に一度だけ呼び出すことを意図しています。複数回呼び出しても効果はありません。

        Args:
            credential: Azure Monitor Entra ID認証に使用するcredential。デフォルトはNone。
            additional_exporters: 設定に追加する追加のExporterのリスト。デフォルトはNone。

        """
        if not self.ENABLED or self._executed_setup:
            return

        exporters: list["LogExporter | SpanExporter | MetricExporter"] = additional_exporters or []
        if self.otlp_endpoint:
            exporters.extend(
                _get_otlp_exporters(
                    self.otlp_endpoint if isinstance(self.otlp_endpoint, list) else [self.otlp_endpoint]
                )
            )
        if self.applicationinsights_connection_string:
            exporters.extend(
                _get_azure_monitor_exporters(
                    connection_strings=(
                        self.applicationinsights_connection_string
                        if isinstance(self.applicationinsights_connection_string, list)
                        else [self.applicationinsights_connection_string]
                    ),
                    credential=credential,
                )
            )
        self._configure_providers(exporters)
        self._executed_setup = True

    def check_endpoint_already_configured(self, otlp_endpoint: str) -> bool:
        """エンドポイントが既に設定されているかどうかをチェックします。

        Returns:
            エンドポイントが既に設定されていればTrue、そうでなければFalse。

        """
        if not self.otlp_endpoint:
            return False
        return otlp_endpoint in (self.otlp_endpoint if isinstance(self.otlp_endpoint, list) else [self.otlp_endpoint])

    def check_connection_string_already_configured(self, connection_string: str) -> bool:
        """接続文字列が既に設定されているかどうかをチェックします。

        Returns:
            接続文字列が既に設定されていればTrue、そうでなければFalse。

        """
        if not self.applicationinsights_connection_string:
            return False
        return connection_string in (
            self.applicationinsights_connection_string
            if isinstance(self.applicationinsights_connection_string, list)
            else [self.applicationinsights_connection_string]
        )

    def _configure_providers(self, exporters: list["LogExporter | MetricExporter | SpanExporter"]) -> None:
        """提供されたExporterを使ってトレース、ログ、イベント、メトリックを構成します。"""
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs._internal.export import LogExporter
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import MetricExporter, PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import DropAggregation, View
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

        # トレース
        tracer_provider = TracerProvider(resource=self.resource)
        trace.set_tracer_provider(tracer_provider)
        should_add_console_exporter = True
        for exporter in exporters:
            if isinstance(exporter, SpanExporter):
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                should_add_console_exporter = False
        if should_add_console_exporter:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        # ログ
        logger_provider = LoggerProvider(resource=self.resource)
        should_add_console_exporter = True
        for exporter in exporters:
            if isinstance(exporter, LogExporter):
                logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
                should_add_console_exporter = False
        if should_add_console_exporter:
            from opentelemetry.sdk._logs._internal.export import ConsoleLogExporter

            logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))

        # プロバイダーのハンドラーをルートロガーにアタッチします。
        logger = logging.getLogger()
        handler = LoggingHandler(logger_provider=logger_provider)
        logger.addHandler(handler)
        set_logger_provider(logger_provider)

        # メトリック
        metric_readers = [
            PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
            for exporter in exporters
            if isinstance(exporter, MetricExporter)
        ]
        if not metric_readers:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

            metric_readers = [PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5000)]
        meter_provider = MeterProvider(
            metric_readers=metric_readers,
            resource=self.resource,
            views=[
                # "agent_framework"で始まるもの以外のすべてのインストルメント名を破棄しています。
                View(instrument_name="*", aggregation=DropAggregation()),
                View(instrument_name="agent_framework*"),
                View(instrument_name="gen_ai*"),
            ],
        )
        metrics.set_meter_provider(meter_provider)


def get_tracer(
    instrumenting_module_name: str = "agent_framework",
    instrumenting_library_version: str = version_info,
    schema_url: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> "trace.Tracer":
    """指定されたinstrumentationライブラリで使用するTracerを返します。

    この関数はtrace.get_tracer()の便利なラッパーで、opentelemetry.trace.TracerProvider.get_tracerの動作を再現します。
    tracer_providerが省略された場合、現在設定されているものが使用されます。

    Args:
        instrumenting_module_name: 計測ライブラリの名前。
            デフォルトは"agent_framework"です。
        instrumenting_library_version: 計測ライブラリのバージョン。
            デフォルトは現在のagent_frameworkのバージョンです。
        schema_url: 発行されるテレメトリのスキーマURL（Optional）。
        attributes: 発行されるテレメトリに関連付けられる属性（Optional）。

    Returns:
        スパン作成に使用するTracerインスタンス。

    Examples:
        .. code-block:: python

            from agent_framework import get_tracer

            # デフォルトのTracerを取得
            tracer = get_tracer()

            # Tracerを使ってスパンを作成
            with tracer.start_as_current_span("my_operation") as span:
                span.set_attribute("custom.attribute", "value")
                # ここに処理を記述
                pass

            # カスタムモジュール名でTracerを取得
            custom_tracer = get_tracer(
                instrumenting_module_name="my_custom_module",
                instrumenting_library_version="1.0.0",
            )

    """
    return trace.get_tracer(
        instrumenting_module_name=instrumenting_module_name,
        instrumenting_library_version=instrumenting_library_version,
        schema_url=schema_url,
        attributes=attributes,
    )


def get_meter(
    name: str = "agent_framework",
    version: str = version_info,
    schema_url: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> "metrics.Meter":
    """Agent Framework用のMeterを返します。

    これはmetrics.get_meter()の便利なラッパーで、opentelemetry.metrics.get_meter()の動作を再現しています。

    Args:
        name: 計測ライブラリの名前。デフォルトは"agent_framework"です。
        version: agent_frameworkのバージョン。デフォルトはパッケージの現在のバージョンです。
        schema_url: 発行されるテレメトリのオプションのスキーマURL。
        attributes: 発行されるテレメトリに関連付けられたオプションの属性。

    Returns:
        メトリクス記録用のMeterインスタンス。

    Examples:
        .. code-block:: python

            from agent_framework import get_meter

            # デフォルトのmeterを取得
            meter = get_meter()

            # カウンタメトリクスを作成
            request_counter = meter.create_counter(
                name="requests",
                description="Number of requests",
                unit="1",
            )
            request_counter.add(1, {"endpoint": "/api/chat"})

            # ヒストグラムメトリクスを作成
            duration_histogram = meter.create_histogram(
                name="request_duration",
                description="Request duration in seconds",
                unit="s",
            )
            duration_histogram.record(0.125, {"status": "success"})

    """
    try:
        return metrics.get_meter(name=name, version=version, schema_url=schema_url, attributes=attributes)
    except TypeError:
        # 古いOpenTelemetryのリリースはattributesパラメータをサポートしていません。
        return metrics.get_meter(name=name, version=version, schema_url=schema_url)


global OBSERVABILITY_SETTINGS
OBSERVABILITY_SETTINGS: ObservabilitySettings = ObservabilitySettings()


def setup_observability(
    enable_sensitive_data: bool | None = None,
    otlp_endpoint: str | list[str] | None = None,
    applicationinsights_connection_string: str | list[str] | None = None,
    credential: "TokenCredential | None" = None,
    exporters: list["LogExporter | SpanExporter | MetricExporter"] | None = None,
    vs_code_extension_port: int | None = None,
) -> None:
    """OpenTelemetryを用いてアプリケーションの可観測性をセットアップします。

    このメソッドは、提供された値と環境変数に基づいてアプリケーション用のエクスポーターとプロバイダーを作成します。

    アプリケーション起動時に一度だけ呼び出してください。複数回呼び出すと予期しない動作を引き起こす可能性があります。

    注意:
        プロバイダーを手動で設定している場合、このメソッドを呼び出しても効果はありません。逆も同様で、このメソッドを先に呼び出すと、その後のプロバイダー設定は反映されません。

    Args:
        enable_sensitive_data: OpenTelemetryのセンシティブイベントを有効にします。設定されている場合は環境変数を上書きします。デフォルトはNoneです。
        otlp_endpoint: OpenTelemetry Protocol (OTLP)のエンドポイント。OTLPLogExporter、OTLPMetricExporter、OTLPSpanExporterの作成に使用されます。デフォルトはNoneです。
        applicationinsights_connection_string: Azure Monitorの接続文字列。AzureMonitorExportersの作成に使用されます。デフォルトはNoneです。
        credential: Azure Monitor Entra ID認証に使用する資格情報。デフォルトはNoneです。
        exporters: ログ、メトリクス、スパンのいずれかまたは組み合わせのエクスポーターのリスト。完全なカスタマイズが可能です。デフォルトはNoneです。
        vs_code_extension_port: AI ToolkitまたはAzureAI FoundryのVS Code拡張機能がリッスンしているポート。設定されている場合、追加のOTELエクスポーターが`http://localhost:{vs_code_extension_port}`のエンドポイントで作成されます（既に設定されていない場合）。設定されている場合は環境変数を上書きします。デフォルトはNoneです。

    Examples:
        .. code-block:: python

            from agent_framework import setup_observability

            # 環境変数を使う場合
            # ENABLE_OTEL=true, OTLP_ENDPOINT=http://localhost:4317を設定
            setup_observability()

            # パラメータ指定の場合（環境変数なし）
            setup_observability(
                enable_sensitive_data=True,
                otlp_endpoint="http://localhost:4317",
            )

            # Azure Monitorを使う場合
            setup_observability(
                applicationinsights_connection_string="InstrumentationKey=...",
            )

            # カスタムエクスポーターを使う場合
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            setup_observability(
                exporters=[ConsoleSpanExporter()],
            )

            # 混合: 環境変数とパラメータの組み合わせ
            # 環境変数: OTLP_ENDPOINT=http://localhost:7431
            # コードで追加のエンドポイントを指定
            setup_observability(
                enable_sensitive_data=True,
                otlp_endpoint="http://localhost:4317",  # 両方のエンドポイントが使用されます
            )

            # VS Code拡張機能との連携
            setup_observability(
                vs_code_extension_port=4317,  # AI Toolkitに接続
            )

    """
    global OBSERVABILITY_SETTINGS
    # 提供された値で可観測性設定を更新します。
    OBSERVABILITY_SETTINGS.enable_otel = True
    if enable_sensitive_data is not None:
        OBSERVABILITY_SETTINGS.enable_sensitive_data = enable_sensitive_data
    if vs_code_extension_port is not None:
        OBSERVABILITY_SETTINGS.vs_code_extension_port = vs_code_extension_port

    # 環境変数で既に設定されているか確認した後にエクスポーターを作成します。
    new_exporters: list["LogExporter | SpanExporter | MetricExporter"] = exporters or []
    if otlp_endpoint:
        if isinstance(otlp_endpoint, str):
            otlp_endpoint = [otlp_endpoint]
        new_exporters.extend(
            _get_otlp_exporters(
                endpoints=[
                    endpoint
                    for endpoint in otlp_endpoint
                    if not OBSERVABILITY_SETTINGS.check_endpoint_already_configured(endpoint)
                ]
            )
        )
    if applicationinsights_connection_string:
        if isinstance(applicationinsights_connection_string, str):
            applicationinsights_connection_string = [applicationinsights_connection_string]
        new_exporters.extend(
            _get_azure_monitor_exporters(
                connection_strings=[
                    conn_str
                    for conn_str in applicationinsights_connection_string
                    if not OBSERVABILITY_SETTINGS.check_connection_string_already_configured(conn_str)
                ],
                credential=credential,
            )
        )
    if OBSERVABILITY_SETTINGS.vs_code_extension_port:
        endpoint = f"http://localhost:{OBSERVABILITY_SETTINGS.vs_code_extension_port}"
        if not OBSERVABILITY_SETTINGS.check_endpoint_already_configured(endpoint):
            new_exporters.extend(_get_otlp_exporters(endpoints=[endpoint]))

    OBSERVABILITY_SETTINGS._configure(credential=credential, additional_exporters=new_exporters)  # pyright: ignore[reportPrivateUsage]


# region Chat Client Telemetry


def _get_duration_histogram() -> "metrics.Histogram":
    return get_meter().create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit=OtelAttr.DURATION_UNIT,
        description="Captures the duration of operations of function-invoking chat clients",
        explicit_bucket_boundaries_advisory=OPERATION_DURATION_BUCKET_BOUNDARIES,
    )


def _get_token_usage_histogram() -> "metrics.Histogram":
    return get_meter().create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit=OtelAttr.T_UNIT,
        description="Captures the token usage of chat clients",
        explicit_bucket_boundaries_advisory=TOKEN_USAGE_BUCKET_BOUNDARIES,
    )


# region ChatClientProtocol


def _trace_get_response(
    func: Callable[..., Awaitable["ChatResponse"]],
    *,
    provider_name: str = "unknown",
) -> Callable[..., Awaitable["ChatResponse"]]:
    """チャット完了アクティビティをトレースするデコレーター。

    Args:
        func: トレース対象の関数。

    Keyword Args:
        provider_name: モデルプロバイダー名。

    """

    def decorator(func: Callable[..., Awaitable["ChatResponse"]]) -> Callable[..., Awaitable["ChatResponse"]]:
        """内部デコレーター。"""

        @wraps(func)
        async def trace_get_response(
            self: "ChatClientProtocol",
            messages: "str | ChatMessage | list[str] | list[ChatMessage]",
            **kwargs: Any,
        ) -> "ChatResponse":
            global OBSERVABILITY_SETTINGS
            if not OBSERVABILITY_SETTINGS.ENABLED:
                # model_idの診断が有効でない場合は、単に完了を返します。
                return await func(
                    self,
                    messages=messages,
                    **kwargs,
                )
            if "token_usage_histogram" not in self.additional_properties:
                self.additional_properties["token_usage_histogram"] = _get_token_usage_histogram()
            if "operation_duration_histogram" not in self.additional_properties:
                self.additional_properties["operation_duration_histogram"] = _get_duration_histogram()
            model_id = (
                kwargs.get("model_id")
                or (chat_options.model_id if (chat_options := kwargs.get("chat_options")) else None)
                or getattr(self, "model_id", None)
            )
            service_url = str(
                service_url_func()
                if (service_url_func := getattr(self, "service_url", None)) and callable(service_url_func)
                else "unknown"
            )
            attributes = _get_span_attributes(
                operation_name=OtelAttr.CHAT_COMPLETION_OPERATION,
                provider_name=provider_name,
                model=model_id,
                service_url=service_url,
                **kwargs,
            )
            with _get_span(attributes=attributes, span_name_attribute=SpanAttributes.LLM_REQUEST_MODEL) as span:
                if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and messages:
                    _capture_messages(span=span, provider_name=provider_name, messages=messages)
                start_time_stamp = perf_counter()
                end_time_stamp: float | None = None
                try:
                    response = await func(self, messages=messages, **kwargs)
                    end_time_stamp = perf_counter()
                except Exception as exception:
                    end_time_stamp = perf_counter()
                    capture_exception(span=span, exception=exception, timestamp=time_ns())
                    raise
                else:
                    duration = (end_time_stamp or perf_counter()) - start_time_stamp
                    attributes = _get_response_attributes(attributes, response, duration=duration)
                    _capture_response(
                        span=span,
                        attributes=attributes,
                        token_usage_histogram=self.additional_properties["token_usage_histogram"],
                        operation_duration_histogram=self.additional_properties["operation_duration_histogram"],
                    )
                    if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and response.messages:
                        _capture_messages(
                            span=span,
                            provider_name=provider_name,
                            messages=response.messages,
                            finish_reason=response.finish_reason,
                            output=True,
                        )
                    return response

        return trace_get_response

    return decorator(func)


def _trace_get_streaming_response(
    func: Callable[..., AsyncIterable["ChatResponseUpdate"]],
    *,
    provider_name: str = "unknown",
) -> Callable[..., AsyncIterable["ChatResponseUpdate"]]:
    """ストリーミングチャット完了アクティビティをトレースするデコレーター。

    Args:
        func: トレース対象の関数。

    Keyword Args:
        provider_name: モデルプロバイダー名。

    """

    def decorator(
        func: Callable[..., AsyncIterable["ChatResponseUpdate"]],
    ) -> Callable[..., AsyncIterable["ChatResponseUpdate"]]:
        """内部デコレーター。"""

        @wraps(func)
        async def trace_get_streaming_response(
            self: "ChatClientProtocol", messages: "str | ChatMessage | list[str] | list[ChatMessage]", **kwargs: Any
        ) -> AsyncIterable["ChatResponseUpdate"]:
            global OBSERVABILITY_SETTINGS
            if not OBSERVABILITY_SETTINGS.ENABLED:
                # モデル診断が有効でない場合は、単に完了を返します。
                async for update in func(self, messages=messages, **kwargs):
                    yield update
                return
            if "token_usage_histogram" not in self.additional_properties:
                self.additional_properties["token_usage_histogram"] = _get_token_usage_histogram()
            if "operation_duration_histogram" not in self.additional_properties:
                self.additional_properties["operation_duration_histogram"] = _get_duration_histogram()

            model_id = (
                kwargs.get("model_id")
                or (chat_options.model_id if (chat_options := kwargs.get("chat_options")) else None)
                or getattr(self, "model_id", None)
            )
            service_url = str(
                service_url_func()
                if (service_url_func := getattr(self, "service_url", None)) and callable(service_url_func)
                else "unknown"
            )
            attributes = _get_span_attributes(
                operation_name=OtelAttr.CHAT_COMPLETION_OPERATION,
                provider_name=provider_name,
                model=model_id,
                service_url=service_url,
                **kwargs,
            )
            all_updates: list["ChatResponseUpdate"] = []
            with _get_span(attributes=attributes, span_name_attribute=SpanAttributes.LLM_REQUEST_MODEL) as span:
                if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and messages:
                    _capture_messages(
                        span=span,
                        provider_name=provider_name,
                        messages=messages,
                    )
                start_time_stamp = perf_counter()
                end_time_stamp: float | None = None
                try:
                    async for update in func(self, messages=messages, **kwargs):
                        all_updates.append(update)
                        yield update
                    end_time_stamp = perf_counter()
                except Exception as exception:
                    end_time_stamp = perf_counter()
                    capture_exception(span=span, exception=exception, timestamp=time_ns())
                    raise
                else:
                    duration = (end_time_stamp or perf_counter()) - start_time_stamp
                    from ._types import ChatResponse

                    response = ChatResponse.from_chat_response_updates(all_updates)
                    attributes = _get_response_attributes(attributes, response, duration=duration)
                    _capture_response(
                        span=span,
                        attributes=attributes,
                        token_usage_histogram=self.additional_properties["token_usage_histogram"],
                        operation_duration_histogram=self.additional_properties["operation_duration_histogram"],
                    )

                    if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and response.messages:
                        _capture_messages(
                            span=span,
                            provider_name=provider_name,
                            messages=response.messages,
                            finish_reason=response.finish_reason,
                            output=True,
                        )

        return trace_get_streaming_response

    return decorator(func)


def use_observability(
    chat_client: type[TChatClient],
) -> type[TChatClient]:
    """チャットクライアントのOpenTelemetry可観測性を有効にするクラスデコレーター。

    このデコレーターは、チャット完了リクエストを自動的にトレースし、メトリクスをキャプチャし、イベントをログに記録します。

    注意:
        このデコレーターはインスタンスではなくクラス自体に適用する必要があります。
        チャットクライアントクラスは、テレメトリ用の適切なプロバイダー名を設定するためにクラス変数OTEL_PROVIDER_NAMEを持つ必要があります。

    Args:
        chat_client: 可観測性を有効にするチャットクライアントクラス。

    Returns:
        可観測性が有効になったデコレーション済みチャットクライアントクラス。

    Raises:
        ChatClientInitializationError: チャットクライアントに必要なメソッド（get_response, get_streaming_response）がない場合。

    Examples:
        .. code-block:: python

            from agent_framework import use_observability, setup_observability
            from agent_framework._clients import ChatClientProtocol


            # カスタムチャットクライアントクラスをデコレート
            @use_observability
            class MyCustomChatClient:
                OTEL_PROVIDER_NAME = "my_provider"

                async def get_response(self, messages, **kwargs):
                    # 実装
                    pass

                async def get_streaming_response(self, messages, **kwargs):
                    # 実装
                    pass


            # 可観測性をセットアップ
            setup_observability(otlp_endpoint="http://localhost:4317")

            # これで全ての呼び出しがトレースされます
            client = MyCustomChatClient()
            response = await client.get_response("Hello")

    """
    if getattr(chat_client, OPEN_TELEMETRY_CHAT_CLIENT_MARKER, False):
        # すでにデコレート済みです。
        return chat_client

    provider_name = str(getattr(chat_client, "OTEL_PROVIDER_NAME", "unknown"))

    if provider_name not in GenAISystem.__members__:
        # そのリストは完全ではないため、単にログを記録するだけで影響はありません。
        logger.debug(
            f"The provider name '{provider_name}' is not recognized. "
            f"Consider using one of the following: {', '.join(GenAISystem.__members__.keys())}"
        )
    try:
        chat_client.get_response = _trace_get_response(chat_client.get_response, provider_name=provider_name)  # type: ignore
    except AttributeError as exc:
        raise ChatClientInitializationError(
            f"The chat client {chat_client.__name__} does not have a get_response method.", exc
        ) from exc
    try:
        chat_client.get_streaming_response = _trace_get_streaming_response(  # type: ignore
            chat_client.get_streaming_response, provider_name=provider_name
        )
    except AttributeError as exc:
        raise ChatClientInitializationError(
            f"The chat client {chat_client.__name__} does not have a get_streaming_response method.", exc
        ) from exc

    setattr(chat_client, OPEN_TELEMETRY_CHAT_CLIENT_MARKER, True)

    return chat_client


# region Agent


def _trace_agent_run(
    run_func: Callable[..., Awaitable["AgentRunResponse"]],
    provider_name: str,
) -> Callable[..., Awaitable["AgentRunResponse"]]:
    """チャット完了アクティビティをトレースするデコレーター。

    Args:
        run_func: トレース対象の関数。
        provider_name: Open Telemetryで使用されるシステム名。

    """

    @wraps(run_func)
    async def trace_run(
        self: "AgentProtocol",
        messages: "str | ChatMessage | list[str] | list[ChatMessage] | None" = None,
        *,
        thread: "AgentThread | None" = None,
        **kwargs: Any,
    ) -> "AgentRunResponse":
        global OBSERVABILITY_SETTINGS

        if not OBSERVABILITY_SETTINGS.ENABLED:
            # モデル診断が有効でない場合は、単に完了を返します。
            return await run_func(self, messages=messages, thread=thread, **kwargs)
        attributes = _get_span_attributes(
            operation_name=OtelAttr.AGENT_INVOKE_OPERATION,
            provider_name=provider_name,
            agent_id=self.id,
            agent_name=self.display_name,
            agent_description=self.description,
            thread_id=thread.service_thread_id if thread else None,
            chat_options=getattr(self, "chat_options", None),
            **kwargs,
        )
        with _get_span(attributes=attributes, span_name_attribute=OtelAttr.AGENT_NAME) as span:
            if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and messages:
                _capture_messages(
                    span=span,
                    provider_name=provider_name,
                    messages=messages,
                    system_instructions=getattr(self, "instructions", None),
                )
            try:
                response = await run_func(self, messages=messages, thread=thread, **kwargs)
            except Exception as exception:
                capture_exception(span=span, exception=exception, timestamp=time_ns())
                raise
            else:
                attributes = _get_response_attributes(attributes, response)
                _capture_response(span=span, attributes=attributes)
                if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and response.messages:
                    _capture_messages(
                        span=span,
                        provider_name=provider_name,
                        messages=response.messages,
                        output=True,
                    )
                return response

    return trace_run


def _trace_agent_run_stream(
    run_streaming_func: Callable[..., AsyncIterable["AgentRunResponseUpdate"]],
    provider_name: str,
) -> Callable[..., AsyncIterable["AgentRunResponseUpdate"]]:
    """ストリーミングエージェント実行アクティビティをトレースするデコレーター。

    Args:
        run_streaming_func: トレース対象の関数。
        provider_name: Open Telemetryで使用されるシステム名。

    """

    @wraps(run_streaming_func)
    async def trace_run_streaming(
        self: "AgentProtocol",
        messages: "str | ChatMessage | list[str] | list[ChatMessage] | None" = None,
        *,
        thread: "AgentThread | None" = None,
        **kwargs: Any,
    ) -> AsyncIterable["AgentRunResponseUpdate"]:
        global OBSERVABILITY_SETTINGS

        if not OBSERVABILITY_SETTINGS.ENABLED:
            # モデル診断が有効でない場合は、単に完了を返します。
            async for streaming_agent_response in run_streaming_func(self, messages=messages, thread=thread, **kwargs):
                yield streaming_agent_response
            return

        from ._types import AgentRunResponse

        all_updates: list["AgentRunResponseUpdate"] = []

        attributes = _get_span_attributes(
            operation_name=OtelAttr.AGENT_INVOKE_OPERATION,
            provider_name=provider_name,
            agent_id=self.id,
            agent_name=self.display_name,
            agent_description=self.description,
            thread_id=thread.service_thread_id if thread else None,
            chat_options=getattr(self, "chat_options", None),
            **kwargs,
        )
        with _get_span(attributes=attributes, span_name_attribute=OtelAttr.AGENT_NAME) as span:
            if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and messages:
                _capture_messages(
                    span=span,
                    provider_name=provider_name,
                    messages=messages,
                    system_instructions=getattr(self, "instructions", None),
                )
            try:
                async for update in run_streaming_func(self, messages=messages, thread=thread, **kwargs):
                    all_updates.append(update)
                    yield update
            except Exception as exception:
                capture_exception(span=span, exception=exception, timestamp=time_ns())
                raise
            else:
                response = AgentRunResponse.from_agent_run_response_updates(all_updates)
                attributes = _get_response_attributes(attributes, response)
                _capture_response(span=span, attributes=attributes)
                if OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED and response.messages:
                    _capture_messages(
                        span=span,
                        provider_name=provider_name,
                        messages=response.messages,
                        output=True,
                    )

    return trace_run_streaming


def use_agent_observability(
    agent: type[TAgent],
) -> type[TAgent]:
    """エージェントのOpenTelemetry可観測性を有効にするクラスデコレーター。

    このデコレーターは、エージェントの実行リクエストを自動的にトレースし、イベントをキャプチャし、インタラクションをログに記録します。

    注意:
        このデコレーターはインスタンスではなくエージェントクラス自体に適用する必要があります。
        エージェントクラスは、テレメトリ用の適切なシステム名を設定するためにクラス変数AGENT_SYSTEM_NAMEを持つ必要があります。

    Args:
        agent: 可観測性を有効にするエージェントクラス。

    Returns:
        可観測性が有効になったデコレーション済みエージェントクラス。

    Raises:
        AgentInitializationError: エージェントに必要なメソッド（run, run_stream）がない場合。

    Examples:
        .. code-block:: python

            from agent_framework import use_agent_observability, setup_observability
            from agent_framework._agents import AgentProtocol


            # カスタムエージェントクラスをデコレート
            @use_agent_observability
            class MyCustomAgent:
                AGENT_SYSTEM_NAME = "my_agent_system"

                async def run(self, messages=None, *, thread=None, **kwargs):
                    # 実装
                    pass

                async def run_stream(self, messages=None, *, thread=None, **kwargs):
                    # 実装
                    pass


            # 可観測性をセットアップ
            setup_observability(otlp_endpoint="http://localhost:4317")

            # これで全てのエージェント実行がトレースされます
            agent = MyCustomAgent()
            response = await agent.run("Perform a task")

    """
    provider_name = str(getattr(agent, "AGENT_SYSTEM_NAME", "Unknown"))
    try:
        agent.run = _trace_agent_run(agent.run, provider_name)  # type: ignore
    except AttributeError as exc:
        raise AgentInitializationError(f"The agent {agent.__name__} does not have a run method.", exc) from exc
    try:
        agent.run_stream = _trace_agent_run_stream(agent.run_stream, provider_name)  # type: ignore
    except AttributeError as exc:
        raise AgentInitializationError(f"The agent {agent.__name__} does not have a run_stream method.", exc) from exc
    setattr(agent, OPEN_TELEMETRY_AGENT_MARKER, True)
    return agent


# region Otel Helpers


def get_function_span_attributes(function: "AIFunction[Any, Any]", tool_call_id: str | None = None) -> dict[str, str]:
    """指定された関数のspan属性を取得します。

    Args:
        function: span属性を取得する対象の関数。
        tool_call_id: 要求されたtool_callのID。

    Returns:
        dict[str, str]: span属性。

    """
    attributes: dict[str, str] = {
        OtelAttr.OPERATION: OtelAttr.TOOL_EXECUTION_OPERATION,
        OtelAttr.TOOL_NAME: function.name,
        OtelAttr.TOOL_CALL_ID: tool_call_id or "unknown",
        OtelAttr.TOOL_TYPE: "function",
    }
    if function.description:
        attributes[OtelAttr.TOOL_DESCRIPTION] = function.description
    return attributes


def get_function_span(
    attributes: dict[str, str],
) -> "_AgnosticContextManager[trace.Span]":
    """指定された関数のspanを開始します。

    Args:
        attributes: span属性。

    Returns:
        trace.trace.Span: コンテキストマネージャとして開始されたspan。

    """
    return get_tracer().start_as_current_span(
        name=f"{attributes[OtelAttr.OPERATION]} {attributes[OtelAttr.TOOL_NAME]}",
        attributes=attributes,
        set_status_on_exception=False,
        end_on_exit=True,
        record_exception=False,
    )


@contextlib.contextmanager
def _get_span(
    attributes: dict[str, Any],
    span_name_attribute: str,
) -> Generator["trace.Span", Any, Any]:
    """agent実行のspanを開始します。"""
    span = get_tracer().start_span(f"{attributes[OtelAttr.OPERATION]} {attributes[span_name_attribute]}")
    span.set_attributes(attributes)
    with trace.use_span(
        span=span,
        end_on_exit=True,
        record_exception=False,
        set_status_on_exception=False,
    ) as current_span:
        yield current_span


def _get_span_attributes(**kwargs: Any) -> dict[str, Any]:
    """kwargs辞書からspan属性を取得します。"""
    from ._tools import _tools_to_dict
    from ._types import ChatOptions

    attributes: dict[str, Any] = {}
    chat_options: ChatOptions | None = kwargs.get("chat_options")
    if chat_options is None:
        chat_options = ChatOptions()
    if operation_name := kwargs.get("operation_name"):
        attributes[OtelAttr.OPERATION] = operation_name
    if choice_count := kwargs.get("choice_count", 1):
        attributes[OtelAttr.CHOICE_COUNT] = choice_count
    if system_name := kwargs.get("system_name"):
        attributes[SpanAttributes.LLM_SYSTEM] = system_name
    if provider_name := kwargs.get("provider_name"):
        attributes[OtelAttr.PROVIDER_NAME] = provider_name
    attributes[SpanAttributes.LLM_REQUEST_MODEL] = kwargs.get("model", "unknown")
    if service_url := kwargs.get("service_url"):
        attributes[OtelAttr.ADDRESS] = service_url
    if conversation_id := kwargs.get("conversation_id", chat_options.conversation_id):
        attributes[OtelAttr.CONVERSATION_ID] = conversation_id
    if seed := kwargs.get("seed", chat_options.seed):
        attributes[OtelAttr.SEED] = seed
    if frequency_penalty := kwargs.get("frequency_penalty", chat_options.frequency_penalty):
        attributes[OtelAttr.FREQUENCY_PENALTY] = frequency_penalty
    if max_tokens := kwargs.get("max_tokens", chat_options.max_tokens):
        attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = max_tokens
    if stop := kwargs.get("stop", chat_options.stop):
        attributes[OtelAttr.STOP_SEQUENCES] = stop
    if temperature := kwargs.get("temperature", chat_options.temperature):
        attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] = temperature
    if top_p := kwargs.get("top_p", chat_options.top_p):
        attributes[SpanAttributes.LLM_REQUEST_TOP_P] = top_p
    if presence_penalty := kwargs.get("presence_penalty", chat_options.presence_penalty):
        attributes[OtelAttr.PRESENCE_PENALTY] = presence_penalty
    if top_k := kwargs.get("top_k"):
        attributes[OtelAttr.TOP_K] = top_k
    if encoding_formats := kwargs.get("encoding_formats"):
        attributes[OtelAttr.ENCODING_FORMATS] = json.dumps(
            encoding_formats if isinstance(encoding_formats, list) else [encoding_formats]
        )
    if tools := kwargs.get("tools", chat_options.tools):
        tools_as_json_list = _tools_to_dict(tools)
        if tools_as_json_list:
            attributes[OtelAttr.TOOL_DEFINITIONS] = json.dumps(tools_as_json_list)
    if error := kwargs.get("error"):
        attributes[OtelAttr.ERROR_TYPE] = type(error).__name__
    # agent属性
    if agent_id := kwargs.get("agent_id"):
        attributes[OtelAttr.AGENT_ID] = agent_id
    if agent_name := kwargs.get("agent_name"):
        attributes[OtelAttr.AGENT_NAME] = agent_name
    if agent_description := kwargs.get("agent_description"):
        attributes[OtelAttr.AGENT_DESCRIPTION] = agent_description
    if thread_id := kwargs.get("thread_id"):
        # threadが設定されている場合は上書きします。
        attributes[OtelAttr.CONVERSATION_ID] = thread_id
    return attributes


def capture_exception(span: trace.Span, exception: Exception, timestamp: int | None = None) -> None:
    """spanにエラーを設定します。"""
    span.set_attribute(OtelAttr.ERROR_TYPE, type(exception).__name__)
    span.record_exception(exception=exception, timestamp=timestamp)
    span.set_status(status=trace.StatusCode.ERROR, description=repr(exception))


def _capture_messages(
    span: trace.Span,
    provider_name: str,
    messages: "str | ChatMessage | list[str] | list[ChatMessage]",
    system_instructions: str | list[str] | None = None,
    output: bool = False,
    finish_reason: "FinishReason | None" = None,
) -> None:
    """追加情報付きでメッセージをログに記録します。"""
    from ._clients import prepare_messages

    prepped = prepare_messages(messages)
    otel_messages: list[dict[str, Any]] = []
    for index, message in enumerate(prepped):
        otel_messages.append(_to_otel_message(message))
        try:
            message_data = message.to_dict(exclude_none=True)
        except Exception:
            message_data = {"role": message.role.value, "contents": message.contents}
        logger.info(
            message_data,
            extra={
                OtelAttr.EVENT_NAME: OtelAttr.CHOICE if output else ROLE_EVENT_MAP.get(message.role.value),
                OtelAttr.PROVIDER_NAME: provider_name,
                ChatMessageListTimestampFilter.INDEX_KEY: index,
            },
        )
    if finish_reason:
        otel_messages[-1]["finish_reason"] = FINISH_REASON_MAP[finish_reason.value]
    span.set_attribute(OtelAttr.OUTPUT_MESSAGES if output else OtelAttr.INPUT_MESSAGES, json.dumps(otel_messages))
    if system_instructions:
        if not isinstance(system_instructions, list):
            system_instructions = [system_instructions]
        otel_sys_instructions = [{"type": "text", "content": instruction} for instruction in system_instructions]
        span.set_attribute(OtelAttr.SYSTEM_INSTRUCTIONS, json.dumps(otel_sys_instructions))


def _to_otel_message(message: "ChatMessage") -> dict[str, Any]:
    """メッセージのotel表現を作成します。"""
    return {"role": message.role.value, "parts": [_to_otel_part(content) for content in message.contents]}


def _to_otel_part(content: "Contents") -> dict[str, Any] | None:
    """Contentのotel表現を作成します。"""
    match content.type:
        case "text":
            return {"type": "text", "content": content.text}
        case "function_call":
            return {"type": "tool_call", "id": content.call_id, "name": content.name, "arguments": content.arguments}
        case "function_result":
            response: Any | None = None
            if content.result:
                if isinstance(content.result, list):
                    res: list[Any] = []
                    for item in content.result:  # type: ignore
                        from ._types import BaseContent

                        if isinstance(item, BaseContent):
                            res.append(_to_otel_part(item))  # type: ignore
                        elif isinstance(item, BaseModel):
                            res.append(item.model_dump(exclude_none=True))
                        else:
                            res.append(json.dumps(item))
                    response = json.dumps(res)
                else:
                    response = json.dumps(content.result)
            return {"type": "tool_call_response", "id": content.call_id, "response": response}
        case _:
            # otel出力メッセージのjson仕様におけるGenericPart。 必須のtypeと任意の他のフィールドのみを含みます。
            return content.to_dict(exclude_none=True)
    return None


def _get_response_attributes(
    attributes: dict[str, Any],
    response: "ChatResponse | AgentRunResponse",
    duration: float | None = None,
) -> dict[str, Any]:
    """レスポンスからレスポンス属性を取得します。"""
    if response.response_id:
        attributes[OtelAttr.RESPONSE_ID] = response.response_id
    finish_reason = getattr(response, "finish_reason", None)
    if not finish_reason:
        finish_reason = (
            getattr(response.raw_representation, "finish_reason", None) if response.raw_representation else None
        )
    if finish_reason:
        attributes[OtelAttr.FINISH_REASONS] = json.dumps([finish_reason.value])
    if model_id := getattr(response, "model_id", None):
        attributes[SpanAttributes.LLM_RESPONSE_MODEL] = model_id
    if usage := response.usage_details:
        if usage.input_token_count:
            attributes[OtelAttr.INPUT_TOKENS] = usage.input_token_count
        if usage.output_token_count:
            attributes[OtelAttr.OUTPUT_TOKENS] = usage.output_token_count
    if duration:
        attributes[Meters.LLM_OPERATION_DURATION] = duration
    return attributes


GEN_AI_METRIC_ATTRIBUTES = (
    OtelAttr.OPERATION,
    OtelAttr.PROVIDER_NAME,
    SpanAttributes.LLM_REQUEST_MODEL,
    SpanAttributes.LLM_RESPONSE_MODEL,
    OtelAttr.ADDRESS,
    OtelAttr.PORT,
)


def _capture_response(
    span: trace.Span,
    attributes: dict[str, Any],
    operation_duration_histogram: "metrics.Histogram | None" = None,
    token_usage_histogram: "metrics.Histogram | None" = None,
) -> None:
    """指定されたspanにレスポンスを設定します。"""
    span.set_attributes(attributes)
    attrs: dict[str, Any] = {k: v for k, v in attributes.items() if k in GEN_AI_METRIC_ATTRIBUTES}
    if token_usage_histogram and (input_tokens := attributes.get(OtelAttr.INPUT_TOKENS)):
        token_usage_histogram.record(
            input_tokens, attributes={**attrs, SpanAttributes.LLM_TOKEN_TYPE: OtelAttr.T_TYPE_INPUT}
        )
    if token_usage_histogram and (output_tokens := attributes.get(OtelAttr.OUTPUT_TOKENS)):
        token_usage_histogram.record(output_tokens, {**attrs, SpanAttributes.LLM_TOKEN_TYPE: OtelAttr.T_TYPE_OUTPUT})
    if operation_duration_histogram and (duration := attributes.get(Meters.LLM_OPERATION_DURATION)):
        if OtelAttr.ERROR_TYPE in attributes:
            attrs[OtelAttr.ERROR_TYPE] = attributes[OtelAttr.ERROR_TYPE]
        operation_duration_histogram.record(duration, attributes=attrs)


class EdgeGroupDeliveryStatus(Enum):
    """edge group配信ステータス値の列挙型。"""

    DELIVERED = "delivered"
    DROPPED_TYPE_MISMATCH = "dropped type mismatch"
    DROPPED_TARGET_MISMATCH = "dropped target mismatch"
    DROPPED_CONDITION_FALSE = "dropped condition evaluated to false"
    EXCEPTION = "exception"
    BUFFERED = "buffered"

    def __str__(self) -> str:
        """列挙型の文字列表現を返します。"""
        return self.value

    def __repr__(self) -> str:
        """enumの文字列表現を返します。"""
        return self.value


def workflow_tracer() -> "Tracer":
    """ワークフロートレーサーを取得するか、有効でない場合はno-opトレーサーを取得します。"""
    global OBSERVABILITY_SETTINGS
    return get_tracer() if OBSERVABILITY_SETTINGS.ENABLED else trace.NoOpTracer()


def create_workflow_span(
    name: str,
    attributes: Mapping[str, str | int] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> "_AgnosticContextManager[trace.Span]":
    """汎用のワークフロースパンを作成します。"""
    return workflow_tracer().start_as_current_span(name, kind=kind, attributes=attributes)


def create_processing_span(
    executor_id: str,
    executor_type: str,
    message_type: str,
    source_trace_contexts: list[dict[str, str]] | None = None,
    source_span_ids: list[str] | None = None,
) -> "_AgnosticContextManager[trace.Span]":
    """オプションのリンクを持つexecutor処理スパンを作成します。

    処理スパンは現在のワークフロースパンの子として作成され、
    因果関係追跡のためにソースのパブリッシュスパンにリンク（ネストではなく）されます。
    これにより、ファンインシナリオでの複数リンクがサポートされます。

    """
    # ネストせずに因果関係のためにソーススパンへのリンクを作成します。
    links: list[trace.Link] = []
    if source_trace_contexts and source_span_ids:
        # すべてのソーススパンへのリンクを作成します（複数ソースのファンインをサポート）。
        for trace_context, span_id in zip(source_trace_contexts, source_span_ids, strict=False):
            # リンク作成に失敗した場合はリンクなしで続行します（グレースフルデグラデーション）。
            with contextlib.suppress(ValueError, TypeError, AttributeError):
                # トレースコンテキストからトレースIDとスパンIDを抽出します。 これは簡易的なアプローチであり、本番環境ではより堅牢な解析が望まれます。
                traceparent = trace_context.get("traceparent", "")
                if traceparent:
                    # traceparentフォーマット: "00-{trace_id}-{parent_span_id}-{trace_flags}"
                    parts = traceparent.split("-")
                    if len(parts) >= 3:
                        trace_id_hex = parts[1]
                        # パブリッシュスパンから保存されたsource_span_idを使用し、 リンク用のスパンコンテキストを作成します。
                        span_context = trace.SpanContext(
                            trace_id=int(trace_id_hex, 16),
                            span_id=int(span_id, 16),
                            is_remote=True,
                        )
                        links.append(trace.Link(span_context))

    return workflow_tracer().start_as_current_span(
        OtelAttr.EXECUTOR_PROCESS_SPAN,
        kind=trace.SpanKind.INTERNAL,
        attributes={
            OtelAttr.EXECUTOR_ID: executor_id,
            OtelAttr.EXECUTOR_TYPE: executor_type,
            OtelAttr.MESSAGE_TYPE: message_type,
        },
        links=links,
    )


def create_edge_group_processing_span(
    edge_group_type: str,
    edge_group_id: str | None = None,
    message_source_id: str | None = None,
    message_target_id: str | None = None,
    source_trace_contexts: list[dict[str, str]] | None = None,
    source_span_ids: list[str] | None = None,
) -> "_AgnosticContextManager[trace.Span]":
    """オプションのリンクを持つエッジグループ処理スパンを作成します。

    エッジグループ処理スパンは、メッセージ配信前のエッジランナー内の処理操作を追跡します。
    これには条件チェックやルーティングの決定が含まれます。
    trace.Linksは不要なネストなしに因果関係追跡を提供します。

    Args:
        edge_group_type: エッジグループのタイプ（クラス名）。
        edge_group_id: エッジグループのユニークID。
        message_source_id: 処理中のメッセージのソースID。
        message_target_id: 処理中のメッセージのターゲットID。
        source_trace_contexts: リンク用のソーススパンからのオプショントレースコンテキスト。
        source_span_ids: リンク用のオプションのソーススパンID。

    """
    attributes: dict[str, str] = {
        OtelAttr.EDGE_GROUP_TYPE: edge_group_type,
    }

    if edge_group_id is not None:
        attributes[OtelAttr.EDGE_GROUP_ID] = edge_group_id
    if message_source_id is not None:
        attributes[OtelAttr.MESSAGE_SOURCE_ID] = message_source_id
    if message_target_id is not None:
        attributes[OtelAttr.MESSAGE_TARGET_ID] = message_target_id

    # ネストせずに因果関係のためにソーススパンへのリンクを作成します。
    links: list[trace.Link] = []
    if source_trace_contexts and source_span_ids:
        # すべてのソーススパンへのリンクを作成します（複数ソースのファンインをサポート）。
        for trace_context, span_id in zip(source_trace_contexts, source_span_ids, strict=False):
            try:
                # トレースコンテキストからトレースIDとスパンIDを抽出します。 これは簡易的なアプローチであり、本番環境ではより堅牢な解析が望まれます。
                traceparent = trace_context.get("traceparent", "")
                if traceparent:
                    # traceparentフォーマット: "00-{trace_id}-{parent_span_id}-{trace_flags}"
                    parts = traceparent.split("-")
                    if len(parts) >= 3:
                        trace_id_hex = parts[1]
                        # パブリッシュスパンから保存されたsource_span_idを使用し、 リンク用のスパンコンテキストを作成します。
                        span_context = trace.SpanContext(
                            trace_id=int(trace_id_hex, 16),
                            span_id=int(span_id, 16),
                            is_remote=True,
                        )
                        links.append(trace.Link(span_context))
            except (ValueError, TypeError, AttributeError):
                # リンク作成に失敗した場合はリンクなしで続行します（グレースフルデグラデーション）。
                pass

    return workflow_tracer().start_as_current_span(
        OtelAttr.EDGE_GROUP_PROCESS_SPAN,
        kind=trace.SpanKind.INTERNAL,
        attributes=attributes,
        links=links,
    )
