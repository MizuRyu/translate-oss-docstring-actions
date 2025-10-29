# Copyright (c) Microsoft. All rights reserved.

"""Agent Framework DevUI - OpenAI互換APIサーバーを備えたデバッグインターフェース。"""

import importlib.metadata
import logging
import webbrowser
from typing import Any

from ._server import DevServer
from .models import AgentFrameworkRequest, OpenAIError, OpenAIResponse, ResponseStreamEvent
from .models._discovery_models import DiscoveryResponse, EntityInfo, EnvVarRequirement

logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # 開発モード用のフォールバック


def serve(
    entities: list[Any] | None = None,
    entities_dir: str | None = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    auto_open: bool = False,
    cors_origins: list[str] | None = None,
    ui_enabled: bool = True,
    tracing_enabled: bool = False,
) -> None:
    """Agent Framework DevUIをシンプルなAPIで起動する。

    Args:
        entities: インメモリ登録用のエンティティリスト（IDは自動生成される）
        entities_dir: エンティティをスキャンするディレクトリ
        port: サーバーを起動するポート
        host: サーバーをバインドするホスト
        auto_open: ブラウザを自動で開くかどうか
        cors_origins: 許可されたCORSオリジンのリスト
        ui_enabled: UIを有効にするかどうか
        tracing_enabled: OpenTelemetryトレースを有効にするかどうか

    """
    import re

    import uvicorn

    # セキュリティのためにhostパラメータを早期に検証する
    if not re.match(r"^(localhost|127\.0\.0\.1|0\.0\.0\.0|[a-zA-Z0-9.-]+)$", host):
        raise ValueError(f"Invalid host: {host}. Must be localhost, IP address, or valid hostname")

    # portパラメータを検証する
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}. Must be integer between 1 and 65535")

    # 有効な場合はトレース環境変数を設定する
    if tracing_enabled:
        import os

        # ユーザーによって既に設定されていなければ設定する
        if not os.environ.get("ENABLE_OTEL"):
            os.environ["ENABLE_OTEL"] = "true"
            logger.info("Set ENABLE_OTEL=true for tracing")

        if not os.environ.get("ENABLE_SENSITIVE_DATA"):
            os.environ["ENABLE_SENSITIVE_DATA"] = "true"
            logger.info("Set ENABLE_SENSITIVE_DATA=true for tracing")

        if not os.environ.get("OTLP_ENDPOINT"):
            os.environ["OTLP_ENDPOINT"] = "http://localhost:4317"
            logger.info("Set OTLP_ENDPOINT=http://localhost:4317 for tracing")

    # 直接パラメータでサーバーを作成する
    server = DevServer(
        entities_dir=entities_dir, port=port, host=host, cors_origins=cors_origins, ui_enabled=ui_enabled
    )

    # 提供された場合はインメモリエンティティを登録する
    if entities:
        logger.info(f"Registering {len(entities)} in-memory entities")
        # サーバー起動時に登録するためにエンティティを保存する
        server._pending_entities = entities

    app = server.get_app()

    if auto_open:

        def open_browser() -> None:
            import http.client
            import re
            import time

            # セキュリティのためにhostとportを検証する
            if not re.match(r"^(localhost|127\.0\.0\.1|0\.0\.0\.0|[a-zA-Z0-9.-]+)$", host):
                logger.warning(f"Invalid host for auto-open: {host}")
                return

            if not isinstance(port, int) or not (1 <= port <= 65535):
                logger.warning(f"Invalid port for auto-open: {port}")
                return

            # ヘルスエンドポイントをチェックしてサーバーの準備完了を待つ
            browser_url = f"http://{host}:{port}"

            for _ in range(30):  # 15 second timeout (30 * 0.5s)
                try:
                    # 安全な接続処理のためにhttp.clientを使用する（標準ライブラリ）
                    conn = http.client.HTTPConnection(host, port, timeout=1)
                    try:
                        conn.request("GET", "/health")
                        response = conn.getresponse()
                        if response.status == 200:
                            webbrowser.open(browser_url)
                            return
                    finally:
                        conn.close()
                except (http.client.HTTPException, OSError, TimeoutError):
                    pass
                time.sleep(0.5)

            # フォールバック：タイムアウト後にブラウザを開く
            webbrowser.open(browser_url)

        import threading

        threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"Starting Agent Framework DevUI on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    """devuiコマンドのCLIエントリポイント。"""
    from ._cli import main as cli_main

    cli_main()


# メインの公開APIをエクスポートする
__all__ = [
    "AgentFrameworkRequest",
    "DevServer",
    "DiscoveryResponse",
    "EntityInfo",
    "EnvVarRequirement",
    "OpenAIError",
    "OpenAIResponse",
    "ResponseStreamEvent",
    "main",
    "serve",
]
