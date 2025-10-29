# Copyright (c) Microsoft. All rights reserved.

"""
Message Capture Script - デバッグ用メッセージフロー
- このスクリプトは、エージェントとワークフローが実行される際にサーバーから発行されるイベントの種類の参照を提供することを目的としています
"""

import asyncio
import contextlib
import http.client
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from openai import OpenAI

from agent_framework_devui import DevServer

logger = logging.getLogger(__name__)


def start_server() -> tuple[str, Any]:
    """samplesディレクトリでサーバーを起動。"""
    # samplesが移動された後の更新されたパスでsamplesディレクトリを取得
    current_dir = Path(__file__).parent
    # samplesは現在python/samples/getting_started/devuiにあります
    samples_dir = current_dir.parent.parent.parent / "samples" / "getting_started" / "devui"

    if not samples_dir.exists():
        raise RuntimeError(f"Samples directory not found: {samples_dir}")

    logger.info(f"Using samples directory: {samples_dir}")

    # 簡略化されたパラメータでサーバーを作成して起動
    server = DevServer(
        entities_dir=str(samples_dir.resolve()),
        host="127.0.0.1",
        port=8085,  # Use different port
        ui_enabled=False,
    )

    app = server.get_app()

    server_config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8085,
        # log_level="info"、  # トレース設定を確認するために詳細に
    )
    server_instance = uvicorn.Server(server_config)

    def run_server():
        asyncio.run(server_instance.serve())

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # サーバーの起動を待機
    time.sleep(5)  # 待機時間を延長

    # リトライでサーバーが稼働していることを確認
    max_retries = 10
    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 8085, timeout=5)
            try:
                conn.request("GET", "/health")
                response = conn.getresponse()
                if response.status == 200:
                    break
            finally:
                conn.close()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"Server failed to start after {max_retries} attempts: {e}") from e

    return "http://127.0.0.1:8085", server_instance


def capture_agent_stream_with_tracing(client: OpenAI, agent_id: str, scenario: str = "success") -> list[dict[str, Any]]:
    """エージェントのストリーミングイベントをキャプチャ。"""

    try:
        stream = client.responses.create(
            model=agent_id,  # DevUI uses model field as entity_id
            input="Tell me about the weather in Tokyo. I want details.",
            stream=True,
        )

        events = []
        for event in stream:
            # イベントオブジェクト全体をシリアライズ
            try:
                event_dict = json.loads(event.model_dump_json())
            except Exception:
                # model_dump_jsonが失敗した場合はdict変換にフォールバック
                event_dict = event.__dict__ if hasattr(event, "__dict__") else str(event)

            events.append(event_dict)

            # そのまますべてをキャプチャ
            if len(events) >= 200:  # Increased limit
                break

        return events

    except Exception as e:
        # エラー情報をイベントとして返す
        error_event = {
            "type": "error",
            "scenario": scenario,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time(),
        }
        return [error_event]


def capture_workflow_stream_with_tracing(
    client: OpenAI, workflow_id: str, scenario: str = "success"
) -> list[dict[str, Any]]:
    """ワークフローのストリーミングイベントをキャプチャ。"""

    try:
        stream = client.responses.create(
            model=workflow_id,  # DevUI uses model field as entity_id
            input=(
                "Process this spam detection workflow with multiple emails: "
                "'Buy now!', 'Hello mom', 'URGENT: Click here!'"
            ),
            stream=True,
        )

        events = []
        for event in stream:
            # イベントオブジェクト全体をシリアライズ
            try:
                event_dict = json.loads(event.model_dump_json())
            except Exception:
                # model_dump_jsonが失敗した場合はdict変換にフォールバック
                event_dict = event.__dict__ if hasattr(event, "__dict__") else str(event)

            events.append(event_dict)

            # そのまますべてをキャプチャ
            if len(events) >= 200:  # Increased limit
                break

        return events

    except Exception as e:
        # エラー情報をイベントとして返す
        error_event = {
            "type": "error",
            "scenario": scenario,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time(),
            "entity_type": "workflow",
        }
        return [error_event]


def main():
    """メインキャプチャスクリプト - 成功と失敗の両方のシナリオをテスト。"""

    # セットアップ
    output_dir = Path(__file__).parent / "captured_messages"
    output_dir.mkdir(exist_ok=True)

    # サーバーを起動
    base_url, server_instance = start_server()

    try:
        # 成功シナリオ用のOpenAIクライアントを作成
        client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy-key")

        # エンティティを発見
        conn = http.client.HTTPConnection("127.0.0.1", 8085, timeout=10)
        try:
            conn.request("GET", "/v1/entities")
            response = conn.getresponse()
            response_data = response.read().decode("utf-8")
            entities = json.loads(response_data)["entities"]
        finally:
            conn.close()

        all_results = {}

        # 各エンティティをテスト
        for entity in entities:
            entity_type = entity["type"]
            entity_id = entity["id"]

            if entity_type == "agent":
                events = capture_agent_stream_with_tracing(client, entity_id, "success")
            elif entity_type == "workflow":
                events = capture_workflow_stream_with_tracing(client, entity_id, "success")
            else:
                continue

            all_results[f"{entity_type}_{entity_id}"] = {"entity_info": entity, "events": events}
        # 結果を保存
        file_path = output_dir / "entities_stream_events.json"
        with open(file_path, "w") as f:
            json.dump(
                {"timestamp": time.time(), "server_type": "DevServer", "entities_tested": all_results},
                f,
                indent=2,
                default=str,
            )

    finally:
        # サーバーをクリーンアップ
        with contextlib.suppress(Exception):
            server_instance.should_exit = True


if __name__ == "__main__":
    main()
