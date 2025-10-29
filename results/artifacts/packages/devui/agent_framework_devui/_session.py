# Copyright (c) Microsoft. All rights reserved.

"""Agent実行トラッキングのためのセッション管理。"""

import logging
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# 可読性向上のための型エイリアス
SessionData = dict[str, Any]
RequestRecord = dict[str, Any]
SessionSummary = dict[str, Any]


class SessionManager:
    """リクエストとContextのトラッキングのための実行セッションを管理する。"""

    def __init__(self) -> None:
        """セッションマネージャーを初期化する。"""
        self.sessions: dict[str, SessionData] = {}

    def create_session(self, session_id: str | None = None) -> str:
        """新しい実行セッションを作成する。

        Args:
            session_id: OptionalなセッションID。指定しない場合は新規生成される

        Returns:
            セッションID

        """
        if not session_id:
            session_id = str(uuid.uuid4())

        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(),
            "requests": [],
            "context": {},
            "active": True,
        }

        logger.debug(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> SessionData | None:
        """セッション情報を取得する。

        Args:
            session_id: セッションID

        Returns:
            セッションデータまたは見つからなければNone

        """
        return self.sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """セッションを閉じてクリーンアップする。

        Args:
            session_id: 閉じるセッションID

        """
        if session_id in self.sessions:
            self.sessions[session_id]["active"] = False
            logger.debug(f"Closed session: {session_id}")

    def add_request_record(
        self, session_id: str, entity_id: str, executor_name: str, request_input: Any, model_id: str
    ) -> str:
        """セッションにリクエスト記録を追加する。

        Args:
            session_id: セッションID
            entity_id: 実行されるエンティティのID
            executor_name: executorの名前
            request_input: リクエストの入力
            model_id: モデル名

        Returns:
            リクエストID

        """
        session = self.get_session(session_id)
        if not session:
            return ""

        request_record: RequestRecord = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "entity_id": entity_id,
            "executor": executor_name,
            "input": request_input,
            "model_id": model_id,
            "stream": True,
        }
        session["requests"].append(request_record)
        return str(request_record["id"])

    def update_request_record(self, session_id: str, request_id: str, updates: dict[str, Any]) -> None:
        """セッション内のリクエスト記録を更新する。

        Args:
            session_id: セッションID
            request_id: 更新するリクエストID
            updates: 適用する更新内容の辞書

        """
        session = self.get_session(session_id)
        if not session:
            return

        for request in session["requests"]:
            if request["id"] == request_id:
                request.update(updates)
                break

    def get_session_history(self, session_id: str) -> SessionSummary | None:
        """セッションの実行履歴を取得する。

        Args:
            session_id: セッションID

        Returns:
            セッション履歴または見つからなければNone

        """
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "active": session["active"],
            "request_count": len(session["requests"]),
            "requests": [
                {
                    "id": req["id"],
                    "timestamp": req["timestamp"].isoformat(),
                    "entity_id": req["entity_id"],
                    "executor": req["executor"],
                    "model": req["model"],
                    "input_length": len(str(req["input"])) if req["input"] else 0,
                    "execution_time": req.get("execution_time"),
                    "status": req.get("status", "unknown"),
                }
                for req in session["requests"]
            ],
        }

    def get_active_sessions(self) -> list[SessionSummary]:
        """アクティブなセッションのリストを取得する。

        Returns:
            アクティブなセッションのサマリーリスト

        """
        active_sessions = []

        for session_id, session in self.sessions.items():
            if session["active"]:
                active_sessions.append({
                    "session_id": session_id,
                    "created_at": session["created_at"].isoformat(),
                    "request_count": len(session["requests"]),
                    "last_activity": (
                        session["requests"][-1]["timestamp"].isoformat()
                        if session["requests"]
                        else session["created_at"].isoformat()
                    ),
                })

        return active_sessions

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """メモリリーク防止のため古いセッションをクリーンアップする。

        Args:
            max_age_hours: 保持するセッションの最大年齢（時間）

        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session["created_at"].timestamp() < cutoff_time:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            logger.debug(f"Cleaned up old session: {session_id}")

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
