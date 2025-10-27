"""ログ出力の共通ユーティリティ"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("comment_translator")


def log_progress(
    stage: str,
    current: int,
    total: int,
    item: str,
    detail: str = "",
) -> None:
    """
    統一されたプログレス表示
    
    Args:
        stage: ステージ名（Extract, Translate, Replace等）
        current: 現在の処理番号
        total: 全体の件数
        item: 処理対象アイテム名
        detail: 追加詳細情報
    
    Example:
        log_progress("Extract", 5, 10, "module.py", "15 items found")
        # Output: [Extract] 5/10 (50%) module.py - 15 items found
    """
    percentage = int(current / total * 100) if total > 0 else 0
    msg = f"[{stage}] {current}/{total} ({percentage}%) {item}"
    if detail:
        msg += f" - {detail}"
    logger.info(msg)


def log_summary(stage: str, stats: Dict[str, Any]) -> None:
    """
    統一されたサマリー表示
    
    Args:
        stage: ステージ名
        stats: 統計情報の辞書
    
    Example:
        log_summary("Extract", {
            "Total Files": 150,
            "Extracted Items": 450,
            "Total Tokens": 12500,
            "Duration": "1.23s"
        })
    """
    logger.info("=" * 60)
    logger.info(f"[{stage}] Complete")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def log_stage_start(stage: str, message: str = "") -> None:
    """
    ステージ開始ログ
    
    Args:
        stage: ステージ名
        message: 追加メッセージ
    """
    logger.info("=" * 60)
    logger.info(f"[{stage}] Start")
    if message:
        logger.info(f"  {message}")
    logger.info("=" * 60)


def log_error(stage: str, message: str, details: Dict[str, Any] = None) -> None:
    """
    統一されたエラーログ
    
    Args:
        stage: ステージ名
        message: エラーメッセージ
        details: エラー詳細情報
    """
    logger.error("=" * 60)
    logger.error(f"[{stage}] Error")
    logger.error(f"  {message}")
    if details:
        for key, value in details.items():
            logger.error(f"  {key}: {value}")
    logger.error("=" * 60)
