"""抽出処理の実行パイプライン。"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, TextIO

from .extraction import extract_text_items
from .translation import count_tokens
from logger import logger as root_logger


logger = root_logger.getChild("pipeline")


def run_extraction(settings: Dict[str, Any]) -> None:
    """設定に従って抽出処理を実行する。"""
    root = settings["root"]
    output_path = settings["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = list(collect_python_files(root, settings.get("exclude", [])))
    total_files = len(files)
    if total_files == 0:
        logger.info("解析対象のPythonファイルが見つかりませんでした")
        return
    logger.info("抽出対象ファイル総数:%d", total_files)
    start = perf_counter()
    stats = {"files": 0, "files_with_items": 0, "items": 0, "text_length": 0, "tokens": 0}
    max_workers = settings.get("jobs") or os.cpu_count() or 1
    logger.info("抽出開始 対象ファイル数:%d スレッド:%d", len(files), max_workers)
    with output_path.open("w", encoding="utf-8") as handle:
        writer = create_writer(handle)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for path, records in executor.map(lambda p: process_single_file(p, settings), files):
                stats["files"] += 1
                if records:
                    stats["files_with_items"] += 1
                stats["items"] += len(records)
                stats["text_length"] += sum(len(rec["text"]) for rec in records)
                stats["tokens"] += sum(count_tokens(rec["text"]) for rec in records)
                writer(records)
    duration = perf_counter() - start
    average_tokens = (stats["tokens"] / stats["items"]) if stats["items"] else 0
    summary = {
        "scanned_files": stats["files"],
        "files_with_items": stats["files_with_items"],
        "items": stats["items"],
        "total_chars": stats["text_length"],
        "total_tokens": stats["tokens"],
        "avg_tokens_per_item": round(average_tokens, 2),
        "duration_sec": round(duration, 3),
    }
    logger.info("抽出完了 %s", json.dumps(summary, ensure_ascii=False))


def collect_python_files(root: Path, excludes: Sequence[str]) -> Iterator[Path]:
    """除外パターンを考慮してPythonファイルを列挙する。"""
    for path in sorted(root.rglob("*.py")):
        if path.is_file() and not should_exclude(path, root, excludes):
            yield path


def should_exclude(path: Path, root: Path, excludes: Sequence[str]) -> bool:
    """指定されたglobパターンに一致する場合に除外する。"""
    relative = path.relative_to(root)
    normalized = str(relative)
    for pattern in excludes:
        if relative.match(pattern) or path.match(pattern) or normalized.startswith(pattern):
            return True
    return False


def process_single_file(path: Path, settings: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]]]:
    """単一ファイルの抽出結果を取得する。"""
    text = path.read_text(encoding="utf-8")
    options = {
        "include_runtime_messages": settings.get("include_runtime_messages", False),
        "include_debug_logs": settings.get("include_debug_logs", False),
    }
    records = extract_text_items(text, path, options)
    return path, records


def create_writer(handle: TextIO):
    """JSONLへ1レコードずつ書き込むクロージャを生成する。"""

    def writer(records: Iterable[Dict[str, Any]]) -> None:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
        handle.flush()

    return writer
