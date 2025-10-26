"""JSONL翻訳フローの実装。"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from .exception import TranslationError, TranslationParseError
from .translation import (
    TranslationItem,
    count_tokens,
    stream_batches,
    translate_with_request,
)
from logger import logger as root_logger


logger = root_logger.getChild("translation_pipeline")


def run_translation(settings: Dict[str, Any]) -> None:
    """翻訳処理を実行してJSONLを生成する。"""
    input_path = Path(settings["input"]).resolve()
    output_path = Path(settings["output"]).resolve()
    failed_path = Path(settings["failed_output"]).resolve()
    system_prompt: str = settings["system_prompt"]
    limit = settings.get("limit")
    batch_size = settings.get("batch_size")
    request_func: Callable[[str, Sequence[str]], str] = settings["request_func"]

    records = list(load_records(input_path, limit))
    if not records:
        logger.info("翻訳対象がありません input=%s", input_path)
        return
    total_tokens = sum(count_tokens(item.record["text"]) for item in records)
    average_tokens = (total_tokens / len(records)) if records else 0
    logger.info(
        "翻訳対象件数:%d 総トークン:%d 平均トークン:%.2f",
        len(records),
        total_tokens,
        average_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    success_count = 0
    failure_count = 0
    with output_path.open("w", encoding="utf-8") as success_handle, failed_path.open(
        "w",
        encoding="utf-8",
    ) as failed_handle:
        batches = stream_batches(records, system_prompt=system_prompt)
        if batch_size:
            batches = _limit_batch_size(batches, batch_size)
        for batch in batches:
            texts = [item.record["text"] for item in batch]
            try:
                translations = translate_with_request(request_func, system_prompt, texts)
            except TranslationError as error:
                failure_count += _record_failures(failed_handle, batch, error)
                logger.warning(
                    "翻訳要求に失敗しました batch_size=%d error=%s",
                    len(batch),
                    error,
                )
                continue
            if len(translations) != len(batch):
                error = TranslationParseError("翻訳結果件数が入力と一致しません")
                failure_count += _record_failures(failed_handle, batch, error)
                logger.warning(
                    "翻訳結果件数不一致 batch_size=%d result_size=%d",
                    len(batch),
                    len(translations),
                )
                logger.debug("翻訳結果: %s", translations)
                continue
            for item, translated in zip(batch, translations):
                write_translated_record(success_handle, item.record, translated)
                success_count += 1
    duration = perf_counter() - start
    logger.info(
        "翻訳完了 成功:%d 失敗:%d 処理時間:%.3f秒",
        success_count,
        failure_count,
        duration,
    )


def load_records(path: Path, limit: Optional[int]) -> Iterator[TranslationItem]:
    """JSONLからレコードを読み込む。"""
    with path.open("r", encoding="utf-8") as handle:
        processed = 0
        for index, line in enumerate(handle):
            if limit is not None and processed >= limit:
                break
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if "text" not in data:
                continue
            yield TranslationItem(record=data, index=index)
            processed += 1


def write_translated_record(handle, record: Dict[str, Any], translated: str) -> None:
    """翻訳済みレコードをJSONLに書き込む。"""
    payload = {
        "path": record.get("path"),
        "kind": record.get("kind"),
        "original": record.get("text"),
        "translated": translated,
        "meta": record.get("meta", {}),
    }
    handle.write(json.dumps(payload, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def _record_failures(handle, batch: Sequence[TranslationItem], error: Exception) -> int:
    """未処理レコードを保存する。"""
    for item in batch:
        payload = {
            "record": item.record,
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
        }
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")
        handle.flush()
    return len(batch)


def _limit_batch_size(
    batches: Iterator[List[TranslationItem]],
    size: int,
) -> Iterator[List[TranslationItem]]:
    """トークン制限とは別にバッチサイズを制限する。"""
    for batch in batches:
        for index in range(0, len(batch), size):
            yield batch[index : index + size]
