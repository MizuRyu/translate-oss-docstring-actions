from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from textwrap import dedent

from dotenv import load_dotenv

from llm import (
    MAX_REQUEST_TOKENS,
    REQUEST_OVERHEAD_TOKENS,
    translate_batch,
)
from util import count_tokens, logger

load_dotenv()


DEFAULT_EXCLUDE_TERMS = (
    "Agent, ID, Thread, Chat, Client, Class, Context, Import, "
    "Export, Key, Token, Secret, Config, Prompt, Request, Response, "
    "State, Message, Optional, None, Middleware, Executor"
)

PROMPT_TEMPLATE = dedent(
    """    あなたは、OSSの日本人コントリビューターであり、翻訳者です。
    ユーザーが提示した、英語で書かれたdocstringやコメントを、ルールに従い、日本語に翻訳してください。
    翻訳結果は、出力フォーマットに従い、JSON形式で出力してください。

    ### ルール
    - 固有名詞は翻訳せずに残してください。
    - 変数名やPythonの構文は翻訳しないでください。
    - docstring内の見出し（Examples など）は英語のままにしてください。
    - 以下の単語はカタカナや和訳にせず、そのまま英語で書いてください。
      {terms}

    ### 出力フォーマット
    {{
      "translations": [
        {{
          "index": <入力index>,
          "translation": "<翻訳結果>"
        }}
      ]
    }}
    """
)

DEFAULT_PROMPT = PROMPT_TEMPLATE.format(terms=DEFAULT_EXCLUDE_TERMS)


async def run(settings: Dict[str, Any]) -> None:
    """翻訳ジョブ全体を調整する。

    Args:
        settings: CLIから渡される設定辞書。
    """

    exclude_terms = settings.get("exclude_terms")
    if exclude_terms:
        prompt_text = PROMPT_TEMPLATE.format(terms=str(exclude_terms))
    else:
        prompt_text = DEFAULT_PROMPT

    cfg = {
        "input_path": Path(settings["input"]).resolve(),
        "output_path": Path(settings["output"]).resolve(),
        "failed_output": Path(settings["failed_output"]).resolve(),
        "limit": settings.get("limit"),
        "system_prompt": settings.get("system_prompt", prompt_text),
        "batch_size": settings.get("batch_size") or 5,
        "is_mock": bool(settings.get("is_mock", False)),
    }

    entries = _load_entries(cfg["input_path"], cfg["limit"])
    if not entries:
        logger.info("翻訳対象がありません input=%s", cfg["input_path"])
        return

    cfg["failed_output"].write_text("", encoding="utf-8")

    batches = _build_batches_within_token_limit(
        entries,
        cfg["system_prompt"],
        cfg["batch_size"],
        cfg["failed_output"],
    )

    cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)
    cfg["failed_output"].parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    success_count = 0
    failure_items: List[Dict[str, Any]] = []

    tasks: List[asyncio.Task] = []
    async with asyncio.TaskGroup() as tg:
        for batch in batches:
            task = tg.create_task(
                translate_batch(
                    cfg["system_prompt"],
                    batch,
                    is_mock=cfg["is_mock"],
                )
            )
            tasks.append(task)

    batch_results = [task.result() for task in tasks]
    results: List[Optional[List[str]]] = []
    errors: List[Optional[Exception]] = []
    stats = {
        "primary_requests": 0,
        "fallback_requests": 0,
        "rate_limit_hits": 0,
        "timeouts": 0,
    }

    for translations, error, stat in batch_results:
        results.append(translations)
        errors.append(error)
        for key in stats:
            stats[key] += stat.get(key, 0)

    with cfg["output_path"].open("w", encoding="utf-8") as success:
        for index, batch in enumerate(batches):
            translations = results[index]
            error = errors[index]
            if translations is None:
                if error:
                    logger.warning(
                        "翻訳要求に失敗しました件数=%d error=%s", len(batch), error
                    )
                    failure_items.extend(
                        [{**item, "error": str(error)} for item in batch]
                    )
                else:
                    logger.warning("翻訳結果を取得できませんでした batch=%d", index)
                    failure_items.extend(batch)
                continue
            if len(translations) != len(batch):
                logger.warning(
                    "翻訳結果件数不一致 input=%d output=%d",
                    len(batch),
                    len(translations),
                )
                failure_items.extend(
                    [{**item, "error": "mismatched_response"} for item in batch]
                )
                continue
            for entry, translated in zip(batch, translations):
                payload = {
                    "path": entry["path"],
                    "kind": entry["kind"],
                    "original": entry["text"],
                    "translated": translated,
                    "meta": entry["meta"],
                }
                success.write(json.dumps(payload, ensure_ascii=False))
                success.write("\n")
                success_count += 1

    if failure_items:
        with cfg["failed_output"].open("a", encoding="utf-8") as failed:
            for entry in failure_items:
                failed.write(json.dumps(entry, ensure_ascii=False))
                failed.write("\n")

    duration = perf_counter() - start
    primary_requests = stats.get("primary_requests", 0)
    fallback_requests = stats.get("fallback_requests", 0)
    rate_hits = stats.get("rate_limit_hits", 0)
    timeouts = stats.get("timeouts", 0)
    logger.info(
        "\nSuccess: %d\nFailed: %d\nLLM Requests Count: %d\nFallback Count: %d\nRate Limits Count: %d\nTimeouts Count: %d\nDuration: %.3f秒",
        success_count,
        len(failure_items),
        primary_requests + fallback_requests,
        fallback_requests,
        rate_hits,
        timeouts,
        duration,
    )


def _load_entries(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if limit is not None and line_number >= limit:
                break
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _write_failures(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def _build_batches_within_token_limit(
    entries: Sequence[Dict[str, Any]],
    system_prompt: str,
    max_items: Optional[int],
    failed_output: Path,
    max_tokens: int = MAX_REQUEST_TOKENS,
) -> List[List[Dict[str, Any]]]:
    system_prompt_tokens = count_tokens(system_prompt)
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    token_usage = 0
    token_usage += system_prompt_tokens

    for entry in entries:
        text = entry["text"]
        tokens = count_tokens(text) + REQUEST_OVERHEAD_TOKENS

        if tokens > max_tokens:
            # 1つのテキストデータがMAX_TOKEN超過する場合は処理しない
            entry_with_error = {**entry, "error": "token_limit_exceeded", "tokens": tokens}
            logger.warning(
                "\nOversized Entry Skipped path=%s kind=%s tokens=%d",
                entry.get("path"),
                entry.get("kind"),
                tokens,
            )
            with failed_output.open("a", encoding="utf-8") as failed_handle:
                failed_handle.write(json.dumps(entry_with_error, ensure_ascii=False))
                failed_handle.write("\n")
            continue

        should_flush = current and (
            token_usage + tokens > max_tokens
            or (max_items is not None and len(current) >= max_items)
        )

        if should_flush:
            batches.append(current)
            current = []
            token_usage = system_prompt_tokens

        current.append(entry)
        token_usage += tokens

    if current:
        batches.append(current)

    return batches
