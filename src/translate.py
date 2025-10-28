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
from log_utils import log_progress, log_stage_start, log_summary
from util import count_tokens, logger

load_dotenv()


# 定数定義
DEFAULT_EXCLUDE_TERMS = (
    "Agent, ID, Thread, Chat, Client, Class, Context, Import, "
    "Export, Key, Token, Secret, Config, Prompt, Request, Response, "
    "State, Message, Optional, None, Middleware, Executor"
)
MAX_OVERSIZED_TOKENS = 50000  # この値を超えるエントリは異常データとして除外

PROMPT_TEMPLATE = dedent(
    """\
    あなたは、OSSの日本人コントリビューターであり、翻訳者です。
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
        prompt_text = PROMPT_TEMPLATE.format(terms=exclude_terms.strip())
    else:
        prompt_text = DEFAULT_PROMPT

    cfg = {
        "input_path": Path(settings["input"]).resolve(),
        "output_path": Path(settings["output"]).resolve(),
        "failed_output": Path(settings["failed_output"]).resolve(),
        "limit": settings.get("limit"),
        "system_prompt": settings.get("system_prompt", prompt_text),
        "is_mock": bool(settings.get("is_mock", False)),
        "enable_fallback": bool(settings.get("enable_fallback", True)),
    }

    entries = _load_entries(cfg["input_path"], cfg["limit"])
    if not entries:
        logger.info("翻訳対象がありません input=%s", cfg["input_path"])
        return

    cfg["failed_output"].write_text("", encoding="utf-8")

    batches, oversized_entries = _build_batches_within_token_limit(
        entries,
        cfg["system_prompt"],
        cfg["failed_output"],
        cfg["enable_fallback"],
    )

    cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)
    cfg["failed_output"].parent.mkdir(parents=True, exist_ok=True)

    log_stage_start(
        "Translate",
        f"Items: {len(entries)}, Batches: {len(batches)}, Oversized: {len(oversized_entries)}",
    )

    start = perf_counter()
    success_count = 0
    failure_items: List[Dict[str, Any]] = []

    tasks: List[asyncio.Task] = []
    async with asyncio.TaskGroup() as tg:
        for index, batch in enumerate(batches, start=1):
            detail = f"{len(batch)} items"
            log_progress("Translate", index, len(batches), f"Batch {index}", detail)
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

    # トークン超過エントリのFallback処理
    oversized_success = 0
    if oversized_entries and cfg["enable_fallback"]:
        logger.info(
            "トークン超過エントリをFallback処理します: %d件", len(oversized_entries)
        )
        oversized_success = await _process_oversized_entries(
            oversized_entries,
            cfg["output_path"],
            cfg["failed_output"],
            cfg["system_prompt"],
            cfg["is_mock"],
        )

    duration = perf_counter() - start
    primary_requests = stats.get("primary_requests", 0)
    fallback_requests = stats.get("fallback_requests", 0)
    rate_hits = stats.get("rate_limit_hits", 0)
    timeouts = stats.get("timeouts", 0)
    
    log_summary("Translate", {
        "Total Items": len(entries),
        "Success": success_count,
        "Oversized Success": oversized_success,
        "Failed": len(failure_items),
        "Oversized Failed": len(oversized_entries) - oversized_success,
        "Batches": len(batches),
        "LLM Requests": primary_requests + fallback_requests,
        "Fallback Count": fallback_requests,
        "Rate Limits": rate_hits,
        "Timeouts": timeouts,
        "Duration": f"{duration:.3f}s",
    })


def _load_entries(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """入力JSONLを読み込む"""

    with path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle]

    if limit and limit > 0:
        return items[:limit]

    return items


async def _process_oversized_entries(
    oversized_entries: List[Dict[str, Any]],
    output_path: Path,
    failed_output: Path,
    system_prompt: str,
    is_mock: bool,
) -> int:
    """
    トークン超過エントリをFallbackで処理する
    
    トークン数が2,500を超えるエントリを1件ずつFallbackモデルで処理する。
    50,000トークンを超える異常データはunprocessedに出力する。
    
    Args:
        oversized_entries: トークン超過エントリリスト
        output_path: 成功した翻訳の出力先
        failed_output: 失敗した翻訳の出力先
        system_prompt: システムプロンプト
        is_mock: モックモード（常にFalse想定）
    
    Returns:
        成功した件数
    """
    success_count = 0
    
    for entry in oversized_entries:
        entry_tokens = entry.get("tokens", 0)
        entry_path = entry.get("path", "unknown")
        entry_line = entry.get("meta", {}).get("line_start", 0)
        
        try:
            # 50,000トークン超は異常データとして除外
            if entry_tokens > MAX_OVERSIZED_TOKENS:
                logger.warning(
                    "異常に大きいエントリをスキップ: %s:%d (tokens=%d, max=%d)",
                    entry_path,
                    entry_line,
                    entry_tokens,
                    MAX_OVERSIZED_TOKENS,
                )
                entry_with_error = {
                    **entry,
                    "error": f"oversized_entry: {entry_tokens} tokens exceeds {MAX_OVERSIZED_TOKENS}",
                }
                with failed_output.open("a", encoding="utf-8") as failed:
                    failed.write(json.dumps(entry_with_error, ensure_ascii=False))
                    failed.write("\n")
                continue
            
            # Fallbackモデルで1件ずつ処理（バッチではなく個別問い合わせ）
            batch_result = await translate_batch(
                system_prompt,
                [entry],
                is_mock=is_mock,
            )
            translations, error, _ = batch_result
            
            if translations and len(translations) > 0:
                translated = translations[0]
            else:
                raise Exception(f"Fallback translation failed: {error}")
            
            # 成功した翻訳を出力
            payload = {
                "path": entry["path"],
                "kind": entry["kind"],
                "original": entry["text"],
                "translated": translated,
                "meta": entry["meta"],
            }
            
            with output_path.open("a", encoding="utf-8") as success:
                success.write(json.dumps(payload, ensure_ascii=False))
                success.write("\n")
            
            success_count += 1
            logger.info(
                "Oversized Entry Processed: %s:%d (tokens=%d)",
                entry_path,
                entry_line,
                entry_tokens,
            )
            
        except Exception as e:
            # Fallback失敗時はunprocessedに出力
            logger.error(
                "Oversized Entry Failed: %s:%d - %s",
                entry_path,
                entry_line,
                str(e),
            )
            entry_with_error = {**entry, "error": f"fallback_failed: {str(e)}"}
            with failed_output.open("a", encoding="utf-8") as failed:
                failed.write(json.dumps(entry_with_error, ensure_ascii=False))
                failed.write("\n")
    
    return success_count


def _write_failures(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def _build_batches_within_token_limit(
    entries: Sequence[Dict[str, Any]],
    system_prompt: str,
    failed_output: Path,
    enable_fallback: bool = True,
    max_tokens: int = MAX_REQUEST_TOKENS,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    エントリーをtoken上限内でバッチ分割する
    
    Args:
        entries: 翻訳対象エントリ
        system_prompt: システムプロンプト
        failed_output: 失敗エントリの出力先
        enable_fallback: Fallback処理を有効化するか
        max_tokens: バッチあたりの最大トークン数
    
    Returns:
        (batches, oversized_entries): バッチリストとトークン超過エントリリスト
    """
    system_prompt_tokens = count_tokens(system_prompt)
    batches: List[List[Dict[str, Any]]] = []
    oversized_entries: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    token_usage = 0
    token_usage += system_prompt_tokens

    for entry in entries:
        text = entry["text"]
        tokens = count_tokens(text) + REQUEST_OVERHEAD_TOKENS

        if tokens > max_tokens:
            # 1つのテキストデータがMAX_TOKEN超過する場合
            entry_with_error = {**entry, "error": "token_limit_exceeded", "tokens": tokens}
            logger.warning(
                "Oversized Entry Detected path=%s kind=%s tokens=%d",
                entry.get("path"),
                entry.get("kind"),
                tokens,
            )
            
            if enable_fallback:
                # Fallback有効時はキューに追加
                oversized_entries.append(entry)
            else:
                # Fallback無効時はunprocessedに出力
                with failed_output.open("a", encoding="utf-8") as failed_handle:
                    failed_handle.write(json.dumps(entry_with_error, ensure_ascii=False))
                    failed_handle.write("\n")
            continue

        should_flush = current and token_usage + tokens > max_tokens

        if should_flush:
            batches.append(current)
            current = []
            token_usage = system_prompt_tokens

        current.append(entry)
        token_usage += tokens

    if current:
        batches.append(current)

    return batches, oversized_entries
