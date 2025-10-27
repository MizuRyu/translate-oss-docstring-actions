from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from extract import run as run_extract
from replace import run as run_replace
from translate import run as run_translate, DEFAULT_PROMPT
from util import logger


def create_parser() -> argparse.ArgumentParser:
    """CLIエントリポイント用のパーサーを生成する"""

    parser = argparse.ArgumentParser(
        prog="comment-translator",
        description="libcst を利用した docstring / コメントの抽出・翻訳パイプライン",
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract
    extract = subparsers.add_parser("extract", help="docstring・コメントを抽出してJSONLに書き出す")
    extract.add_argument("root", nargs="?", default=".")
    extract.add_argument("--output", default="out/extracted.jsonl")
    extract.add_argument("--jobs", type=int, default=None, help="未使用オプション（互換用）")
    extract.add_argument(
        "--include-log-messages",
        "--include-runtime-messages",
        action="store_true",
        dest="include_log_messages",
        default=False,
    )
    extract.add_argument(
        "--verbose",
        "--include-debug-logs",
        action="store_true",
        dest="verbose",
        default=False,
    )
    extract.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    extract.add_argument("--log-level", default="INFO")

    # translate
    translate = subparsers.add_parser("translate", help="抽出JSONLを翻訳する")
    translate.add_argument("input")
    translate.add_argument("--output", default="out/translated.jsonl")
    translate.add_argument("--failed-output", default="out/unprocessed.jsonl")
    translate.add_argument("--limit", type=int, default=None)
    translate.add_argument("--system-prompt-file", default=None)
    translate.add_argument("--exclude-terms", default=None)
    translate.add_argument("--mock", action="store_true", dest="is_mock", help="LLMを呼び出さずモックで翻訳する")
    translate.add_argument("--batch-size", type=int, default=5)
    translate.add_argument("--log-level", default="INFO")

    # replace
    apply = subparsers.add_parser("replace", help="翻訳済みJSONLをソースへ適用する")
    apply.add_argument("input")
    apply.add_argument("--output-dir", default="out/translated_sources")
    apply.add_argument("--root", default=".")
    apply.add_argument("--mode", default="indirect")
    apply.add_argument("--log-level", default="INFO")

    return parser


def parse_args(argv: Optional[Sequence[str]]) -> Dict[str, Any]:
    """引数を解析してサブコマンドごとの設定辞書を構築する"""

    parser = create_parser()
    args = parser.parse_args(argv)
    command = args.command or "extract"
    if command == "extract":
        return {
            "command": "extract",
            "settings": {
                "root": Path(args.root).resolve(),
                "output": Path(args.output).resolve(),
                "exclude": args.exclude,
                "include_log_messages": args.include_log_messages,
                "verbose": args.verbose,
                "log_level": args.log_level,
            },
        }
    if command == "translate":
        prompt = DEFAULT_PROMPT
        if args.system_prompt_file:
            prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
        return {
            "command": "translate",
            "settings": {
                "input": Path(args.input).resolve(),
                "output": Path(args.output).resolve(),
                "failed_output": Path(args.failed_output).resolve(),
                "limit": args.limit,
                "system_prompt": prompt,
                "exclude_terms": args.exclude_terms,
                "batch_size": args.batch_size,
                "is_mock": args.is_mock,
                "log_level": args.log_level,
            },
        }
    return {
        "command": "replace",
        "settings": {
            "input": Path(args.input).resolve(),
            "output_dir": Path(args.output_dir).resolve(),
            "root": Path(args.root).resolve(),
            "mode": args.mode,
            "log_level": args.log_level,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLIのエントリーポイント"""

    config = parse_args(argv)
    settings = dict(config["settings"])
    level = settings.pop("log_level", "INFO")
    os.environ["LOG_LEVEL"] = level
    logger.setLevel(level.upper())

    command = config["command"]
    if command == "extract":
        run_extract(settings)
    elif command == "translate":
        asyncio.run(run_translate(settings))
    else:
        run_replace(settings)


if __name__ == "__main__":  # pragma: no cover
    main()
