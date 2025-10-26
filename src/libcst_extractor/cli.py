"""コマンドラインインタフェース定義。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .apply import run_apply
from .pipeline import run_extraction
from .translation_pipeline import run_translation
from logger import logger as base_logger


log = base_logger.getChild("cli")

DEFAULT_SYSTEM_PROMPT = """あなたは、OSSの日本人コントリビューターであり、翻訳者です。\nユーザーが提示した、英語で書かれたdocstringやコメントを、ルールに従い、日本語に翻訳してください。\n翻訳結果は、出力フォーマットに従い、JSON形式で出力してください。\n\n### ルール\n- 固有名詞は、翻訳しなくていいです。（例：Azure AI Project など）\n- 変数名(ex.agent_name)やpython言語にまつわる定義、各パラメータなど、markdownのコードブロックなどのmd表現は翻訳しなくていいです。\n- docstringにおいて、”Examples”, “Keyword Args” などの見出しとなる部分は、どの言語でも共通して英語が用いられる慣習があるので、そのままとして良いです。\n- 翻訳不要単語が存在します。翻訳不要単語はカタカナや和訳せずにそのまま英語で表現してください。\n\n### 翻訳不要単語\nAgent, ID, Thread, Chat, Client, Class, Context, Import, Export, Key, Token, Secret, Config, Prompt, Request, Response, State, Message, Optional, None, Middleware, Executor\n\n### 出力フォーマット\n{\n  \"translations\": [\n    {\n      \"index\": <入力と同じindex>,\n      \"translation\": \"<翻訳後の和訳文>\"\n    }\n  ]\n}\n- translations 配列の要素数は入力と同じにしてください。\n- index は入力値をそのまま写してください。\n"""

COMMAND_EXTRACT = "extract"
COMMAND_TRANSLATE = "translate"
COMMAND_APPLY = "apply"


def create_parser() -> argparse.ArgumentParser:
    """トップレベルの引数パーサを生成する。"""
    parser = argparse.ArgumentParser(
        prog="libcst-extractor",
        description="libcstを用いた抽出と翻訳パイプライン",
    )
    subparsers = parser.add_subparsers(dest="command")
    _configure_extract_parser(subparsers)
    _configure_translate_parser(subparsers)
    _configure_apply_parser(subparsers)
    return parser


def _configure_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    """抽出サブコマンドを設定する。"""
    parser = subparsers.add_parser(COMMAND_EXTRACT, help="docstringやコメントを抽出する")
    parser.add_argument("root", nargs="?", default=".", help="解析するルートディレクトリ")
    parser.add_argument(
        "--output",
        default="extracted.jsonl",
        help="抽出結果を書き出すJSONLファイルパス",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="並列処理に利用するスレッド数(未指定時はCPU数)",
    )
    parser.add_argument(
        "--include-runtime-messages",
        action="store_true",
        help="print文とlogger呼び出しのメッセージも抽出対象に含める",
    )
    parser.add_argument(
        "--include-debug-logs",
        action="store_true",
        help="logger.debugのメッセージも抽出する",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="抽出対象から除外するパスのglobパターン(複数指定可)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログ出力レベル",
    )


def _configure_translate_parser(subparsers: argparse._SubParsersAction) -> None:
    """翻訳サブコマンドを設定する。"""
    parser = subparsers.add_parser(COMMAND_TRANSLATE, help="抽出結果JSONLを翻訳する")
    parser.add_argument("input", help="翻訳対象のJSONLファイル")
    parser.add_argument(
        "--output",
        default="translated.jsonl",
        help="翻訳済みJSONLの出力パス",
    )
    parser.add_argument(
        "--failed-output",
        default="unprocessed.jsonl",
        help="翻訳できなかったレコードの保存先",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭から処理する件数を制限する",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="システムプロンプトを記載したファイルパス",
    )
    parser.add_argument(
        "--translator-kind",
        default="azure",
        choices=["azure", "dummy"],
        help="利用する翻訳クライアントの種類",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="1バッチあたりの最大件数 (省略時はトークン制限で自動調整)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログ出力レベル",
    )


def _configure_apply_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(COMMAND_APPLY, help="翻訳結果をソースに適用する")
    parser.add_argument("input", help="翻訳済みJSONLファイル")
    parser.add_argument(
        "--output-dir",
        default="translated_sources",
        help="生成したソースを書き出すディレクトリ",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="元ソースのルートディレクトリ",
    )
    parser.add_argument(
        "--mode",
        default="indirect",
        choices=["indirect", "direct"],
        help="適用方法（directは未実装）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログ出力レベル",
    )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI引数を解析する。"""
    if argv is None:
        argv = sys.argv[1:]
    normalized = _normalize_argv(argv)
    parser = create_parser()
    args = parser.parse_args(normalized)
    command = args.command or COMMAND_EXTRACT
    if command == COMMAND_EXTRACT:
        return {
            "command": COMMAND_EXTRACT,
            "settings": _build_extract_settings(args),
        }
    if command == COMMAND_TRANSLATE:
        return {
            "command": COMMAND_TRANSLATE,
            "settings": _build_translate_settings(args),
        }
    return {
        "command": COMMAND_APPLY,
        "settings": _build_apply_settings(args),
    }


def _normalize_argv(argv: Optional[Sequence[str]]) -> Sequence[str]:
    """旧仕様互換のために引数を補正する。"""
    if argv is None or not argv:
        return [COMMAND_EXTRACT]
    values = list(argv)
    first = values[0]
    if first in {COMMAND_EXTRACT, COMMAND_TRANSLATE, COMMAND_APPLY}:
        return values
    return [COMMAND_EXTRACT, *values]


def _build_extract_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """抽出処理用の設定辞書を組み立てる。"""
    include_debug_logs = getattr(args, "include_debug_logs", False)
    include_runtime_messages = getattr(args, "include_runtime_messages", False)
    if include_debug_logs and not include_runtime_messages:
        log.debug("--include-debug-logsが指定されたためprint/logger抽出を有効化します")
        include_runtime_messages = True
    return {
        "root": Path(args.root).resolve(),
        "output": Path(args.output).resolve(),
        "jobs": getattr(args, "jobs", None),
        "include_runtime_messages": include_runtime_messages,
        "include_debug_logs": include_debug_logs,
        "exclude": list(getattr(args, "exclude", [])),
        "log_level": args.log_level,
    }


def _build_translate_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """翻訳処理用の設定辞書を組み立てる。"""
    system_prompt = _load_system_prompt(args.system_prompt_file)
    request_func = _create_request_func(args.translator_kind)
    return {
        "input": Path(args.input).resolve(),
        "output": Path(args.output).resolve(),
        "failed_output": Path(args.failed_output).resolve(),
        "limit": args.limit,
        "system_prompt": system_prompt,
        "request_func": request_func,
        "log_level": args.log_level,
        "batch_size": args.batch_size,
    }


def _build_apply_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "input": Path(args.input).resolve(),
        "output_dir": Path(args.output_dir).resolve(),
        "root": Path(args.root).resolve(),
        "mode": args.mode,
        "log_level": args.log_level,
    }


def _load_system_prompt(path: Optional[str]) -> str:
    """システムプロンプトの文字列を取得する。"""
    if path is None:
        return DEFAULT_SYSTEM_PROMPT
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8")


def _create_request_func(kind: str) -> Callable[[str, Sequence[str]], str]:
    """翻訳リクエスト関数を取得する。"""
    if kind == "dummy":
        def request(prompt: str, texts: Sequence[str]) -> str:
            items = [{"translation": f"{text} (mock)"} for text in texts]
            return json.dumps(items, ensure_ascii=False)

        return request
    if kind == "azure":
        from translate import request_translations

        return request_translations
    raise ValueError(f"未対応のtranslator-kindです: {kind}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLIエントリポイント。"""
    parsed = parse_arguments(argv)
    command = parsed["command"]
    settings = dict(parsed["settings"])
    log_level_name = settings.pop("log_level", "INFO")
    level = getattr(logging, log_level_name.upper(), logging.INFO)
    base_logger.setLevel(level)
    if command == COMMAND_EXTRACT:
        run_extraction(settings)
    elif command == COMMAND_TRANSLATE:
        run_translation(settings)
    else:
        run_apply(settings)
