from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from util import count_tokens


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する"""

    parser = argparse.ArgumentParser(
        description="JSONLファイルのトークンサマリを計算するツール",
    )
    parser.add_argument("--input", required=True, help="入力 JSONL ファイルのパス")
    parser.add_argument(
        "--output",
        required=True,
        help="サマリを書き出す JSON ファイルのパス",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="翻訳件数の上限（translation_limit）",
    )
    return parser.parse_args()


def summarize_tokens(input_path: Path, limit: int | None = None) -> dict[str, float | int]:
    """JSONLファイルからトークンサマリを計算する"""

    items = 0
    tokens = 0
    chars = 0
    limited_tokens = 0
    limited_chars = 0
    max_tokens = 0
    min_tokens = float('inf')
    
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            item_tokens = count_tokens(text)
            item_chars = len(text)
            
            tokens += item_tokens
            chars += item_chars
            items += 1
            max_tokens = max(max_tokens, item_tokens)
            min_tokens = min(min_tokens, item_tokens)
            
            if limit is None or items <= limit:
                limited_tokens += item_tokens
                limited_chars += item_chars
    
    average = tokens / items if items else 0.0
    actual_items = min(items, limit) if limit else items
    
    return {
        "total_items": items,
        "total_tokens": tokens,
        "total_chars": chars,
        "average_tokens": average,
        "max_tokens": max_tokens if items > 0 else 0,
        "min_tokens": int(min_tokens) if items > 0 else 0,
        "translation_limit": limit,
        "actual_items": actual_items,
        "actual_tokens": limited_tokens,
        "actual_chars": limited_chars,
    }


def main() -> None:
    """スクリプトエントリーポイント"""

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_tokens(input_path, args.limit)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    print("=" * 44)
    print("📊 トークン使用量サマリ")
    print("=" * 44)
    if args.limit:
        print(f"抽出件数: {summary['actual_items']:,}件 (制限: {args.limit:,}件)")
    else:
        print(f"抽出件数: {summary['total_items']:,}件")
    print(f"総トークン数: {summary['actual_tokens']:,}")
    print(f"平均トークン数: {summary['average_tokens']:.2f}")
    print(f"最大トークン数: {summary['max_tokens']:,}")
    print(f"最小トークン数: {summary['min_tokens']:,}")
    print("=" * 44)


if __name__ == "__main__":
    main()
