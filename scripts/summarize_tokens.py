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
    return parser.parse_args()


def summarize_tokens(input_path: Path) -> dict[str, float]:
    """JSONLファイルからトークンサマリを計算する"""

    items = 0
    tokens = 0
    chars = 0
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            tokens += count_tokens(text)
            chars += len(text)
            items += 1
    average = tokens / items if items else 0.0
    return {
        "items": items,
        "tokens": tokens,
        "chars": chars,
        "average_tokens": average,
    }


def main() -> None:
    """スクリプトエントリーポイント"""

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_tokens(input_path)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
