import asyncio
import importlib
import json
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

translate = importlib.import_module("translate")


class TranslateDummyTests(unittest.TestCase):
    def test_dummy_translator_writes_translated_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            meta = {
                "comment_type": "block",
                "original_block": "# hello\n",
                "position": {"start": {"line": 1, "column": 0}, "end": {"line": 1, "column": 7}},
            }
            entry = {"path": "sample.py", "kind": "comment", "text": "hello", "meta": meta}
            # given: 1件のコメントを含む入力JSONLを用意する
            input_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

            output_path = Path(tmpdir) / "translated.jsonl"
            failed_path = Path(tmpdir) / "failed.jsonl"

            # when: dummy翻訳モードで翻訳処理を実行する
            asyncio.run(
                translate.run(
                    {
                        "input": str(input_path),
                        "output": str(output_path),
                        "failed_output": str(failed_path),
                        "is_mock": True,
                        "batch_size": 2,
                        "system_prompt": "test",
                    }
                )
            )

            # then: 成功出力に mock 翻訳が書き出され、失敗出力は空である
            translated_lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(translated_lines), 1)
            payload = json.loads(translated_lines[0])
            self.assertEqual(payload["original"], "hello")
            self.assertEqual(payload["translated"], "hello (mock)")
            self.assertEqual(payload["meta"], meta)

            self.assertEqual(failed_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
