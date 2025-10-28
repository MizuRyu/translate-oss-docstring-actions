import asyncio
import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

translate = importlib.import_module("translate")
util = importlib.import_module("util")


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


class TranslateOversizedTests(unittest.TestCase):
    """トークン超過エントリのFallback処理テスト"""

    def test_build_batches_with_oversized_entry_fallback_enabled(self) -> None:
        """トークン超過エントリがFallback有効時にoversized_entriesに追加される"""
        # 通常エントリ（短いテキスト）
        normal_entry = {
            "path": "normal.py",
            "kind": "docstring",
            "text": "Short text",
            "meta": {"line_start": 1},
        }
        # トークン超過エントリ（非常に長いテキスト）
        # 約3000トークンを生成するため、長い文字列を作成
        oversized_text = " ".join(["word"] * 10000)  # 約10,000ワード
        oversized_entry = {
            "path": "oversized.py",
            "kind": "docstring",
            "text": oversized_text,
            "meta": {"line_start": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            failed_path = Path(tmpdir) / "failed.jsonl"
            failed_path.write_text("", encoding="utf-8")

            batches, oversized_entries = translate._build_batches_within_token_limit(
                [normal_entry, oversized_entry],
                "system prompt",
                failed_path,
                enable_fallback=True,
                max_tokens=2500,
            )

            # 通常エントリは1バッチに含まれる
            self.assertEqual(len(batches), 1)
            self.assertGreaterEqual(len(batches[0]), 1)
            
            # 少なくとも1つはoversized_entriesに追加される
            self.assertGreaterEqual(len(oversized_entries), 1)
            self.assertEqual(oversized_entries[0]["path"], "oversized.py")

            # failed.jsonlには書き込まれない
            self.assertEqual(failed_path.read_text(encoding="utf-8"), "")

    def test_build_batches_with_oversized_entry_fallback_disabled(self) -> None:
        """トークン超過エントリがFallback無効時にfailed.jsonlに出力される"""
        # トークン超過エントリ（非常に長いテキスト）
        oversized_text = " ".join(["word"] * 10000)  # 約10,000ワード
        oversized_entry = {
            "path": "oversized.py",
            "kind": "docstring",
            "text": oversized_text,
            "meta": {"line_start": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            failed_path = Path(tmpdir) / "failed.jsonl"
            failed_path.write_text("", encoding="utf-8")

            batches, oversized_entries = translate._build_batches_within_token_limit(
                [oversized_entry],
                "system prompt",
                failed_path,
                enable_fallback=False,
                max_tokens=2500,
            )

            # バッチには含まれない
            self.assertEqual(len(batches), 0)

            # oversized_entriesも空
            self.assertEqual(len(oversized_entries), 0)

            # failed.jsonlに書き込まれる
            failed_content = failed_path.read_text(encoding="utf-8")
            self.assertNotEqual(failed_content, "")
            failed_data = json.loads(failed_content.strip())
            self.assertEqual(failed_data["path"], "oversized.py")
            self.assertIn("error", failed_data)

    def test_process_oversized_entries_success(self) -> None:
        """_process_oversized_entriesが正常に処理できる"""
        # 3000トークンのエントリ（50,000未満）
        oversized_entry = {
            "path": "oversized.py",
            "kind": "docstring",
            "text": "Large docstring " * 200,
            "meta": {"line_start": 10, "col_offset": 4},
            "tokens": 3000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "translated.jsonl"
            failed_path = Path(tmpdir) / "failed.jsonl"
            output_path.write_text("", encoding="utf-8")
            failed_path.write_text("", encoding="utf-8")

            # モックモードで実行（LLM呼び出しなし）
            success_count = asyncio.run(
                translate._process_oversized_entries(
                    [oversized_entry],
                    output_path,
                    failed_path,
                    "system prompt",
                    is_mock=True,
                )
            )

            # 成功カウント確認
            self.assertEqual(success_count, 1)

            # translated.jsonlに書き込まれる
            translated_content = output_path.read_text(encoding="utf-8")
            self.assertNotEqual(translated_content, "")
            translated_data = json.loads(translated_content.strip())
            self.assertEqual(translated_data["path"], "oversized.py")
            self.assertIn("translated", translated_data)

            # failed.jsonlは空
            self.assertEqual(failed_path.read_text(encoding="utf-8"), "")

    def test_process_oversized_entries_exceeds_max_tokens(self) -> None:
        """50,000トークン超のエントリが異常データとして除外される"""
        # 50,000トークン超のエントリ
        huge_entry = {
            "path": "huge.py",
            "kind": "docstring",
            "text": "X" * 60000,
            "meta": {"line_start": 1, "col_offset": 0},
            "tokens": 60000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "translated.jsonl"
            failed_path = Path(tmpdir) / "failed.jsonl"
            output_path.write_text("", encoding="utf-8")
            failed_path.write_text("", encoding="utf-8")

            success_count = asyncio.run(
                translate._process_oversized_entries(
                    [huge_entry],
                    output_path,
                    failed_path,
                    "system prompt",
                    is_mock=True,
                )
            )

            # 成功カウントは0
            self.assertEqual(success_count, 0)

            # translated.jsonlは空
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")

            # failed.jsonlに異常データとして出力される
            failed_content = failed_path.read_text(encoding="utf-8")
            self.assertNotEqual(failed_content, "")
            failed_data = json.loads(failed_content.strip())
            self.assertEqual(failed_data["path"], "huge.py")
            self.assertIn("oversized_entry", failed_data["error"])

    def test_translate_run_with_enable_fallback_true(self) -> None:
        """translate.run()がenable_fallback=Trueで動作する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            # 通常エントリ
            normal_entry = {
                "path": "normal.py",
                "kind": "docstring",
                "text": "Normal docstring",
                "meta": {"line_start": 1, "col_offset": 0},
            }
            input_path.write_text(json.dumps(normal_entry, ensure_ascii=False) + "\n", encoding="utf-8")

            output_path = Path(tmpdir) / "translated.jsonl"
            failed_path = Path(tmpdir) / "failed.jsonl"

            asyncio.run(
                translate.run(
                    {
                        "input": str(input_path),
                        "output": str(output_path),
                        "failed_output": str(failed_path),
                        "is_mock": True,
                        "enable_fallback": True,
                        "system_prompt": "test",
                    }
                )
            )

            # 成功出力に翻訳結果がある
            translated_lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(translated_lines), 1)

    def test_translate_run_with_enable_fallback_false(self) -> None:
        """translate.run()がenable_fallback=Falseで動作する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            # 通常エントリ
            normal_entry = {
                "path": "normal.py",
                "kind": "docstring",
                "text": "Normal docstring",
                "meta": {"line_start": 1, "col_offset": 0},
            }
            input_path.write_text(json.dumps(normal_entry, ensure_ascii=False) + "\n", encoding="utf-8")

            output_path = Path(tmpdir) / "translated.jsonl"
            failed_path = Path(tmpdir) / "failed.jsonl"

            asyncio.run(
                translate.run(
                    {
                        "input": str(input_path),
                        "output": str(output_path),
                        "failed_output": str(failed_path),
                        "is_mock": True,
                        "enable_fallback": False,
                        "system_prompt": "test",
                    }
                )
            )

            # 成功出力に翻訳結果がある
            translated_lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(translated_lines), 1)


if __name__ == "__main__":
    unittest.main()
