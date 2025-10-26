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

extract = importlib.import_module("extract")
translate = importlib.import_module("translate")
replace = importlib.import_module("replace")


class PipelineScenarioTests(unittest.TestCase):
    def test_full_mock_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "project"
            workspace.mkdir()
            module_path = workspace / "module.py"
            # given: docstring とコメントを含むテストモジュール
            module_path.write_text(
                """\"\"\"Summary\"\"\"\n\n# multi line comment part one\n# multi line comment part two\n\n\ndef example():\n    \"\"\"Details\"\"\"\n    value = 1  # inline note\n    return value\n""",
                encoding="utf-8",
            )

            extracted = Path(tmpdir) / "extracted.jsonl"
            translated = Path(tmpdir) / "translated.jsonl"
            failed = Path(tmpdir) / "failed.jsonl"
            output_dir = Path(tmpdir) / "output"

            # when: 抽出 → mock翻訳 → 置換を通しで実行する
            extract.run(
                {
                    "root": str(workspace),
                    "output": str(extracted),
                    "include_log_messages": False,
                    "verbose": False,
                    "exclude": [],
                }
            )

            translate.run(
                {
                    "input": str(extracted),
                    "output": str(translated),
                    "failed_output": str(failed),
                    "translator_kind": "dummy",
                    "batch_size": 4,
                    "system_prompt": "test",
                }
            )

            replace.run(
                {
                    "input": str(translated),
                    "output_dir": str(output_dir),
                    "root": str(workspace),
                    "mode": "indirect",
                }
            )

            target_file = output_dir / "module.py"
            content = target_file.read_text(encoding="utf-8")

            # then: 翻訳済みの docstring とコメントが反映され、失敗出力は空である
            self.assertIn("Summary (mock)", content)
            self.assertIn("Details (mock)", content)
            self.assertIn("# multi line comment part one multi line comment part two (mock)", content)
            self.assertIn("inline note (mock)", content)

            failures = failed.read_text(encoding="utf-8")
            self.assertEqual(failures, "")

            # then: JSONL にも期待した種類のレコードが含まれている
            translated_records = [json.loads(line) for line in translated.read_text(encoding="utf-8").splitlines()]
            kinds = {item["kind"] for item in translated_records}
            self.assertIn("module_docstring", kinds)
            self.assertIn("function_docstring", kinds)
            self.assertIn("comment", kinds)


if __name__ == "__main__":
    unittest.main()
