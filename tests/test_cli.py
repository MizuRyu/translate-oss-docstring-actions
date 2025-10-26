import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cli  # noqa: E402


class CLIMockScenarioTests(unittest.TestCase):
    def test_translate_command_with_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "extracted.jsonl"
            output_path = tmpdir_path / "translated.jsonl"
            failed_path = tmpdir_path / "unprocessed.jsonl"

            record = {
                "path": "sample.py",
                "kind": "comment",
                "text": "Hello world",
                "meta": {"comment_type": "block", "original_block": "# Hello world\n"},
            }
            input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            cli.main(
                [
                    "translate",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--failed-output",
                    str(failed_path),
                    "--mock",
                ]
            )

            translated_lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(translated_lines), 1)
            payload = json.loads(translated_lines[0])
            self.assertEqual(payload["translated"], "Hello world (mock)")
            failed_content = failed_path.read_text(encoding="utf-8")
            self.assertTrue(failed_content == "" or failed_content == "\n")


if __name__ == "__main__":
    unittest.main()
