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


def _run_extract(source: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "workspace"
        root.mkdir()
        python_file = root / "sample.py"
        # given: テスト用のソースコードを一時ディレクトリに作成する
        python_file.write_text(source, encoding="utf-8")
        output_path = Path(tmpdir) / "extracted.jsonl"
        # when: 抽出処理を実行する
        extract.run(
            {
                "root": str(root),
                "output": str(output_path),
                "include_log_messages": False,
                "verbose": False,
                "exclude": [],
            }
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            return []
        # then: 出力の JSONL を読み込んで検証データとして返す
        return [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]


class ExtractDocstringTests(unittest.TestCase):
    def test_module_docstring_extracted(self) -> None:
        # given: モジュール docstring を含むソース
        records = _run_extract("""\"\"\"Module summary\"\"\"\n""")
        # when: 抽出結果を取得する
        self.assertTrue(any(item["kind"] == "module_docstring" for item in records))
        # then: docstring が期待通りの内容で取得される
        module_doc = next(item for item in records if item["kind"] == "module_docstring")
        self.assertEqual(module_doc["text"], "Module summary")

    def test_function_docstring_extracted(self) -> None:
        # given: 関数 docstring を含むソース
        records = _run_extract(
            """\n\ndef sample():\n    \"\"\"Details\"\"\"\n    return True\n"""
        )
        # when: 抽出結果を確認する
        self.assertTrue(any(item["kind"] == "function_docstring" for item in records))
        # then: 関数 docstring の内容が取得される
        function_doc = next(item for item in records if item["kind"] == "function_docstring")
        self.assertEqual(function_doc["text"], "Details")


class ExtractCommentTests(unittest.TestCase):
    def test_block_comment_keeps_original_block(self) -> None:
        # given: 複数行ブロックコメントを含むソース
        records = _run_extract("# first\n# second\nvalue = 1\n")
        # when: ブロックコメントのメタ情報を取得する
        block = next(item for item in records if item["kind"] == "comment")
        self.assertEqual(block["text"], "first\nsecond")
        meta = block["meta"]
        # then: ブロックコメントの属性が保持されている
        self.assertEqual(meta["comment_type"], "block")
        self.assertEqual(meta["line_count"], 2)
        self.assertEqual(meta["original_block"], "# first\n# second\n")

    def test_inline_comment_has_original_line(self) -> None:
        # given: インラインコメントを含むソース
        records = _run_extract("value = 1  # inline\n")
        # when: インラインコメントのメタ情報を確認する
        inline = next(item for item in records if item["kind"] == "comment")
        meta = inline["meta"]
        # then: 元の1行コメント文字列が保持される
        self.assertEqual(meta["comment_type"], "inline")
        self.assertEqual(meta["original_line"], "value = 1  # inline\n")
        self.assertEqual(inline["text"], "inline")

    def test_no_comments_results_in_empty_output(self) -> None:
        # given: コメントを含まないソース
        records = _run_extract("value = 1\n")
        # then: 抽出結果が空になる
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
