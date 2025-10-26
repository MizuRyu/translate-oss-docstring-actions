import importlib
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

replace_module = importlib.import_module("replace")
_apply_comments = replace_module._apply_comments  # type: ignore[attr-defined]


class ReplaceCommentTests(unittest.TestCase):
    def test_replace_multiple_block_comments(self) -> None:
        original = """# region Tools\nvalue = 1\n# region Tools\nvalue = 2\n"""
        comments = [
            {
                "kind": "comment",
                "translated": "First block",
                "meta": {
                    "comment_type": "block",
                    "line_count": 1,
                    "original_block": "# region Tools\n",
                    "position": {
                        "start": {"line": 1, "column": 0},
                        "end": {"line": 1, "column": 14},
                    },
                },
            },
            {
                "kind": "comment",
                "translated": "Second block",
                "meta": {
                    "comment_type": "block",
                    "line_count": 1,
                    "original_block": "# region Tools\n",
                    "position": {
                        "start": {"line": 3, "column": 0},
                        "end": {"line": 3, "column": 14},
                    },
                },
            },
        ]

        # given: 同一文言のブロックコメントが複数あるソース
        replaced = _apply_comments(original, original, comments)

        expected = """# First block\nvalue = 1\n# Second block\nvalue = 2\n"""
        # then: 先頭から順に個別の翻訳が差し込まれる
        self.assertEqual(replaced, expected)

    def test_replace_inline_comments_in_order(self) -> None:
        original = """value = 1  # note\nvalue = 2  # note\n"""
        comments = [
            {
                "kind": "comment",
                "translated": "first",
                "meta": {
                    "comment_type": "inline",
                    "original_line": "value = 1  # note\n",
                    "position": {
                        "start": {"line": 1, "column": 0},
                        "end": {"line": 1, "column": 18},
                    },
                },
            },
            {
                "kind": "comment",
                "translated": "second",
                "meta": {
                    "comment_type": "inline",
                    "original_line": "value = 2  # note\n",
                    "position": {
                        "start": {"line": 2, "column": 0},
                        "end": {"line": 2, "column": 18},
                    },
                },
            },
        ]

        # when: インラインコメントを順番通り置換する
        replaced = _apply_comments(original, original, comments)

        expected = """value = 1  # first\nvalue = 2  # second\n"""
        # then: 各行に対応する翻訳が正しく反映される
        self.assertEqual(replaced, expected)


if __name__ == "__main__":
    unittest.main()
