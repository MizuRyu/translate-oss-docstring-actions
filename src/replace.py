from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List

import libcst as cst
from libcst import metadata

from util import logger


def run(settings: Dict[str, Any]) -> None:
    """翻訳結果をソースコードへ適用する"""

    input_path = Path(settings["input"]).resolve()
    output_dir = Path(settings["output_dir"]).resolve()
    root = Path(settings["root"]).resolve()
    mode = settings.get("mode", "indirect")
    if mode != "indirect":
        raise NotImplementedError("direct モードは未実装です")

    records = _load_records(input_path)
    if not records:
        logger.info("適用対象がありません input=%s", input_path)
        return

    grouped: Dict[Path, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(Path(record["path"]).resolve(), []).append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, items in grouped.items():
        translated = _apply_to_file(path, items)
        relative = _relative_path(path, root)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(translated, encoding="utf-8")
        docstrings = sum(1 for item in items if item["kind"].endswith("docstring"))
        comments = sum(1 for item in items if item["kind"] == "comment")
        logger.info(
            "\nGenerated Complete\nPath: %s\nDocstrings: %d\nComments: %d",
            relative,
            docstrings,
            comments,
        )


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """翻訳結果JSONLを読み込みリストで返す"""

    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _apply_to_file(path: Path, records: List[Dict[str, Any]]) -> str:
    """ファイル全体に翻訳結果を適用したコードを返す"""

    source = path.read_text(encoding="utf-8")
    module = cst.parse_module(source)
    wrapper = metadata.MetadataWrapper(module)
    doc_mapping = _build_docstring_mapping(records)
    updated = _apply_docstrings(wrapper.module, doc_mapping)
    code = updated.code
    comments = [item for item in records if item["kind"] == "comment"]
    if comments:
        code = _apply_comments(source, code, comments)
    return code


def _build_docstring_mapping(records: Iterable[Dict[str, Any]]) -> Dict[tuple, Dict[str, str]]:
    """docstring置換用のマッピングを構築する"""

    mapping: Dict[tuple, Dict[str, str]] = {}
    for record in records:
        if not record["kind"].endswith("docstring"):
            continue
        position = record["meta"]["position"]
        key = (
            position["start"]["line"],
            position["start"]["column"],
            position["end"]["line"],
            position["end"]["column"],
        )
        mapping[key] = {"original": record["original"], "translated": record["translated"]}
    return mapping


def _apply_docstrings(module: cst.Module, mapping: Dict[tuple, Dict[str, str]]) -> cst.Module:
    """docstring置換を行ったCSTモジュールを返す"""

    class Transformer(cst.CSTTransformer):
        METADATA_DEPENDENCIES = (metadata.PositionProvider,)

        def __init__(self) -> None:
            self.applied = 0

        def leave_SimpleString(self, original_node: cst.SimpleString, updated_node: cst.SimpleString) -> cst.CSTNode:
            position = self.get_metadata(metadata.PositionProvider, original_node)
            key = (
                position.start.line,
                position.start.column,
                position.end.line,
                position.end.column,
            )
            record = mapping.get(key)
            if record is None:
                return updated_node
            if _literal_value(original_node) != record["original"]:
                return updated_node
            new_literal = _build_docstring_literal(original_node, record["translated"])
            self.applied += 1
            return updated_node.with_changes(value=new_literal)

    transformer = Transformer()
    updated = metadata.MetadataWrapper(module).visit(transformer)
    if transformer.applied != len(mapping):
        logger.warning(
            "docstringの置換件数が一致しません expected=%d actual=%d",
            len(mapping),
            transformer.applied,
        )
    return updated


def _apply_comments(original_code: str, code: str, comments: Iterable[Dict[str, Any]]) -> str:
    """コメントを翻訳済みに差し替える"""

    ordered = sorted(comments, key=_comment_sort_key)
    result = code
    search_index = 0
    for record in ordered:
        meta = record["meta"]
        comment_type = meta.get("comment_type")
        if comment_type == "inline":
            original_line = meta.get("original_line", "")
            if not original_line:
                logger.warning("inlineコメント情報が不足しています path=%s", record.get("path"))
                continue
            new_line = _format_inline_line(original_line, record["translated"])
            index = result.find(original_line, search_index)
            if index == -1:
                logger.warning("コメント反映対象が見つかりません path=%s", record.get("path"))
                continue
            result = result[:index] + new_line + result[index + len(original_line) :]
            search_index = index + len(new_line)
        else:
            original_block = meta.get("original_block", "")
            if not original_block:
                logger.warning("コメント反映対象が見つかりません path=%s", record.get("path"))
                continue
            index = result.find(original_block, search_index)
            if index == -1:
                logger.warning("コメント反映対象が見つかりません path=%s", record.get("path"))
                continue
            indent = _block_indent(original_block)
            trailing = original_block.endswith("\n")
            width = max(10, 88 - len(indent) - 2)
            new_block = _format_comment_block(record["translated"], indent, width, trailing)
            result = result[:index] + new_block + result[index + len(original_block) :]
            search_index = index + len(new_block)
    return result


def _comment_sort_key(record: Dict[str, Any]) -> tuple[int, int]:
    """コメント適用順を決めるソートキー"""

    position = record.get("meta", {}).get("position", {})
    start = position.get("start", {})
    line = start.get("line", 0)
    column = start.get("column", 0)
    return line, column


def _format_comment_block(text: str, indent: str, width: int, trailing_newline: bool) -> str:
    """ブロックコメントを整形して返す"""

    wrapper = textwrap.TextWrapper(width=width, break_long_words=False, break_on_hyphens=False)
    lines = wrapper.wrap(text) or [""]
    formatted = [f"{indent}# {line}" if line else f"{indent}#" for line in lines]
    block = "\n".join(formatted)
    if trailing_newline:
        block += "\n"
    return block


def _format_inline_line(original_line: str, translated: str) -> str:
    """インラインコメント1行を翻訳結果に置き換える"""

    newline = ""
    if original_line.endswith("\n"):
        newline = "\n"
        original_line = original_line[:-1]
    hash_index = original_line.find("#")
    if hash_index == -1:
        return original_line + newline
    prefix = original_line[: hash_index]
    return f"{prefix}# {translated}{newline}"


def _block_indent(block: str) -> str:
    """コメントブロックのインデント文字列を抽出する"""

    for line in block.splitlines():
        if "#" in line:
            return line[: line.find("#")]
    return ""


def _relative_path(path: Path, root: Path) -> Path:
    """出力ディレクトリに対する相対パスを計算する"""

    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _literal_value(node: cst.SimpleString) -> str:
    try:
        return ast.literal_eval(node.value)
    except Exception:
        return node.value.strip('"\'')


def _build_docstring_literal(node: cst.SimpleString, text: str) -> str:
    prefix = ""
    value = node.value
    index = 0
    while index < len(value) and value[index].lower() in {"r", "u", "f", "b"}:
        prefix += value[index]
        index += 1
    quote = value[index : index + 3]
    if quote in {"\"\"\"", "'''"}:
        try:
            original = ast.literal_eval(value)
        except Exception:
            original = ""
        if original:
            text = _reindent_docstring(text, original)
            suffix = _extract_newline_suffix(original)
            if suffix and not text.endswith(suffix):
                text = text.rstrip(" \t") + suffix
        escaped = text.replace(quote, "\\" + quote)
        return prefix + quote + escaped + quote
    quote = value[index]
    escaped = text.replace("\\", "\\\\").replace(quote, "\\" + quote)
    escaped = escaped.replace("\n", "\\n")
    return prefix + quote + escaped + quote


def _extract_newline_suffix(value: str) -> str:
    match = re.search(r"\n[ \t]*\Z", value)
    return match.group(0) if match else ""


def _reindent_docstring(text: str, original: str) -> str:
    lines = text.split("\n")
    if len(lines) <= 1:
        return text
    indent = _infer_docstring_indent(original)
    if not indent:
        return text
    result = [lines[0]]
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            result.append(indent.rstrip(" "))
        elif line.startswith(indent):
            result.append(line)
        else:
            result.append(f"{indent}{stripped}")
    return "\n".join(result)


def _infer_docstring_indent(original: str) -> str:
    lines = original.split("\n")
    for line in lines[1:]:
        if line.strip():
            prefix = len(line) - len(line.lstrip(" \t"))
            return line[:prefix]
    return ""
