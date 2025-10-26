"""翻訳結果をソースへ反映する間接置換処理。"""

from __future__ import annotations

import ast
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import libcst as cst
from libcst import metadata

from .translation import count_tokens
from logger import logger as root_logger

DOCSTRING_KINDS = {"module_docstring", "class_docstring", "function_docstring", "comment"}


logger = root_logger.getChild("apply")


@dataclass
class DocstringRecord:
    path: Path
    key: Tuple[int, int, int, int]
    kind: str
    original: str
    translated: str
    meta: Dict[str, Any]


def run_apply(settings: Dict[str, object]) -> None:
    """翻訳JSONLを読み込み、docstringを置換したファイルを生成する。"""
    input_path = Path(settings["input"]).resolve()
    output_dir = Path(settings["output_dir"]).resolve()
    root = Path(settings["root"]).resolve()
    mode = settings.get("mode", "indirect")
    if mode != "indirect":
        raise NotImplementedError("directモードは未実装です")
    records = list(_load_docstring_records(input_path))
    if not records:
        logger.info("docstringの翻訳対象が見つかりません")
        return
    grouped = _group_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_path, docstrings in grouped.items():
        rewritten = _rewrite_source(source_path, docstrings)
        relative = _relative_path(source_path, root)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewritten, encoding="utf-8")
        doc_count = sum(1 for item in docstrings if item.kind != "comment")
        comment_count = len(docstrings) - doc_count
        logger.info(
            "生成しました path=%s docstrings=%d comments=%d tokens=%d",
            relative,
            doc_count,
            comment_count,
            sum(count_tokens(item.translated) for item in docstrings),
        )


def _load_docstring_records(path: Path) -> Iterable[DocstringRecord]:
    """JSONLからdocstringレコードを抽出する。"""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            if data.get("kind") not in DOCSTRING_KINDS:
                continue
            meta = data.get("meta", {})
            position = meta.get("position")
            if position is None:
                continue
            start = position.get("start", {})
            end = position.get("end", {})
            key = (
                int(start.get("line", -1)),
                int(start.get("column", -1)),
                int(end.get("line", -1)),
                int(end.get("column", -1)),
            )
            if -1 in key:
                continue
            path_value = data.get("path")
            if not path_value:
                continue
            yield DocstringRecord(
                path=Path(path_value).resolve(),
                key=key,
                kind=str(data.get("kind", "")),
                original=str(data.get("original", "")),
                translated=str(data.get("translated", "")),
                meta=meta,
            )


def _group_records(records: Sequence[DocstringRecord]) -> Dict[Path, List[DocstringRecord]]:
    grouped: Dict[Path, List[DocstringRecord]] = {}
    for record in records:
        grouped.setdefault(record.path, []).append(record)
    return grouped


def _rewrite_source(path: Path, records: Sequence[DocstringRecord]) -> str:
    source = path.read_text(encoding="utf-8")
    module = cst.parse_module(source)
    wrapper = metadata.MetadataWrapper(module)
    doc_records = [record for record in records if record.kind != "comment"]
    comment_records = [
        record
        for record in records
        if record.kind == "comment" and record.meta.get("comment_type") == "block"
    ]
    mapping = {record.key: record for record in doc_records}

    class Transformer(cst.CSTTransformer):
        METADATA_DEPENDENCIES = (metadata.PositionProvider,)

        def __init__(self) -> None:
            self.applied = 0

        def leave_SimpleString(
            self,
            original_node: cst.SimpleString,
            updated_node: cst.SimpleString,
        ) -> cst.CSTNode:
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
            if _evaluate_string(original_node.value) != record.original:
                return updated_node
            new_literal = _build_literal(original_node.value, record.translated)
            self.applied += 1
            return updated_node.with_changes(value=new_literal)

    transformer = Transformer()
    new_module = wrapper.visit(transformer)
    if transformer.applied != len(doc_records):
        logger.warning(
            "docstringの置換件数が一致しません path=%s expected=%d actual=%d",
            path,
            len(doc_records),
            transformer.applied,
        )
    code = new_module.code
    if not comment_records:
        return code
    return _rewrite_comment_blocks(code, comment_records)


def _evaluate_string(value: str) -> str:
    try:
        return ast.literal_eval(value)
    except Exception:  # pragma: no cover - 異常値は未変換
        return value


def _build_literal(template: str, text: str) -> str:
    prefix = ""
    index = 0
    while index < len(template) and template[index].lower() in {"r", "u", "f", "b"}:
        prefix += template[index]
        index += 1
    quote = template[index: index + 3]
    if quote in {"\"\"\"", "'''"}:
        try:
            original_value = ast.literal_eval(template)
        except Exception:  # pragma: no cover - 安全側のフォールバック
            original_value = ""
        if original_value:
            text = _reindent_docstring(text, original_value)
            suffix = _extract_trailing_newline_suffix(original_value)
            if suffix and not text.endswith(suffix):
                text = text.rstrip(" \t") + suffix
        escaped = text.replace(quote, "\\" + quote)
        return prefix + quote + escaped + quote
    quote = template[index]
    escaped = text.replace("\\", "\\\\").replace(quote, "\\" + quote)
    escaped = escaped.replace("\n", "\\n")
    return prefix + quote + escaped + quote


def _relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return Path(path.name)


def _rewrite_comment_blocks(source_code: str, comments: Sequence[DocstringRecord]) -> str:
    result = source_code
    for record in comments:
        original_block = record.meta.get("original_block")
        if not original_block:
            logger.warning("原文コメントが取得できません path=%s", record.path)
            continue
        indent = _infer_indent(original_block)
        trailing_newline = original_block.endswith("\n")
        body_width = max(10, 88 - len(indent) - 2)
        new_block = _build_comment_block(record.translated, indent, body_width, trailing_newline)
        if original_block not in result:
            logger.warning("コメント置換対象が現在のコードに存在しません path=%s", record.path)
            continue
        result = result.replace(original_block, new_block, 1)
    return result


def _infer_indent(block: str) -> str:
    for line in block.splitlines():
        if "#" in line:
            return line[: line.find("#")]
    return ""


def _build_comment_block(text: str, indent: str, width: int, trailing_newline: bool) -> str:
    lines = _wrap_comment_text(text, indent, width)
    block = "\n".join(lines)
    if trailing_newline:
        block += "\n"
    return block


def _extract_trailing_newline_suffix(value: str) -> str:
    import re

    match = re.search(r"\n[ \t]*\Z", value)
    return match.group(0) if match else ""


def _reindent_docstring(text: str, original_value: str) -> str:
    lines = text.split("\n")
    if len(lines) <= 1:
        return text
    indent = _infer_docstring_indent(original_value)
    if not indent:
        return text
    reindented = [lines[0]]
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            reindented.append(indent.rstrip(" "))
            continue
        if line.startswith(indent):
            reindented.append(line)
        else:
            reindented.append(f"{indent}{stripped}")
    return "\n".join(reindented)


def _infer_docstring_indent(value: str) -> str:
    lines = value.split("\n")
    for line in lines[1:]:
        if line.strip():
            prefix_len = len(line) - len(line.lstrip(" \t"))
            return line[:prefix_len]
    return ""


def _wrap_comment_text(text: str, indent: str, width: int) -> List[str]:
    paragraphs: List[str] = []
    buffer: List[str] = []
    for raw_line in text.split("\n"):
        if raw_line.strip():
            buffer.append(raw_line.strip())
        else:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            paragraphs.append("")
    if buffer:
        paragraphs.append(" ".join(buffer))
    if not paragraphs:
        paragraphs = [""]

    wrapped: List[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            wrapped.append(f"{indent}#")
            continue
        lines = _wrap_paragraph(paragraph, width)
        if not lines:
            wrapped.append(f"{indent}#")
            continue
        wrapped.extend(f"{indent}# {line}" if line else f"{indent}#" for line in lines)
    return wrapped


def _wrap_paragraph(paragraph: str, width: int) -> List[str]:
    if not paragraph:
        return [""]
    wrapper = textwrap.TextWrapper(
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    lines = wrapper.wrap(paragraph)
    if not lines:
        lines = [paragraph]
    result: List[str] = []
    for line in lines:
        if len(line) <= width:
            result.append(line)
            continue
        result.extend(line[i : i + width] for i in range(0, len(line), width))
    return result
