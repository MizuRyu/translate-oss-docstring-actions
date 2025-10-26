"""libcstを用いた抽出ロジック。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import libcst as cst
from libcst import metadata

from logger import logger as root_logger


logger = root_logger.getChild("extraction")


def extract_text_items(source: str, path: Path, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ソースコードから抽出対象テキストを収集する。"""
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as error:  # pragma: no cover - 解析失敗はログのみ
        logger.warning("解析に失敗しました path=%s error=%s", path, error)
        return []
    wrapper = metadata.MetadataWrapper(module)
    positions = wrapper.resolve(metadata.PositionProvider)
    collector = _create_collector(source, positions, path, options)
    wrapper.visit(collector)
    if hasattr(collector, "_flush_comment_block"):
        collector._flush_comment_block()
    return collector.records


def _create_collector(
    source: str,
    positions: metadata.PositionProvider,
    path: Path,
    options: Dict[str, Any],
) -> cst.CSTVisitor:
    """Visitorを生成して抽出結果を保持する。"""
    include_runtime = bool(options.get("include_runtime_messages"))
    include_debug = bool(options.get("include_debug_logs"))

    class TextCollector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (metadata.PositionProvider,)

        def __init__(self) -> None:
            self.scope: List[str] = []
            self.records: List[Dict[str, Any]] = []
            self._comment_block: List[Dict[str, Any]] = []
            self._source_lines = source.splitlines(True)

        def visit_Module(self, node: cst.Module) -> None:
            string_node = _find_docstring_node(node.body)
            if string_node is not None:
                self._append_docstring("module_docstring", "__module__", string_node)

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            self._flush_comment_block()
            self.scope.append(node.name.value)
            string_node = _find_docstring_node(node.body.body)
            if string_node is not None:
                self._append_docstring("class_docstring", node.name.value, string_node)

        def leave_ClassDef(self, node: cst.ClassDef) -> None:
            self._flush_comment_block()
            self.scope.pop()

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            self._flush_comment_block()
            self.scope.append(node.name.value)
            string_node = _find_docstring_node(node.body.body)
            if string_node is not None:
                self._append_docstring("function_docstring", node.name.value, string_node)

        def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
            self._flush_comment_block()
            self.scope.pop()

        def visit_EmptyLine(self, node: cst.EmptyLine) -> None:
            if node.comment is not None:
                self._add_comment(node.comment, "standalone")

        def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
            comment = node.trailing_whitespace.comment
            if comment is not None:
                self._add_comment(comment, "inline")

        def visit_Call(self, node: cst.Call) -> None:
            if include_runtime:
                call_meta = _detect_runtime_call(node, include_debug)
                if call_meta is not None:
                    message = call_meta.pop("message")
                    self._append_record("runtime_message", message, node, call_meta)

        def _append_docstring(self, kind: str, name: str, string_node: cst.SimpleString) -> None:
            self._flush_comment_block()
            text = _evaluate_string(string_node)
            meta = {"target": name}
            self._append_record(kind, text, string_node, meta)

        def _append_record(
            self,
            kind: str,
            text: str,
            node: cst.CSTNode,
            extra_meta: Optional[Dict[str, Any]] = None,
        ) -> None:
            position = positions[node]
            qualname = _build_qualname(self.scope, extra_meta)
            meta = {
                "scope": list(self.scope),
                "position": {
                    "start": {"line": position.start.line, "column": position.start.column},
                    "end": {"line": position.end.line, "column": position.end.column},
                },
            }
            if extra_meta is not None:
                meta.update(extra_meta)
            record = {
                "path": str(path),
                "kind": kind,
                "text": text,
                "meta": meta,
            }
            if qualname is not None:
                record["meta"]["qualname"] = qualname
            self.records.append(record)

        def _add_comment(self, comment: cst.Comment, comment_type: str) -> None:
            text = _clean_comment(comment.value)
            if _should_skip_comment(text):
                self._flush_comment_block()
                return
            position = positions[comment]
            if comment_type == "inline":
                self._flush_comment_block()
                meta = {
                    "scope": list(self.scope),
                    "position": {
                        "start": {"line": position.start.line, "column": position.start.column},
                        "end": {"line": position.end.line, "column": position.end.column},
                    },
                    "comment_type": "inline",
                }
                qualname = _build_qualname(self.scope, None)
                if qualname is not None:
                    meta["qualname"] = qualname
                record = {
                    "path": str(path),
                    "kind": "comment",
                    "text": text,
                    "meta": meta,
                }
                self.records.append(record)
                return
            entry = {
                "text": text,
                "position": position,
                "scope": list(self.scope),
                "comment_type": comment_type,
            }
            if self._comment_block:
                prev = self._comment_block[-1]
                prev_pos = prev["position"]
                if (
                    entry["scope"] == prev["scope"]
                    and position.start.line == prev_pos.end.line + 1
                ):
                    self._comment_block.append(entry)
                else:
                    self._flush_comment_block()
                    self._comment_block = [entry]
            else:
                self._comment_block = [entry]

        def _flush_comment_block(self) -> None:
            if not self._comment_block:
                return
            first = self._comment_block[0]
            last = self._comment_block[-1]
            text = "\n".join(item["text"] for item in self._comment_block)
            position = {
                "start": {
                    "line": first["position"].start.line,
                    "column": first["position"].start.column,
                },
                "end": {
                    "line": last["position"].end.line,
                    "column": last["position"].end.column,
                },
            }
            meta = {
                "scope": first["scope"],
                "position": position,
                "comment_type": "block",
                "line_count": len(self._comment_block),
                "original_block": self._extract_original_block(
                    first["position"],
                    last["position"],
                ),
            }
            qualname = _build_qualname(first["scope"], None)
            if qualname is not None:
                meta["qualname"] = qualname
            record = {
                "path": str(path),
                "kind": "comment",
                "text": text,
                "meta": meta,
            }
            self.records.append(record)
            self._comment_block = []

        def _extract_original_block(
            self,
            start_position: metadata.Position,
            end_position: metadata.Position,
        ) -> str:
            start_line = start_position.start.line - 1
            end_line = end_position.end.line - 1
            if start_line < 0 or end_line >= len(self._source_lines):
                return ""
            return "".join(self._source_lines[start_line : end_line + 1])

    return TextCollector()


def _find_docstring_node(body: Sequence[cst.CSTNode]) -> Optional[cst.SimpleString]:
    """docstringに相当するSimpleStringノードを取得する。"""
    for node in body:
        if isinstance(node, cst.EmptyLine):
            continue
        if isinstance(node, cst.SimpleStatementLine) and len(node.body) > 0:
            expr = node.body[0]
            if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                return expr.value
        return None
    return None


def _evaluate_string(node: cst.SimpleString) -> str:
    """文字列リテラルから実際のテキストを取得する。"""
    try:
        return ast.literal_eval(node.value)
    except Exception:
        return node.value.strip('"\'')


def _clean_comment(source: str) -> str:
    """コメントから先頭記号を取り除く。"""
    text = source.lstrip("#")
    return text.lstrip()


def _should_skip_comment(text: str) -> bool:
    """リンター制御用コメントを除外する。"""
    normalized = text.strip().lower()
    if not normalized:
        return False
    prefixes = (
        "type: ignore",
        "pragma: no cover",
        "noqa",
    )
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _build_qualname(scope: Sequence[str], extra_meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """スコープ情報から疑似的な完全修飾名を生成する。"""
    if len(scope) == 0:
        if extra_meta is None:
            return None
        target = extra_meta.get("target") if isinstance(extra_meta, dict) else None
        if isinstance(target, str) and target != "__module__":
            return target
        return None
    return "::".join(scope)


def _detect_runtime_call(node: cst.Call, include_debug: bool) -> Optional[Dict[str, Any]]:
    """print/Logger呼び出しを判定してメタ情報を返す。"""
    target = node.func
    if isinstance(target, cst.Name) and target.value == "print":
        message = _join_string_arguments(node.args)
        if message is not None:
            return {"category": "print", "message": message}
        return None
    if isinstance(target, cst.Attribute) and isinstance(target.attr, cst.Name):
        level = target.attr.value
        if level == "debug" and not include_debug:
            return None
        if level in {"debug", "info", "warning", "error", "exception", "critical"}:
            message = _first_string_argument(node.args)
            if message is not None:
                return {"category": "logger", "level": level, "message": message}
    return None


def _join_string_arguments(args: Sequence[cst.Arg]) -> Optional[str]:
    """複数引数のprint文から文字列リテラルを結合する。"""
    parts: List[str] = []
    for arg in args:
        value = _extract_string_from_arg(arg)
        if value is None:
            return None
        parts.append(value)
    if len(parts) == 0:
        return None
    return " ".join(parts)


def _first_string_argument(args: Sequence[cst.Arg]) -> Optional[str]:
    """最初の位置引数が文字列リテラルであれば返す。"""
    for arg in args:
        if arg.keyword is None:
            value = _extract_string_from_arg(arg)
            if value is not None:
                return value
            return None
    return None


def _extract_string_from_arg(arg: cst.Arg) -> Optional[str]:
    """Argから文字列リテラルを抽出する。"""
    value = arg.value
    if isinstance(value, cst.SimpleString):
        return _evaluate_string(value)
    return None
