from __future__ import annotations

import ast
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import libcst as cst
from libcst import metadata

from util import count_tokens, logger


def run(settings: Dict[str, Any]) -> None:
    root = Path(settings["root"]).resolve()
    output_path = Path(settings["output"]).resolve()
    exclude_patterns: Sequence[str] = settings.get("exclude", [])  # type: ignore[assignment]
    include_log = bool(settings.get("include_log_messages"))
    verbose = bool(settings.get("verbose"))

    python_files = _glob_python_files(root, exclude_patterns)
    if not python_files:
        logger.info("解析対象のPythonファイルが見つかりませんでした")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("抽出開始 対象ファイル数:%d", len(python_files))

    start = perf_counter()
    stats = {"files": len(python_files), "files_with_items": 0, "items": 0, "tokens": 0, "chars": 0}

    with output_path.open("w", encoding="utf-8") as handle:
        for path in python_files:
            source = path.read_text(encoding="utf-8")
            entries = _extract_from_file(source, path, include_log, verbose)
            if entries:
                stats["files_with_items"] += 1
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")
                stats["items"] += 1
                stats["tokens"] += count_tokens(entry["text"])
                stats["chars"] += len(entry["text"])

    duration = perf_counter() - start
    summary = {
        "target_files": stats["files"],
        "files_with_extracted_items": stats["files_with_items"],
        "items": stats["items"],
        "total_tokens": stats["tokens"],
        "total_chars": stats["chars"],
        "execution_time_sec": round(duration, 3),
    }
    logger.info("抽出完了 %s", json.dumps(summary, ensure_ascii=False))


def _glob_python_files(root: Path, excludes: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for path in sorted(root.rglob("*.py")):
        if path.is_file() and not _is_excluded(path, root, excludes):
            files.append(path)
    return files


def _is_excluded(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    relative = path.relative_to(root)
    for pattern in patterns:
        if relative.match(pattern) or path.match(pattern):
            return True
    return False


def _extract_from_file(
    source: str,
    path: Path,
    include_log: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as error:  # pragma: no cover
        logger.warning("解析に失敗しました path=%s error=%s", path, error)
        return []

    wrapper = metadata.MetadataWrapper(module)
    positions = wrapper.resolve(metadata.PositionProvider)
    collector = _Collector(source, path, positions, include_log, verbose)
    wrapper.visit(collector)
    collector.flush_block()
    return collector.records


class _Collector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        source: str,
        path: Path,
        positions: metadata.PositionProvider,
        include_log: bool,
        verbose: bool,
    ) -> None:
        self.source_lines = source.splitlines(True)
        self.path = path
        self.positions = positions
        self.include_log = include_log
        self.verbose = verbose
        self.scope: List[str] = []
        self.records: List[Dict[str, Any]] = []
        self.block: List[Dict[str, Any]] = []

    # --- docstrings ---

    def visit_Module(self, node: cst.Module) -> None:
        value = _find_docstring(node.body)
        if value is not None:
            self._record_docstring("module_docstring", "__module__", value)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.flush_block()
        self.scope.append(node.name.value)
        value = _find_docstring(node.body.body)
        if value is not None:
            self._record_docstring("class_docstring", node.name.value, value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.flush_block()
        self.scope.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.flush_block()
        self.scope.append(node.name.value)
        value = _find_docstring(node.body.body)
        if value is not None:
            self._record_docstring("function_docstring", node.name.value, value)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.flush_block()
        self.scope.pop()

    # --- comments ---

    def visit_EmptyLine(self, node: cst.EmptyLine) -> None:
        if node.comment:
            self._collect_comment(node.comment, "block")

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        self.flush_block()
        if node.trailing_whitespace.comment:
            self._record_inline(node.trailing_whitespace.comment)

    def flush_block(self) -> None:
        if not self.block:
            return
        first = self.block[0]
        last = self.block[-1]
        position_start = first["position"].start
        position_end = last["position"].end
        original = self._collect_block_lines()
        text = "\n".join(item["text"] for item in self.block)
        meta = {
            "comment_type": "block",
            "line_count": len(self.block),
            "original_block": original,
        }
        record = self._base_record(
            "comment",
            text,
            meta,
            {
                "start": {"line": position_start.line, "column": position_start.column},
                "end": {"line": position_end.line, "column": position_end.column},
            },
        )
        self.records.append(record)
        self.block = []

    def _collect_block_lines(self) -> str:
        lines: List[str] = []
        previous_line: Optional[int] = None
        for item in self.block:
            current_line = item["line"]
            if previous_line is not None and current_line - previous_line > 1:
                lines.extend(self._capture_intermediate_lines(previous_line, current_line))
            lines.append(self.source_lines[current_line - 1])
            previous_line = current_line
        return "".join(lines)

    def _capture_intermediate_lines(self, start_line: int, end_line: int) -> List[str]:
        collected: List[str] = []
        for index in range(start_line, end_line - 1):
            collected.append(self.source_lines[index])
        return collected

    def _has_interleaved_code(self, start_line: int, end_line: int) -> bool:
        for index in range(start_line, end_line - 1):
            line = self.source_lines[index]
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            return True
        return False

    def _collect_comment(self, comment: cst.Comment, comment_type: str) -> None:
        text = _clean_comment(comment.value)
        if _is_lint_comment(text):
            return
        position = self.positions[comment]
        line_no = position.start.line
        if self.block:
            previous_line = self.block[-1]["line"]
            if line_no - previous_line > 1 and self._has_interleaved_code(previous_line, line_no):
                self.flush_block()
        self.block.append({"position": position, "text": text, "line": line_no})

    def _record_inline(self, comment: cst.Comment) -> None:
        text = _clean_comment(comment.value)
        if _is_lint_comment(text):
            return
        position = self.positions[comment]
        original_line = self.source_lines[position.start.line - 1]
        meta = {
            "comment_type": "inline",
            "original_line": original_line,
        }
        record = self._base_record(
            "comment",
            text,
            meta,
            {
                "start": {"line": position.start.line, "column": position.start.column},
                "end": {"line": position.end.line, "column": position.end.column},
            },
        )
        self.records.append(record)

    # --- runtime ---

    def visit_Call(self, node: cst.Call) -> None:
        if not self.include_log:
            return
        info = _detect_runtime_call(node, self.verbose)
        if info is None:
            return
        message = info.pop("message")
        position = self.positions[node]
        record = self._base_record(
            "runtime_message",
            message,
            info,
            {
                "start": {"line": position.start.line, "column": position.start.column},
                "end": {"line": position.end.line, "column": position.end.column},
            },
        )
        self.records.append(record)

    # --- helpers ---

    def _record_docstring(self, kind: str, name: str, value: cst.SimpleString) -> None:
        text = _literal_value(value)
        position = self.positions[value]
        meta = {"target": name}
        record = self._base_record(
            kind,
            text,
            meta,
            {
                "start": {"line": position.start.line, "column": position.start.column},
                "end": {"line": position.end.line, "column": position.end.column},
            },
        )
        self.records.append(record)

    def _base_record(
        self,
        kind: str,
        text: str,
        extra: Dict[str, Any],
        position: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        meta = {"scope": list(self.scope), "position": position}
        meta.update(extra)
        qualname = _build_qualname(self.scope, extra)
        if qualname:
            meta["qualname"] = qualname
        return {"path": str(self.path), "kind": kind, "text": text, "meta": meta}


def _find_docstring(body: Sequence[cst.CSTNode]) -> Optional[cst.SimpleString]:
    for node in body:
        if isinstance(node, cst.EmptyLine):
            continue
        if isinstance(node, cst.SimpleStatementLine) and node.body:
            expr = node.body[0]
            if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                return expr.value
        return None
    return None


def _literal_value(node: cst.SimpleString) -> str:
    try:
        return ast.literal_eval(node.value)
    except Exception:
        return node.value.strip('"\'')


def _clean_comment(raw: str) -> str:
    return raw.lstrip("#").lstrip()


def _is_lint_comment(text: str) -> bool:
    head = text.strip().lower()
    return head.startswith("type: ignore") or head.startswith("pragma: no cover") or head.startswith("noqa")


def _build_qualname(scope: Sequence[str], extra: Optional[Dict[str, Any]]) -> Optional[str]:
    if not scope:
        if not extra:
            return None
        target = extra.get("target") if isinstance(extra, dict) else None
        if isinstance(target, str) and target != "__module__":
            return target
        return None
    return "::".join(scope)


def _detect_runtime_call(node: cst.Call, verbose: bool) -> Optional[Dict[str, str]]:
    func = node.func
    if isinstance(func, cst.Name) and func.value == "print":
        message = _join_string_arguments(node.args)
        if message is not None:
            return {"category": "print", "message": message}
        return None
    if isinstance(func, cst.Attribute) and isinstance(func.attr, cst.Name):
        level = func.attr.value
        if level == "debug" and not verbose:
            return None
        if level in {"debug", "info", "warning", "error", "exception", "critical"}:
            message = _first_string_argument(node.args)
            if message is not None:
                return {"category": "logger", "level": level, "message": message}
    return None


def _join_string_arguments(args: Sequence[cst.Arg]) -> Optional[str]:
    parts: List[str] = []
    for arg in args:
        value = _extract_string(arg)
        if value is None:
            return None
        parts.append(value)
    return " ".join(parts) if parts else None


def _first_string_argument(args: Sequence[cst.Arg]) -> Optional[str]:
    for arg in args:
        if arg.keyword is None:
            return _extract_string(arg)
    return None


def _extract_string(arg: cst.Arg) -> Optional[str]:
    if isinstance(arg.value, cst.SimpleString):
        return _literal_value(arg.value)
    return None
