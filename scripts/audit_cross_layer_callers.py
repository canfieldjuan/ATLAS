#!/usr/bin/env python3
"""Surface non-diff references to changed Python symbols.

The audit reads the working tree, so direct dirty-tree runs can include
uncommitted edits. The local review wrapper runs it after the clean-tree gate.
Decorator-only edits and deleted symbols are advisory blind spots.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

CODE_SUFFIXES = {
    ".py",
    ".pyi",
    ".sh",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
}
SYMBOL_MIN_LENGTH = 4


@dataclass(frozen=True)
class ChangedSymbol:
    path: str
    name: str
    qualname: str
    kind: str
    line: int


@dataclass(frozen=True)
class Reference:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class CallerHint:
    symbol: ChangedSymbol
    references: tuple[Reference, ...]


class AuditError(RuntimeError):
    """Raised when the audit cannot safely inspect the repository."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List non-diff references to changed Python functions/classes.",
    )
    parser.add_argument(
        "base_ref",
        nargs="?",
        default="origin/main",
        help="base ref to compare against (default: origin/main)",
    )
    args = parser.parse_args(argv)

    try:
        hints = build_hints(args.base_ref)
    except AuditError as exc:
        print(f"cross-layer caller audit error: {exc}", file=sys.stderr)
        return 2

    render_hints(args.base_ref, hints)
    return 0


def build_hints(base_ref: str) -> tuple[CallerHint, ...]:
    ensure_git_ref(base_ref)
    base = git_stdout(["merge-base", "HEAD", base_ref]).strip()
    changed = changed_files(base)
    modified = modified_files(base)
    changed_python = sorted(path for path in modified if path.endswith(".py"))
    symbols: list[ChangedSymbol] = []
    for path in changed_python:
        validate_repo_path(path)
        changed_lines = changed_line_numbers(base, path)
        if not changed_lines:
            continue
        symbols.extend(changed_symbols(path, changed_lines))

    if not symbols:
        return ()

    searchable = sorted(
        path
        for path in tracked_files()
        if path not in changed and is_code_path(path)
    )
    return tuple(
        CallerHint(symbol=symbol, references=find_references(symbol, searchable))
        for symbol in symbols
    )


def ensure_git_ref(ref: str) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AuditError(f"base ref not found: {ref}")


def changed_files(base: str) -> set[str]:
    output = git_stdout(["diff", "--name-only", f"{base}...HEAD"])
    files = {line for line in output.splitlines() if line.strip()}
    for path in files:
        validate_repo_path(path)
    return files


def modified_files(base: str) -> set[str]:
    output = git_stdout(["diff", "--name-status", f"{base}...HEAD"])
    files: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        status, *paths = line.split("\t")
        if status == "M" and paths:
            files.add(paths[0])
        elif status.startswith("R") and len(paths) == 2:
            files.add(paths[1])
    for path in files:
        validate_repo_path(path)
    return files


def tracked_files() -> set[str]:
    output = git_stdout(["ls-files"])
    files = {line for line in output.splitlines() if line.strip()}
    for path in files:
        validate_repo_path(path)
    return files


def changed_line_numbers(base: str, path: str) -> set[int]:
    output = git_stdout(["diff", "--unified=0", f"{base}...HEAD", "--", path])
    lines: set[int] = set()
    for raw in output.splitlines():
        match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", raw)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count == 0:
            continue
        lines.update(range(start, start + count))
    return lines


def changed_symbols(path: str, changed_lines: set[int]) -> tuple[ChangedSymbol, ...]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise AuditError(f"could not parse changed Python file {path}: {exc}") from exc

    symbols: list[ChangedSymbol] = []
    for qualname, node in iter_symbol_nodes(tree):
        name = node.name
        if len(name) < SYMBOL_MIN_LENGTH:
            continue
        if node.end_lineno is None:
            continue
        span = range(node.lineno, node.end_lineno + 1)
        if changed_lines.intersection(span):
            symbols.append(
                ChangedSymbol(
                    path=path,
                    name=name,
                    qualname=qualname,
                    kind=_symbol_kind(node),
                    line=node.lineno,
                )
            )
    return tuple(symbols)


def iter_symbol_nodes(tree: ast.AST) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]]:
    def visit(
        node: ast.AST,
        parents: tuple[str, ...],
    ) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = ".".join((*parents, child.name))
                yield qualname, child
                yield from visit(child, (*parents, child.name))

    yield from visit(tree, ())


def _symbol_kind(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_function"
    return "function"


def find_references(symbol: ChangedSymbol, paths: Sequence[str]) -> tuple[Reference, ...]:
    references: list[Reference] = []
    patterns = reference_patterns(symbol)
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        candidate_lines = code_reference_lines(path, text, symbol)
        for index, line in enumerate(text.splitlines(), start=1):
            if index in candidate_lines and any(pattern.search(line) for pattern in patterns):
                references.append(
                    Reference(path=path, line=index, text=line.strip()[:160])
                )
    return tuple(references)


def reference_patterns(symbol: ChangedSymbol) -> tuple[re.Pattern[str], ...]:
    escaped = re.escape(symbol.name)
    if symbol.kind == "class":
        return (re.compile(rf"\b{escaped}\b"),)
    if "." in symbol.qualname:
        owner = re.escape(symbol.qualname.rsplit(".", 1)[0].split(".")[-1])
        return (
            re.compile(rf"\.{escaped}\s*\("),
            re.compile(rf"\b{owner}\.{escaped}\s*\("),
        )
    return (re.compile(rf"\b{escaped}\s*\("),)


def code_reference_lines(path: str, text: str, symbol: ChangedSymbol) -> set[int]:
    if PurePosixPath(path).suffix in {".py", ".pyi"}:
        return python_name_lines(text, symbol.name)
    return non_python_candidate_lines(text)


def python_name_lines(text: str, name: str) -> set[int]:
    lines: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.NAME and token.string == name:
                lines.add(token.start[0])
    except tokenize.TokenError:
        return set(range(1, len(text.splitlines()) + 1))
    return lines


def non_python_candidate_lines(text: str) -> set[int]:
    lines: set[int] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith(("//", "#", "/*", "*")):
            lines.add(index)
    return lines


def is_code_path(path: str) -> bool:
    return PurePosixPath(path).suffix in CODE_SUFFIXES


def validate_repo_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or not path.strip():
        raise AuditError(f"unsafe repository path: {path}")


def render_hints(base_ref: str, hints: Sequence[CallerHint]) -> None:
    references = sum(len(hint.references) for hint in hints)
    print("cross-layer caller hints")
    print(f"base ref: {base_ref}")
    print(f"changed python symbols: {len(hints)}")
    print(f"non-diff references: {references}")

    if not hints:
        print("OK: no changed Python symbols found")
        return
    if references == 0:
        print("OK: no non-diff references found")
        return

    print("WARN: changed symbols have references outside this PR diff")
    for hint in hints:
        if not hint.references:
            continue
        symbol = hint.symbol
        print(f"- {symbol.qualname} ({symbol.path}:{symbol.line})")
        for reference in hint.references[:8]:
            print(f"  - {reference.path}:{reference.line}: {reference.text}")
        if len(hint.references) > 8:
            print(f"  - ... {len(hint.references) - 8} more reference(s)")


def git_stdout(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise AuditError(message or f"git {' '.join(args)} failed")
    return result.stdout


if __name__ == "__main__":
    sys.exit(main())
