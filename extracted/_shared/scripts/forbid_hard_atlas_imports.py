#!/usr/bin/env python3
"""Fail closed when an extracted product file imports atlas_brain unconditionally.

The extracted packages reach atlas_brain only through lazy-bridge stubs.
Two legitimate forms are in use across the scaffolds:

  1. try / except ImportError fallback:
       try:
           from atlas_brain.X import Y
       except ImportError:
           from ._standalone.X import Y

  2. if / else env-gated branch (e.g. EXTRACTED_LLM_INFRA_STANDALONE):
       if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
           from ._standalone.X import Y
       else:
           from atlas_brain.X import Y

Either form ensures atlas_brain is reached only through a gated bridge.
A hard top-level `from atlas_brain` / `import atlas_brain` -- not nested
inside an If, Try, FunctionDef, or AsyncFunctionDef -- breaks the
extraction contract by forcing atlas_brain on sys.path even in standalone
mode.

This script walks every .py file under a product directory, parses it with
ast, and flags atlas_brain imports whose nearest enclosing scope is the
module body or a class body.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


_ALLOWED_ANCESTOR_TYPES = (
    ast.If,
    ast.Try,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def _module_targets_atlas_brain(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module == "atlas_brain" or module.startswith("atlas_brain.")
    if isinstance(node, ast.Import):
        return any(
            alias.name == "atlas_brain" or alias.name.startswith("atlas_brain.")
            for alias in node.names
        )
    return False


def _has_allowed_ancestor(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    cursor = parents.get(id(node))
    while cursor is not None:
        if isinstance(cursor, _ALLOWED_ANCESTOR_TYPES):
            return True
        cursor = parents.get(id(cursor))
    return False


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def scan_file(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"syntax error: {exc.msg}")]

    parents = _build_parent_map(tree)
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not _module_targets_atlas_brain(node):
            continue
        if _has_allowed_ancestor(node, parents):
            continue
        if isinstance(node, ast.ImportFrom):
            target = f"from {node.module} import ..."
        else:
            target = "import " + ", ".join(alias.name for alias in node.names)
        violations.append((node.lineno, target))
    return violations


def scan_package(package_dir: Path) -> dict[Path, list[tuple[int, str]]]:
    findings: dict[Path, list[tuple[int, str]]] = {}
    for path in sorted(package_dir.rglob("*.py")):
        violations = scan_file(path)
        if violations:
            findings[path] = violations
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed on hard top-level atlas_brain imports in an extracted product."
    )
    parser.add_argument(
        "package",
        type=Path,
        help="Path to the extracted package directory (e.g. extracted_competitive_intelligence).",
    )
    args = parser.parse_args()

    package = args.package.resolve()
    if not package.is_dir():
        print(f"ERROR: not a directory: {package}", file=sys.stderr)
        return 2

    findings = scan_package(package)
    if not findings:
        print(f"forbid_hard_atlas_imports: clean ({package.name})")
        return 0

    print(f"forbid_hard_atlas_imports: violations in {package.name}", file=sys.stderr)
    for path, violations in findings.items():
        rel = path.relative_to(package.parent) if package.parent in path.parents else path
        for lineno, target in violations:
            print(f"  {rel}:{lineno}: {target}", file=sys.stderr)
    print(
        "\nHint: wrap the import in a try/except ImportError block (the lazy-bridge "
        "pattern), defer it to inside a function, or remove it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
