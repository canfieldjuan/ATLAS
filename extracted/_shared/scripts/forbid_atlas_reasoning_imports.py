#!/usr/bin/env python3
"""Fail closed when an extracted product imports ``atlas_brain.reasoning``.

The audit's PR 7 acceptance criterion: "no runtime
``atlas_brain.reasoning`` imports in extracted products". Post-PR-C4
the reasoning core (``extracted_reasoning_core``) provides every
reasoning-side surface the products need (helpers, ports, adapters,
orchestrator). Any leftover atlas-reasoning imports indicate either
dead code (e.g. the ``__getattr__`` bridge PR-D7a removes) or a
contract drift that breaks standalone-product builds.

This guard is *stricter* than ``forbid_hard_atlas_imports.py``: it
fails on **all** atlas-reasoning import forms, including those wrapped
in try/except / env-gated bridges. Sibling-script
``forbid_hard_atlas_imports.py`` allows gated escapes for non-reasoning
atlas modules; here even gated forms are out of contract because the
target (atlas_brain.reasoning) has a ready-to-use core replacement.

Three import forms covered:

1. ``from atlas_brain.reasoning[...] import X`` -- AST ``ImportFrom``.
2. ``import atlas_brain.reasoning[...]`` -- AST ``Import``.
3. ``importlib.import_module("atlas_brain.reasoning[...]")`` --
   AST ``Call`` whose first arg is a string literal starting with
   the forbidden prefix. Catches the lazy-bridge pattern
   PR-D7a removed from ``extracted_competitive_intelligence/reasoning/__init__.py``.

The check ignores documentation references in docstrings and comments
-- only real ``import`` / ``import_module`` syntax fires. Run as:

    python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py PRODUCT_DIR
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


_FORBIDDEN_PREFIX = "atlas_brain.reasoning"


def _module_targets_atlas_reasoning(module: str | None) -> bool:
    if not module:
        return False
    return module == _FORBIDDEN_PREFIX or module.startswith(_FORBIDDEN_PREFIX + ".")


def _import_node_targets_atlas_reasoning(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return _module_targets_atlas_reasoning(node.module)
    if isinstance(node, ast.Import):
        return any(
            _module_targets_atlas_reasoning(alias.name) for alias in node.names
        )
    return False


def _is_import_module_call(node: ast.Call) -> bool:
    """Return True for ``importlib.import_module(...)`` callable shape.

    Matches both ``importlib.import_module("...")`` (Attribute) and
    ``import_module("...")`` (Name -- when the function was imported
    as ``from importlib import import_module``).
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "import_module"
    if isinstance(func, ast.Name):
        return func.id == "import_module"
    return False


def _import_module_call_targets_atlas_reasoning(
    node: ast.Call,
) -> str | None:
    """Return the forbidden module name if this call targets atlas_brain.reasoning.

    Inspects both positional and keyword forms -- ``importlib.import_module``
    accepts ``name`` as a keyword (and ``package`` for relative imports), so
    a call like ``importlib.import_module(name="atlas_brain.reasoning")``
    must trip the guard the same as the positional form. Returns the
    string the call would resolve, or ``None`` if it doesn't statically
    target the forbidden prefix.
    """
    if not _is_import_module_call(node):
        return None
    candidate: ast.expr | None = None
    if node.args:
        candidate = node.args[0]
    if candidate is None:
        for kw in node.keywords:
            if kw.arg == "name":
                candidate = kw.value
                break
    if candidate is None:
        return None
    if not isinstance(candidate, ast.Constant):
        return None
    if not isinstance(candidate.value, str):
        return None
    if not _module_targets_atlas_reasoning(candidate.value):
        return None
    return candidate.value


def scan_file(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"syntax error: {exc.msg}")]

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _import_node_targets_atlas_reasoning(node):
            violations.append(
                (node.lineno, f"from {node.module} import ...")
            )
            continue
        if isinstance(node, ast.Import) and _import_node_targets_atlas_reasoning(node):
            target = "import " + ", ".join(
                alias.name
                for alias in node.names
                if _module_targets_atlas_reasoning(alias.name)
            )
            violations.append((node.lineno, target))
            continue
        if isinstance(node, ast.Call):
            forbidden_name = _import_module_call_targets_atlas_reasoning(node)
            if forbidden_name is not None:
                violations.append(
                    (node.lineno, f"importlib.import_module({forbidden_name!r})")
                )
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
        description=(
            "Fail closed on any atlas_brain.reasoning import "
            "(direct, wrapped, or via importlib) in an extracted product."
        )
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
        print(f"forbid_atlas_reasoning_imports: clean ({package.name})")
        return 0

    print(
        f"forbid_atlas_reasoning_imports: violations in {package.name}",
        file=sys.stderr,
    )
    for path, violations in findings.items():
        rel = path.relative_to(package.parent) if package.parent in path.parents else path
        for lineno, target in violations:
            print(f"  {rel}:{lineno}: {target}", file=sys.stderr)
    print(
        "\nHint: import the equivalent symbol from extracted_reasoning_core "
        "(api / types / ports / concrete module) instead of atlas_brain.reasoning.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
