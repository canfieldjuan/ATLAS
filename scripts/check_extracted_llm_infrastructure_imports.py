#!/usr/bin/env python3
"""Audit relative imports inside the extracted_llm_infrastructure scaffold.

For each Python file in the manifest, walk its AST and verify that every
``from .x import y`` style import resolves to a real path -- either inside
the scaffold itself, or, during the Phase 1 transition, inside
``atlas_brain`` via the same parent path. Imports that cannot be resolved
must be listed in ``extracted_llm_infrastructure/import_debt_allowlist.txt``
or this script fails.

This is the Phase 1 trip-wire: it documents (in the allowlist) every
atlas_brain import the scaffold still depends on. Phase 2 work shrinks
the allowlist.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "extracted_llm_infrastructure"


def manifest_python_targets() -> list[Path]:
    obj = json.loads((SCAFFOLD / "manifest.json").read_text())
    return [
        ROOT / mapping["target"]
        for mapping in obj["mappings"]
        if mapping["target"].endswith(".py")
    ]


def resolve_relative(module_path: Path, level: int, module: str | None) -> list[Path]:
    """Return candidate filesystem locations for a relative import.

    Honors Python's actual relative-import semantics: ``level=1`` means
    "current package" (sibling of the importing file), ``level=2`` means
    "parent package", ``level=3`` means "grandparent", etc. The ascend
    is therefore ``level - 1`` package components, not ``level``.

    Tries inside the scaffold first, then falls back to the atlas_brain
    root. The scaffold and atlas_brain mirror the same package
    hierarchy under their root, so a `from ..config import settings` in
    a copied file resolves to ``atlas_brain.config`` only when the
    scaffold lacks the equivalent path.
    """
    base_parts = list(module_path.relative_to(ROOT).parts)
    package_parts = base_parts[:-1]
    ascend = max(0, level - 1)
    target_parts = package_parts[: max(0, len(package_parts) - ascend)]
    if module:
        target_parts.extend(module.split("."))

    rel = Path(*target_parts) if target_parts else Path()
    candidates: list[Path] = []

    candidates.append((ROOT / rel).with_suffix(".py"))
    candidates.append(ROOT / rel / "__init__.py")

    if target_parts and target_parts[0] == "extracted_llm_infrastructure":
        atlas_rel = Path("atlas_brain", *target_parts[1:])
        candidates.append((ROOT / atlas_rel).with_suffix(".py"))
        candidates.append(ROOT / atlas_rel / "__init__.py")

    return candidates


def load_allowlist() -> set[str]:
    path = SCAFFOLD / "import_debt_allowlist.txt"
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def check_file(path: Path, allowlist: set[str]) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            candidates = resolve_relative(path, node.level, node.module)
            if not any(c.exists() for c in candidates):
                mod = f"{'.' * node.level}{node.module or ''}"
                if mod not in allowlist:
                    errors.append(f"{path}: unresolved relative import '{mod}'")
    return errors


def main() -> int:
    py_files = manifest_python_targets()
    allowlist = load_allowlist()
    errors: list[str] = []
    for f in py_files:
        errors.extend(check_file(f, allowlist))

    if errors:
        print("Import check failed:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(
        f"Import check passed for {len(py_files)} extracted_llm_infrastructure module(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
