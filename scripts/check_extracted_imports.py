#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "extracted_content_pipeline" / "autonomous" / "tasks"

# Relative-import fallback roots for copied modules that still reference atlas code.
def resolve_relative(module_path: Path, level: int, module: str | None) -> list[Path]:
    base_parts = list(module_path.relative_to(ROOT).parts)
    package_parts = base_parts[:-1]
    target_parts = package_parts[: max(0, len(package_parts) - level)]
    if module:
        target_parts.extend(module.split("."))

    rel = Path(*target_parts) if target_parts else Path()
    candidates: list[Path] = []

    # Candidate 1: exact relative path from repository root.
    candidates.append((ROOT / rel).with_suffix(".py"))
    candidates.append(ROOT / rel / "__init__.py")

    # Candidate 2: atlas fallback by swapping extracted root for atlas_brain.
    if target_parts and target_parts[0] == "extracted_content_pipeline":
        atlas_rel = Path("atlas_brain", *target_parts[1:])
        candidates.append((ROOT / atlas_rel).with_suffix(".py"))
        candidates.append(ROOT / atlas_rel / "__init__.py")

    return candidates


def load_allowlist() -> set[str]:
    path = ROOT / "extracted_content_pipeline" / "import_debt_allowlist.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")}


def check_file(path: Path, allowlist: set[str]) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            candidates = resolve_relative(path, node.level, node.module)
            if not any(c.exists() for c in candidates):
                mod = f"{'.' * node.level}{node.module or ''}"
                if mod.startswith("..."):
                    continue
                if mod not in allowlist:
                    errors.append(f"{path}: unresolved relative import '{mod}'")
    return errors


def main() -> int:
    tracked = [
        "blog_post_generation.py",
        "b2b_blog_post_generation.py",
        "complaint_content_generation.py",
        "complaint_enrichment.py",
        "article_enrichment.py",
    ]
    py_files = [TASK_DIR / name for name in tracked]
    allowlist = load_allowlist()
    errors: list[str] = []
    for f in py_files:
        errors.extend(check_file(f, allowlist))

    if errors:
        print("Import check failed:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Import check passed for extracted task modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
