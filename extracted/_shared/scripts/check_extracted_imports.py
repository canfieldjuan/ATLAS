#!/usr/bin/env python3
"""Audit relative imports inside an extracted product scaffold."""

from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def manifest_python_targets(product: Path) -> list[Path]:
    obj = json.loads((product / "manifest.json").read_text())
    targets = [
        ROOT / mapping["target"]
        for mapping in obj["mappings"]
        if mapping["target"].endswith(".py")
    ]
    targets.extend(
        ROOT / entry["target"]
        for entry in obj.get("owned", [])
        if entry["target"].endswith(".py")
    )
    return targets


def resolve_relative(
    module_path: Path,
    product: Path,
    level: int,
    module: str | None,
    allow_atlas_fallback: bool,
) -> list[Path]:
    base_parts = list(module_path.relative_to(ROOT).parts)
    package_parts = base_parts[:-1]
    ascend = max(0, level - 1)
    target_parts = package_parts[: max(0, len(package_parts) - ascend)]
    if module:
        target_parts.extend(module.split("."))

    rel = Path(*target_parts) if target_parts else Path()
    candidates = [
        (ROOT / rel).with_suffix(".py"),
        ROOT / rel / "__init__.py",
    ]
    if allow_atlas_fallback and target_parts and target_parts[0] == product.name:
        atlas_rel = Path("atlas_brain", *target_parts[1:])
        candidates.extend([
            (ROOT / atlas_rel).with_suffix(".py"),
            ROOT / atlas_rel / "__init__.py",
        ])
    return candidates


def load_allowlist(product: Path) -> set[str]:
    path = product / "import_debt_allowlist.txt"
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def check_file(
    path: Path,
    product: Path,
    allowlist: set[str],
    allow_atlas_fallback: bool,
) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            candidates = resolve_relative(
                path,
                product,
                node.level,
                node.module,
                allow_atlas_fallback,
            )
            if not any(candidate.exists() for candidate in candidates):
                mod = f"{'.' * node.level}{node.module or ''}"
                if mod not in allowlist:
                    errors.append(f"{path}: unresolved relative import '{mod}'")
    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("product_dir")
    parser.add_argument(
        "--no-atlas-fallback",
        action="store_true",
        help="Require relative imports to resolve inside the extracted product only.",
    )
    args = parser.parse_args(argv[1:])

    product = Path(args.product_dir)
    manifest = product / "manifest.json"
    if not manifest.exists():
        print(f"ERROR missing manifest: {manifest}", file=sys.stderr)
        return 2

    py_files = manifest_python_targets(product)
    allowlist = load_allowlist(product)
    errors: list[str] = []
    for file_path in py_files:
        errors.extend(
            check_file(
                file_path,
                product,
                allowlist,
                allow_atlas_fallback=not args.no_atlas_fallback,
            )
        )

    if errors:
        print("Import check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Import check passed for {len(py_files)} {product} module(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
