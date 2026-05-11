#!/usr/bin/env python3
"""Verify a plan doc's Mechanism / Verification claims match shipped code.

The plan doc's Mechanism and Verification sections describe what
the scripts DO; without verification, the descriptions can drift
from the actual implementation. This auditor cross-checks two
shapes of backticked claim against the codebase:

  1. Backticked file paths -- `scripts/foo.py`, `atlas_brain/x.py` --
     must exist on disk under the repo root.

  2. Backticked function-call literals -- `foo_bar()`,
     `parse_estimate()` -- must appear as a "def foo_bar" or
     "async def foo_bar" line in at least one .py file under
     scripts/ or atlas_brain/.

False-positive guard: function-call claims must be at least 4
characters long to avoid noise on built-ins (`get()`, `set()`).

Exits 0 if every claim resolves. Exits 1 on any drift.

Usage:
    python scripts/audit_plan_code_consistency.py PLAN_PATH
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Path claims: a token containing at least one "/" and ending in a
# ".<ext>" suffix. Accepted in backticked form (`scripts/foo.py`) AND
# bare in code blocks (where the path appears literally on a command
# line). Either way the file must exist on disk.
PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_./\-])"
    r"([A-Za-z0-9_][A-Za-z0-9_./\-]*/[A-Za-z0-9_./\-]+\.[a-z][a-z0-9]{0,5})"
    r"(?![A-Za-z0-9_./\-])"
)

# Function-call claims: backticked snake_case identifier (>=4 chars)
# followed by "()". Backtick-only because bare prose mentions of
# function names are too noisy to enforce.
BACKTICK_FUNC = re.compile(r"`([a-z_][a-z0-9_]{3,})\(\)`")


def _slice_sections(plan_text: str, section_titles: tuple[str, ...]) -> str:
    """Return the concatenated body of plan doc sections matching any
    title in `section_titles` (case-insensitive, h2-anchored)."""
    out: list[str] = []
    in_section = False
    for line in plan_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            in_section = any(t.lower() in heading for t in section_titles)
            continue
        if in_section:
            out.append(line)
    return "\n".join(out)


def parse_claims(plan_text: str) -> tuple[set[str], set[str]]:
    # Paths show up in Scope (Files touched), Mechanism, and
    # Verification. Functions claims live in Mechanism / Verification.
    path_body = _slice_sections(plan_text, ("Scope", "Mechanism", "Verification"))
    func_body = _slice_sections(plan_text, ("Mechanism", "Verification"))
    paths = set(PATH_TOKEN.findall(path_body))
    funcs = set(BACKTICK_FUNC.findall(func_body))
    return paths, funcs


def collect_def_names() -> set[str]:
    """Walk scripts/ and atlas_brain/ for every `def name` / `async def name`."""
    names: set[str] = set()
    for root in ("scripts", "atlas_brain"):
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
    return names


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: audit_plan_code_consistency.py PLAN_PATH",
            file=sys.stderr,
        )
        return 2
    plan_path = Path(sys.argv[1])
    if not plan_path.exists():
        print(f"plan doc not found: {plan_path}", file=sys.stderr)
        return 2

    plan_text = plan_path.read_text(encoding="utf-8")
    claimed_paths, claimed_funcs = parse_claims(plan_text)

    print(f"plan doc: {plan_path}")
    print(f"path claims:     {len(claimed_paths)}")
    print(f"function claims: {len(claimed_funcs)}")
    print("-" * 60)

    drift = False

    if claimed_paths:
        missing_paths = sorted(
            p for p in claimed_paths if not (REPO_ROOT / p).exists()
        )
        if missing_paths:
            drift = True
            print(f"MISSING PATHS ({len(missing_paths)}):")
            for p in missing_paths:
                print(f"  - {p}")
        else:
            print(f"OK: all {len(claimed_paths)} path claims exist on disk.")
    else:
        print("OK: no path claims found.")

    if claimed_funcs:
        defs = collect_def_names()
        missing_funcs = sorted(claimed_funcs - defs)
        if missing_funcs:
            drift = True
            print(
                f"MISSING FUNCTION DEFS ({len(missing_funcs)}); "
                f"checked scripts/ and atlas_brain/:"
            )
            for fn in missing_funcs:
                print(f"  - {fn}()")
        else:
            print(f"OK: all {len(claimed_funcs)} function claims resolve to a def.")
    else:
        print("OK: no function-call claims found.")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
