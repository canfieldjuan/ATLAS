#!/usr/bin/env python3
"""Verify CLAUDE.md's review-source count matches the ReviewSource enum.

Counts members of `class ReviewSource` in
atlas_brain/services/scraping/sources.py via `ast`. Scans CLAUDE.md
for claims of the form "N review sources" / "N review sites" / etc.
Reports each claim line with expected vs actual; exits 1 on any drift.

Usage:
    python scripts/audit_review_source_count.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_PY = REPO_ROOT / "atlas_brain" / "services" / "scraping" / "sources.py"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

CLAIM_PATTERN = re.compile(
    r"(\d+)\s+review\s+(?:source[s]?|site[s]?)",
    re.IGNORECASE,
)


def count_review_sources() -> int:
    if not SOURCES_PY.exists():
        return -1
    tree = ast.parse(SOURCES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ReviewSource":
            return sum(1 for n in node.body if isinstance(n, ast.Assign))
    return -1


def main() -> int:
    actual = count_review_sources()
    if actual < 0:
        print(
            "Could not locate `class ReviewSource` in "
            f"{SOURCES_PY.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        return 2

    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return 2

    text = CLAUDE_MD.read_text(encoding="utf-8")
    print(f"ReviewSource enum members: {actual}")
    print(f"Claims in {CLAUDE_MD.name}:")
    print("-" * 50)

    drift = False
    any_claim = False
    for i, line in enumerate(text.splitlines(), start=1):
        for m in CLAIM_PATTERN.finditer(line):
            any_claim = True
            claimed = int(m.group(1))
            status = "OK" if claimed == actual else "DRIFT"
            if status == "DRIFT":
                drift = True
            print(f"  line {i:>4}: claims {claimed:>2}  -> {status}")
            print(f"           : {line.strip()}")

    if not any_claim:
        print("  (no review-source claims found)")
        # No claim is not drift per se; treat as OK.
        return 0

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
