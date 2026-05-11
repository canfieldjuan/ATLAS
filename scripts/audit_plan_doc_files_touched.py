#!/usr/bin/env python3
"""Verify a plan doc's "Files touched" subsection matches the actual diff.

Looks for a "Files touched" sub-heading in the plan doc (under Scope),
extracts backticked file paths from that subsection, and compares to
`git diff --name-only <BASE_REF>...HEAD`.

Reports:
    missing-in-doc -> changed in git, not named in plan doc (scope creep)
    extra-in-doc   -> named in plan doc, not changed in git (mismatch)

Exits 0 if Files touched matches the actual diff. Exits 1 on any drift.

Usage:
    python scripts/audit_plan_doc_files_touched.py PATH [BASE_REF]
        BASE_REF defaults to origin/main.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATH_BACKTICK = re.compile(r"`([A-Za-z0-9_./\-]+\.[a-z][a-z0-9]*)`")


def parse_files_touched(plan_text: str) -> set[str]:
    """Return the set of backticked paths under the Files touched sub-heading."""
    lines = plan_text.splitlines()
    in_section = False
    section: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if (
                stripped.lower().startswith("### files touched")
                or stripped.lower().startswith("**files touched**")
                or stripped.lower().startswith("## files touched")
            ):
                in_section = True
                continue
        else:
            # Section ends at the next "##" or "###" heading.
            if stripped.startswith("## ") or stripped.startswith("### "):
                break
            section.append(line)

    if not in_section:
        return set()

    return set(PATH_BACKTICK.findall("\n".join(section)))


def actual_files_changed(base_ref: str) -> set[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {p for p in out.stdout.splitlines() if p.strip()}


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "usage: audit_plan_doc_files_touched.py PATH [BASE_REF]",
            file=sys.stderr,
        )
        return 2
    path = Path(sys.argv[1])
    base_ref = sys.argv[2] if len(sys.argv) == 3 else "origin/main"

    if not path.exists():
        print(f"plan doc not found: {path}", file=sys.stderr)
        return 2

    claimed = parse_files_touched(path.read_text(encoding="utf-8"))
    if not claimed:
        print(
            f"could not find a 'Files touched' subsection in {path}",
            file=sys.stderr,
        )
        return 2

    try:
        actual = actual_files_changed(base_ref)
    except subprocess.CalledProcessError as exc:
        print(f"git diff failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    missing_in_doc = actual - claimed
    extra_in_doc = claimed - actual

    print(f"plan doc: {path}")
    print(f"base ref: {base_ref}")
    print(f"claimed in doc: {len(claimed)}")
    print(f"actual in diff: {len(actual)}")
    print("-" * 50)

    if missing_in_doc:
        print(f"missing in doc ({len(missing_in_doc)}; possible scope creep):")
        for p in sorted(missing_in_doc):
            print(f"  - {p}")
    if extra_in_doc:
        print(f"extra in doc ({len(extra_in_doc)}; plan/code mismatch):")
        for p in sorted(extra_in_doc):
            print(f"  - {p}")
    if not missing_in_doc and not extra_in_doc:
        print("OK: plan-doc Files touched matches the diff.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
