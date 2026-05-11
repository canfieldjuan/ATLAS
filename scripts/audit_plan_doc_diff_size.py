#!/usr/bin/env python3
"""Verify a plan doc's "Estimated diff size" matches the actual diff size.

Parses the plan doc for a "Total" row containing a "~N" LOC estimate
(typical shape: "| **Total** | **~310** |"), runs
`git diff --shortstat <BASE_REF>...HEAD` for the actual additions plus
deletions, and compares.

Threshold:
    drift_pct <= 25%   -> OK
    25% < pct <= 50%   -> WARN (exit 0, print warning)
    pct > 50%          -> FAIL (exit 1)

Usage:
    python scripts/audit_plan_doc_diff_size.py PATH [BASE_REF]
        BASE_REF defaults to origin/main.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TOTAL_ROW = re.compile(
    r"\|\s*\*?\*?Total\*?\*?\s*\|\s*\*?\*?~(\d+)\*?\*?\s*\|",
    re.IGNORECASE,
)
SHORTSTAT = re.compile(
    r"(\d+)\s+insertion[s]?\(\+\)|(\d+)\s+deletion[s]?\(-\)"
)


def parse_estimate(plan_text: str) -> int | None:
    """Find the Total LOC inside the '## Estimated diff size' section.

    Scoped to that section so example tables elsewhere (Mechanism notes,
    illustrative snippets) do not get picked up as the canonical Total.
    """
    lines = plan_text.splitlines()
    in_section = False
    section: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if stripped.startswith("## ") and "estimated diff size" in stripped.lower():
                in_section = True
                continue
        else:
            if stripped.startswith("## "):
                break
            section.append(line)
    if not in_section:
        return None
    m = TOTAL_ROW.search("\n".join(section))
    return int(m.group(1)) if m else None


def actual_diff_size(base_ref: str) -> int:
    out = subprocess.run(
        ["git", "diff", "--shortstat", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    total = 0
    for m in SHORTSTAT.finditer(out.stdout):
        total += int(m.group(1) or m.group(2))
    return total


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "usage: audit_plan_doc_diff_size.py PATH [BASE_REF]",
            file=sys.stderr,
        )
        return 2
    path = Path(sys.argv[1])
    base_ref = sys.argv[2] if len(sys.argv) == 3 else "origin/main"

    if not path.exists():
        print(f"plan doc not found: {path}", file=sys.stderr)
        return 2

    estimate = parse_estimate(path.read_text(encoding="utf-8"))
    if estimate is None:
        print(
            f"could not find a '**Total** | **~N**' row in {path}",
            file=sys.stderr,
        )
        return 2

    try:
        actual = actual_diff_size(base_ref)
    except subprocess.CalledProcessError as exc:
        print(f"git diff failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    drift = abs(actual - estimate) / estimate if estimate else 1.0
    pct = drift * 100

    print(f"plan doc: {path}")
    print(f"base ref: {base_ref}")
    print(f"estimate: ~{estimate} LOC")
    print(f"actual:    {actual} LOC")
    print(f"drift:     {pct:.1f}%")
    print("-" * 50)

    if drift <= 0.25:
        print("OK: actual within 25% of estimate.")
        return 0
    if drift <= 0.50:
        print("WARN: actual within 25-50% of estimate.")
        print("      Plan doc estimate is loose; consider tightening.")
        return 0
    print("FAIL: actual exceeds estimate by more than 50%.")
    print("      Plan doc no longer reflects the diff; update Why or estimate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
