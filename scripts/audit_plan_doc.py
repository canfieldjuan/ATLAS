#!/usr/bin/env python3
"""Verify a plan doc has the 7 AGENTS.md sections, in order.

Required sections per AGENTS.md section 1a:
    Why this slice exists
    Scope                 (matches "Scope" or "Scope (this PR)")
    Mechanism
    Intentional
    Deferred
    Verification
    Estimated diff size

Scans for "## <title>" headings (case-insensitive substring match) and
checks each required title appears at least once, in the order above.

Exits 0 if all present and ordered. Exits 1 otherwise.

Usage:
    python scripts/audit_plan_doc.py plans/PR-Some-Slice.md
"""
from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = [
    "Why this slice exists",
    "Scope",
    "Mechanism",
    "Intentional",
    "Deferred",
    "Verification",
    "Estimated diff size",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_plan_doc.py PATH", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"plan doc not found: {path}", file=sys.stderr)
        return 2

    headings = []  # list of (line_no, heading_text) for "## ..." lines
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if line.startswith("## "):
            headings.append((i, line[3:].strip()))

    print(f"plan doc: {path}")
    print("-" * 60)

    last_index = -1
    drift = False
    for required in REQUIRED:
        match = None
        for idx, (line_no, heading) in enumerate(headings):
            if required.lower() in heading.lower():
                match = (idx, line_no, heading)
                break
        if match is None:
            print(f"MISSING        ## {required}")
            drift = True
            continue
        idx, line_no, heading = match
        if idx <= last_index:
            print(f"OUT OF ORDER   line {line_no}: ## {heading}")
            drift = True
        else:
            print(f"OK  line {line_no:>4}  ## {heading}")
            last_index = idx

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
