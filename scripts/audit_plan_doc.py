#!/usr/bin/env python3
"""Verify a plan doc has the 7 AGENTS.md sections, in order.

Required sections per AGENTS.md section 1a (with allowed variants):
    Why this slice exists       ("Why this slice exists" only)
    Scope                       ("Scope" or "Scope (this PR)")
    Mechanism                   ("Mechanism")
    Intentional                 ("Intentional")
    Deferred                    ("Deferred")
    Verification                ("Verification")
    Estimated diff size         ("Estimated diff size")

Scans for "## <title>" headings and checks each required slot is filled
by a heading whose normalized text (lowercased, whitespace-collapsed) is
in the slot's explicit allowlist. Substring matching is intentionally
avoided so that headings like "## Out of scope" cannot pass the "Scope"
requirement.

Exits 0 if all present and ordered. Exits 1 otherwise.

Usage:
    python scripts/audit_plan_doc.py plans/PR-Some-Slice.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Each tuple is (canonical_title, allowlist_of_acceptable_variants).
# Variants are matched on a normalized form (lowercased, whitespace
# collapsed) so reviewers can use small wording tweaks without breaking
# the audit, but unrelated headings like "Out of scope" cannot satisfy
# the "Scope" slot.
REQUIRED: list[tuple[str, tuple[str, ...]]] = [
    ("Why this slice exists", ("why this slice exists",)),
    ("Scope", ("scope", "scope (this pr)")),
    ("Mechanism", ("mechanism",)),
    ("Intentional", ("intentional",)),
    ("Deferred", ("deferred",)),
    ("Verification", ("verification",)),
    ("Estimated diff size", ("estimated diff size",)),
]

_WS = re.compile(r"\s+")


def _normalize(heading: str) -> str:
    return _WS.sub(" ", heading.strip().lower())


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_plan_doc.py PATH", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"plan doc not found: {path}", file=sys.stderr)
        return 2

    headings = []  # list of (line_no, heading_text) for "## ..." lines
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("## "):
            headings.append((i, line[3:].strip()))

    print(f"plan doc: {path}")
    print("-" * 60)

    last_index = -1
    drift = False
    for canonical, variants in REQUIRED:
        match = None
        for idx, (line_no, heading) in enumerate(headings):
            if _normalize(heading) in variants:
                match = (idx, line_no, heading)
                break
        if match is None:
            print(f"MISSING        ## {canonical}")
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
