#!/usr/bin/env python3
"""Verify a plan doc has the required AGENTS.md sections, in order."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class SectionAuditRow:
    canonical: str
    status: str
    line_no: int | None = None
    heading: str = ""


def _normalize(heading: str) -> str:
    return _WS.sub(" ", heading.strip().lower())


def plan_headings(text: str) -> list[tuple[int, str]]:
    return [
        (line_no, line[3:].strip())
        for line_no, line in enumerate(text.splitlines(), start=1)
        if line.startswith("## ")
    ]


def audit_plan_text(text: str) -> list[SectionAuditRow]:
    headings = plan_headings(text)
    last_index = -1
    rows: list[SectionAuditRow] = []

    for canonical, variants in REQUIRED:
        matches = [
            (idx, line_no, heading)
            for idx, (line_no, heading) in enumerate(headings)
            if _normalize(heading) in variants
        ]
        if not matches:
            rows.append(SectionAuditRow(canonical=canonical, status="MISSING"))
            continue

        idx, line_no, heading = matches[0]
        if len(matches) > 1:
            rows.append(
                SectionAuditRow(
                    canonical=canonical,
                    status="DUPLICATE",
                    line_no=line_no,
                    heading=heading,
                )
            )
        elif idx <= last_index:
            rows.append(
                SectionAuditRow(
                    canonical=canonical,
                    status="OUT OF ORDER",
                    line_no=line_no,
                    heading=heading,
                )
            )
        else:
            rows.append(
                SectionAuditRow(
                    canonical=canonical,
                    status="OK",
                    line_no=line_no,
                    heading=heading,
                )
            )
            last_index = idx
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_plan_doc.py PATH", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"plan doc not found: {path}", file=sys.stderr)
        return 2

    print(f"plan doc: {path}")
    print("-" * 60)

    drift = False
    for row in audit_plan_text(path.read_text(encoding="utf-8")):
        if row.status != "OK":
            drift = True
        if row.line_no is None:
            print(f"{row.status:<14} ## {row.canonical}")
        else:
            print(f"{row.status:<14} line {row.line_no:>4}: ## {row.heading}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
