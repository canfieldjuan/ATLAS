#!/usr/bin/env python3
"""Verify a plan doc has the required AGENTS.md sections, in order."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from itertools import islice
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
PROBLEM_CONTRACT_HEADING = "Problem-derived contract"
REVIEW_CONTRACT_HEADING = "Review Contract"

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


def _find_problem_contract(text: str, why_line_no: int) -> tuple[int, str] | None:
    lines = text.splitlines()
    for line_no, line in enumerate(lines[why_line_no:], start=why_line_no + 1):
        if line.startswith("## "):
            return None
        if line.startswith("### ") and _normalize(line[4:]) == _normalize(
            PROBLEM_CONTRACT_HEADING
        ):
            return line_no, line[4:].strip()
    return None


def _nested_heading_matches(text: str, heading: str) -> list[tuple[int, str]]:
    return [
        (line_no, line[4:].strip())
        for line_no, line in enumerate(text.splitlines(), start=1)
        if line.startswith("### ") and _normalize(line[4:]) == _normalize(heading)
    ]


def _scope_contains_line(headings: list[tuple[int, str]], line_no: int) -> bool:
    scope_lines = [
        heading_line
        for heading_line, heading in headings
        if _normalize(heading) in {"scope", "scope (this pr)"}
    ]
    if len(scope_lines) != 1:
        return False
    scope_line = next(iter(scope_lines))
    next_top_level = next(
        (heading_line for heading_line, _ in headings if heading_line > scope_line),
        None,
    )
    return line_no > scope_line and (next_top_level is None or line_no < next_top_level)


def _nested_heading_has_content(text: str, line_no: int) -> bool:
    lines = text.splitlines()
    for line in islice(lines, line_no, None):
        if line.startswith("## ") or line.startswith("### "):
            return False
        if line.strip():
            return True
    return False


def _review_contract_row(text: str, headings: list[tuple[int, str]]) -> SectionAuditRow:
    matches = _nested_heading_matches(text, REVIEW_CONTRACT_HEADING)
    if not matches:
        return SectionAuditRow(canonical=REVIEW_CONTRACT_HEADING, status="MISSING")
    line_no, heading = next(iter(matches))
    if len(matches) > 1:
        return SectionAuditRow(
            canonical=REVIEW_CONTRACT_HEADING,
            status="DUPLICATE",
            line_no=line_no,
            heading=heading,
        )
    if not _scope_contains_line(headings, line_no):
        return SectionAuditRow(
            canonical=REVIEW_CONTRACT_HEADING,
            status="OUT OF SCOPE",
            line_no=line_no,
            heading=heading,
        )
    if not _nested_heading_has_content(text, line_no):
        return SectionAuditRow(
            canonical=REVIEW_CONTRACT_HEADING,
            status="EMPTY",
            line_no=line_no,
            heading=heading,
        )
    return SectionAuditRow(
        canonical=REVIEW_CONTRACT_HEADING,
        status="OK",
        line_no=line_no,
        heading=heading,
    )


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

        idx, line_no, heading = next(iter(matches))
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
            if canonical == "Why this slice exists":
                problem_contract = _find_problem_contract(text, line_no)
                if problem_contract is None:
                    rows.append(
                        SectionAuditRow(
                            canonical=PROBLEM_CONTRACT_HEADING,
                            status="MISSING",
                        )
                    )
                else:
                    contract_line_no, contract_heading = problem_contract
                    rows.append(
                        SectionAuditRow(
                            canonical=PROBLEM_CONTRACT_HEADING,
                            status="OK",
                            line_no=contract_line_no,
                            heading=contract_heading,
                        )
                    )
    rows.append(_review_contract_row(text, headings))
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
            marker = (
                "###"
                if row.canonical in {PROBLEM_CONTRACT_HEADING, REVIEW_CONTRACT_HEADING}
                else "##"
            )
            print(f"{row.status:<14} line {row.line_no:>4}: {marker} {row.heading}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
