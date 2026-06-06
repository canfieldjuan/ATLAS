#!/usr/bin/env python3
"""Summarize the reviewer-side miss ledger (REVIEW_MISSES.md).

Reads the Ledger table and reports reviewer-quality metrics: how many escaped
defects were logged, how they break down by who missed them (human / AI / CI),
how many have been converted into a durable gate, and -- the signal the
operating-model gap (b) cares about most -- how many were AI findings a human
reviewer missed.

This is a report tool, not a blocking gate, so it exits 0 by default. Pass
--fail-on-ungated to enforce the ledger rule "no escaped defect is fixed only
once" (every real row must name a gate) in CI once the ledger has real entries.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = "REVIEW_MISSES.md"

LEDGER_HEADING_RE = re.compile(r"^\s*#+\s*Ledger\b", re.IGNORECASE)
NEXT_HEADING_RE = re.compile(r"^\s*#+\s+\S")
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")

# A cell that carries no real content: empty, a dash, or an italic placeholder.
EMPTY_CELL_RE = re.compile(r"^\s*(|-+|_.*_)\s*$")
# Human-written "no value here" markers, treated as empty (e.g. an ungated row).
NO_CONTENT_MARKERS = {"", "-", "--", "n/a", "na", "none", "none yet", "pending", "tbd", "todo", "tba"}


def _ledger_rows(text: str) -> list[list[str]]:
    """Return the data-row cell lists of the Ledger table."""
    lines = text.splitlines()
    in_section = False
    rows: list[list[str]] = []
    for line in lines:
        if LEDGER_HEADING_RE.match(line):
            in_section = True
            continue
        if in_section and NEXT_HEADING_RE.match(line):
            break
        if in_section:
            match = TABLE_ROW_RE.match(line)
            if match:
                rows.append([c.strip() for c in match.group(1).split("|")])
    return rows


def _is_empty(cell: str) -> bool:
    if EMPTY_CELL_RE.match(cell) or not cell.strip():
        return True
    return cell.strip().strip("()").strip().lower() in NO_CONTENT_MARKERS


def _is_placeholder_row(cells: Sequence[str]) -> bool:
    """Header, separator, or the seed/example row carry no real data."""
    joined = " ".join(cells).lower()
    if not cells or set(" ".join(cells)) <= set("-: |"):
        return True
    if cells[0].lower().startswith("date"):  # header
        return True
    if "_seed_" in joined or "first real entry" in joined:
        return True
    # A row whose first two cells are both empty/placeholder is not a real miss.
    return _is_empty(cells[0]) and (len(cells) < 2 or _is_empty(cells[1]))


def _ledger_table_present(text: str) -> bool:
    """True when a Ledger section with at least one table row exists.

    Distinguishes a valid (possibly seed-only) ledger from a missing/renamed
    Ledger heading or the wrong file -- the latter must fail closed rather than
    masquerade as "no misses logged".
    """
    return len(_ledger_rows(text)) >= 1


def parse_ledger(text: str) -> list[dict[str, str]]:
    """Return real miss rows as dicts (date, issue, missed_by, root_cause, gate, owner)."""
    fields = ["date", "issue", "missed_by", "root_cause", "gate", "owner"]
    out: list[dict[str, str]] = []
    for cells in _ledger_rows(text):
        if _is_placeholder_row(cells):
            continue
        row = {fields[i]: (cells[i] if i < len(cells) else "") for i in range(len(fields))}
        out.append(row)
    return out


def _buckets(missed_by: str) -> set[str]:
    low = missed_by.lower()
    found = set()
    if re.search(r"\bhuman\b", low):
        found.add("human")
    if re.search(r"\bai\b|codex|copilot", low):
        found.add("ai")
    if re.search(r"\bci\b", low):
        found.add("ci")
    return found


def summarize(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    summary = {
        "total": len(rows),
        "by_human": 0,
        "by_ai": 0,
        "by_ci": 0,
        "gated": 0,
        "ungated": 0,
        "ai_missed_by_human": 0,
    }
    for row in rows:
        buckets = _buckets(row.get("missed_by", ""))
        if "human" in buckets:
            summary["by_human"] += 1
        if "ai" in buckets:
            summary["by_ai"] += 1
        if "ci" in buckets:
            summary["by_ci"] += 1
        if _is_empty(row.get("gate", "")):
            summary["ungated"] += 1
        else:
            summary["gated"] += 1
        # An AI finding a human reviewer missed: the row attributes the miss to a
        # human, and the issue/root-cause names an AI/bot finding.
        ctx = (row.get("issue", "") + " " + row.get("root_cause", "")).lower()
        if "human" in buckets and re.search(r"\bai\b|codex|copilot|bot\b", ctx):
            summary["ai_missed_by_human"] += 1
    return summary


def ungated_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if _is_empty(r.get("gate", ""))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the reviewer miss ledger.")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument(
        "--fail-on-ungated",
        action="store_true",
        help="exit 1 if any logged miss has not been converted into a gate",
    )
    args = parser.parse_args(argv)

    path = REPO_ROOT / args.ledger if not Path(args.ledger).is_absolute() else Path(args.ledger)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"miss-metrics: cannot read {args.ledger}: {exc}", file=sys.stderr)
        return 2

    rows = parse_ledger(text)
    summary = summarize(rows)

    print("reviewer miss ledger metrics")
    print(f"ledger: {args.ledger}")
    print("-" * 60)
    if not _ledger_table_present(text):
        print(
            "FAIL: no Ledger table found; expected a '## Ledger' section with a "
            "table (renamed heading, deleted table, or wrong file?)",
            file=sys.stderr,
        )
        return 2
    if summary["total"] == 0:
        print("no escaped defects logged yet (ledger holds only the seed row).")
        return 0
    print(f"total escaped defects: {summary['total']}")
    print(f"  missed by human:     {summary['by_human']}")
    print(f"  missed by AI:        {summary['by_ai']}")
    print(f"  missed by CI:        {summary['by_ci']}")
    print(f"converted to a gate:   {summary['gated']}")
    print(f"not yet gated:         {summary['ungated']}")
    print(f"AI findings missed by human reviewer: {summary['ai_missed_by_human']}")

    if args.fail_on_ungated and summary["ungated"]:
        print("-" * 60)
        print("FAIL: ungated misses (every escaped defect must become a gate):")
        for row in ungated_rows(rows):
            print(f"  - {row.get('date', '?')}: {row.get('issue', '')[:60]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
