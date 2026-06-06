#!/usr/bin/env python3
"""Drain stale parked items out of HARDENING.md and keep it lean.

HARDENING.md is an append-only register: sessions park non-blocking hardening
discoveries under dated ``## YYYY-MM-DD`` headings (newest first), and the file
itself asks to "Periodically drain stale entries". Left alone it grows without
bound, so a fresh session scanning it for same-lane parked items pays an
ever-growing orientation tax.

This tool gives the register a lifecycle:

- ``drain``  move every parked entry older than ``--max-age-days`` (default 90)
             out of HARDENING.md and append it to the drain archive
             (``docs/technical-debt/hardening-archive.md``). The preamble, the
             standing footer note, and any still-fresh entries are preserved.
- ``check``  print a non-blocking warning when HARDENING.md exceeds
             ``--max-lines`` (default 200) or holds an entry older than
             ``--max-age-days``. Always exits 0 so it never adds PR friction.

Parsing is anchored on the literal ``## Parked Items`` marker, so the example
``## YYYY-MM-DD`` heading inside the fenced "Entry Format" block in the preamble
is never mistaken for a real entry.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PARKED_MARKER = "## Parked Items"
DEFAULT_MAX_AGE_DAYS = 90
DEFAULT_MAX_LINES = 200
DEFAULT_ARCHIVE = REPO_ROOT / "docs" / "technical-debt" / "hardening-archive.md"

_DATE_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$")


@dataclass(frozen=True)
class Entry:
    date_str: str
    body: str

    @property
    def date_value(self) -> date | None:
        try:
            return date.fromisoformat(self.date_str)
        except ValueError:
            return None


def split_sections(text: str) -> tuple[str, str, str]:
    """Split HARDENING.md into (preamble, entries_region, footer).

    The preamble runs through the ``## Parked Items`` marker. The footer is the
    trailing block-quote note (if any). Everything between is the dated entries.
    """
    lines = text.splitlines()
    marker_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == PARKED_MARKER), None
    )
    if marker_idx is None:
        return text.rstrip("\n"), "", ""

    preamble = "\n".join(lines[: marker_idx + 1])
    remainder = lines[marker_idx + 1 :]

    foot_start = len(remainder)
    j = len(remainder) - 1
    while j >= 0 and (remainder[j].strip() == "" or remainder[j].lstrip().startswith(">")):
        foot_start = j
        j -= 1
    footer_lines = remainder[foot_start:]
    if not any(line.lstrip().startswith(">") for line in footer_lines):
        footer_lines = []
        foot_start = len(remainder)

    middle = "\n".join(remainder[:foot_start])
    footer = "\n".join(footer_lines).strip("\n")
    return preamble, middle, footer


def _entry_starts(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if _DATE_HEADING.match(line)]


def leading_text(entries_region: str) -> str:
    """Region text before the first dated entry (undated notes, typo'd headings).

    This is the only content in the entries region not carried by a dated block, so
    it must be preserved verbatim across a drain rather than silently dropped.
    """
    lines = entries_region.splitlines()
    starts = _entry_starts(lines)
    if not starts:
        return entries_region.strip("\n")
    return "\n".join(lines[: starts[0]]).strip("\n")


def parse_entries(entries_region: str) -> list[Entry]:
    """Parse the entries region into dated ``Entry`` blocks."""
    lines = entries_region.splitlines()
    starts = _entry_starts(lines)
    entries: list[Entry] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        date_str = _DATE_HEADING.match(lines[start]).group(1)  # type: ignore[union-attr]
        body = "\n".join(lines[start:end]).strip("\n")
        entries.append(Entry(date_str=date_str, body=body))
    return entries


def partition_by_age(
    entries: list[Entry], today: date, max_age_days: int
) -> tuple[list[Entry], list[Entry]]:
    """Return (kept, drained). Unparseable dates are always kept."""
    cutoff = today - timedelta(days=max_age_days)
    kept: list[Entry] = []
    drained: list[Entry] = []
    for entry in entries:
        value = entry.date_value
        if value is not None and value < cutoff:
            drained.append(entry)
        else:
            kept.append(entry)
    return kept, drained


def render_hardening(
    preamble: str, leading: str, kept: list[Entry], footer: str
) -> str:
    """Reassemble HARDENING.md from preamble, carried leading text, kept, footer."""
    parts = [preamble.strip("\n")]
    if leading.strip():
        parts.append(leading.strip("\n"))
    parts.extend(entry.body.strip("\n") for entry in kept)
    if footer.strip():
        parts.append(footer.strip("\n"))
    return "\n\n".join(parts).rstrip() + "\n"


def append_to_archive(archive_path: Path, drained: list[Entry]) -> None:
    """Append drained blocks to the archive (O(1) append; header only on create)."""
    if not drained:
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n\n".join(entry.body.strip("\n") for entry in drained)
    is_new = not archive_path.exists() or archive_path.stat().st_size == 0
    with archive_path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write(
                "# HARDENING archive\n\n"
                "Parked items drained from HARDENING.md by scripts/drain_hardening.py.\n"
            )
        handle.write("\n" + blocks + "\n")


def oldest_entry_age_days(entries: list[Entry], today: date) -> int | None:
    ages = [(today - e.date_value).days for e in entries if e.date_value is not None]
    return max(ages) if ages else None


def drain(
    hardening_path: Path, archive_path: Path, today: date, max_age_days: int
) -> list[Entry]:
    """Drain stale entries; write the leaner HARDENING.md and the archive."""
    text = hardening_path.read_text(encoding="utf-8")
    preamble, region, footer = split_sections(text)
    leading = leading_text(region)
    entries = parse_entries(region)
    kept, drained = partition_by_age(entries, today, max_age_days)
    if drained:
        append_to_archive(archive_path, drained)
        hardening_path.write_text(
            render_hardening(preamble, leading, kept, footer), encoding="utf-8"
        )
    return drained


def check_warnings(
    text: str, today: date, max_age_days: int, max_lines: int
) -> list[str]:
    warnings: list[str] = []
    line_count = len(text.splitlines())
    if line_count > max_lines:
        warnings.append(f"{line_count} lines exceeds threshold {max_lines}")
    _, region, _ = split_sections(text)
    entries = parse_entries(region)
    oldest = oldest_entry_age_days(entries, today)
    if oldest is not None and oldest > max_age_days:
        warnings.append(f"oldest parked entry is {oldest} days old (> {max_age_days})")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("drain", "check"))
    parser.add_argument(
        "--hardening-file", type=Path, default=REPO_ROOT / "HARDENING.md"
    )
    parser.add_argument("--archive-file", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument(
        "--today", default=None, help="override today (YYYY-MM-DD) for testing"
    )
    args = parser.parse_args(argv)

    hardening_path: Path = args.hardening_file
    if not hardening_path.is_file():
        print(f"hardening file not found: {hardening_path}", file=sys.stderr)
        return 2

    today = date.fromisoformat(args.today) if args.today else date.today()

    if args.command == "drain":
        drained = drain(hardening_path, args.archive_file, today, args.max_age_days)
        if drained:
            print(
                f"drained {len(drained)} stale entry(s) -> {args.archive_file}"
            )
            print(f"rewrote {hardening_path}")
        else:
            print(f"nothing to drain (no entry older than {args.max_age_days} days).")
        return 0

    # check: non-blocking nudge, always exits 0
    warnings = check_warnings(
        hardening_path.read_text(encoding="utf-8"), today, args.max_age_days, args.max_lines
    )
    if warnings:
        for warning in warnings:
            print(f"WARNING: {hardening_path}: {warning}")
        print(
            "    run `python scripts/drain_hardening.py drain` to archive stale entries."
        )
    else:
        print(f"OK: {hardening_path} within thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
