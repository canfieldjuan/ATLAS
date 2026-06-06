#!/usr/bin/env python3
"""Validate the AI-finding reconciliation record in a PR body.

Mechanizes the AGENTS.md section 4a.1 / docs/REVIEWER_RULES.md rule that a PR
may not be LGTM'd until every automated-review (Codex/Copilot) finding is either
fixed or explicitly waived with a reason.

Local tooling cannot read live GitHub bot threads (gh is not present in the
local/CI bundle, see local_pr_review.sh), so this audit enforces the half that
is mechanically checkable from the PR body itself: when the body declares an
"AI reconciliation" section, that record must be internally resolved -- no
finding left unresolved, every waiver carrying a reason. Fail closed on a
contradictory or empty reconciliation block so a recorded reconciliation can be
trusted. With --require, also fail when the section is absent.

Cross-checking the recorded reconciliation against the live bot threads is a
CI-side follow-up (needs gh/API) and is intentionally out of scope here.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence

# A reconciliation section is anchored on a heading-like line so prose that
# merely mentions "reconciliation" elsewhere does not get treated as the record.
SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*{0,2})ai[ -]?reconciliation\b",
    re.IGNORECASE,
)
# A Markdown ATX heading; the capture group is used to compare heading levels so
# subheadings *inside* the record (e.g. "### Copilot") do not truncate it.
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+\S")
# Default level for a record anchored on a non-ATX line (bold or bare), so a
# later "##"/"#" still closes the record but "###+" subheadings stay inside it.
DEFAULT_SECTION_LEVEL = 2

# Markers that say the record is resolved (OR set -- any one is enough). Note
# the negative lookahead: a bare "no findings waived" must NOT count as
# resolution (it only says nothing was waived, not that findings were handled).
RESOLVED_RE = re.compile(
    r"(all\s+(?:findings\s+)?(?:fixed|fixed\s+or\s+waived)\s*:?\s*\byes\b"
    r"|no\s+(?:automated[ -]review\s+|outstanding\s+|remaining\s+)?findings\b(?!\s+waived)"
    r"|nothing\s+to\s+reconcile\b)",
    re.IGNORECASE,
)
# Markers that say the record is NOT resolved (any one fails the audit).
UNRESOLVED_RE = re.compile(
    r"(fixed\s+or\s+waived\s*:?\s*no\b"
    r"|all\s+(?:findings\s+)?fixed\s*:?\s*no\b"
    r"|findings?\s+(?:still\s+)?(?:open|outstanding|unaddressed|pending))",
    re.IGNORECASE,
)
# A waiver line that carries no rationale (placeholder or empty after the colon).
WAIVER_NO_REASON_RE = re.compile(
    r"waiv(?:e|ed|er)\b[^:\n]*:\s*(?:todo|tbd|\.*)?\s*$",
    re.IGNORECASE,
)


def _heading_level(line: str) -> int | None:
    match = HEADING_RE.match(line)
    return len(match.group(1)) if match else None


def extract_section(body: str) -> str | None:
    """Return the reconciliation record body, or None if there is no section.

    The record runs until the next heading at the same or higher level than the
    section heading, so subheadings (e.g. "### Copilot") stay inside the record
    and a later unresolved marker is not silently truncated away.
    """
    lines = body.splitlines()
    start = None
    section_level = DEFAULT_SECTION_LEVEL
    anchor_remainder = ""
    for idx, line in enumerate(lines):
        match = SECTION_RE.match(line)
        if match:
            start = idx
            level = _heading_level(line)
            if level is not None:
                section_level = level
            # Keep any inline content after the label so a one-line record like
            # "**AI reconciliation:** All fixed or waived: Yes" (the AGENTS.md
            # section 2a template shape) is validated, not read as empty.
            anchor_remainder = line[match.end():]
            break
    if start is None:
        return None
    collected: list[str] = []
    if anchor_remainder.strip():
        collected.append(anchor_remainder)
    for line in lines[start + 1:]:
        level = _heading_level(line)
        if level is not None and level <= section_level:
            break
        collected.append(line)
    return "\n".join(collected)


def reconciliation_errors(body: str, require: bool) -> list[str]:
    """Return reconciliation problems found in the PR body."""
    section = extract_section(body)
    if section is None:
        if require:
            return [
                "no 'AI reconciliation' section found; record the "
                "fixed-or-waived state of every automated-review finding "
                "(AGENTS.md section 4a.1)"
            ]
        return []

    errors: list[str] = []
    if UNRESOLVED_RE.search(section):
        errors.append(
            "reconciliation incomplete: an automated-review finding is "
            "unresolved (open/pending, or marked neither fixed nor waived); "
            "resolve or waive it with a reason before LGTM"
        )
    for line in section.splitlines():
        if WAIVER_NO_REASON_RE.search(line):
            errors.append(f"waived finding has no reason: {line.strip()!r}")
    if not errors and not RESOLVED_RE.search(section):
        errors.append(
            "reconciliation section present but states no resolution; add an "
            "explicit 'all fixed or waived: yes' / 'no findings' marker"
        )
    return errors


def read_body(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the AI-finding reconciliation record in a PR body."
    )
    parser.add_argument(
        "--current-pr-body-file",
        default=os.environ.get("ATLAS_CURRENT_PR_BODY_FILE") or None,
        help="path to the PR body file (defaults to $ATLAS_CURRENT_PR_BODY_FILE)",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="fail when the PR body has no AI reconciliation section at all",
    )
    args = parser.parse_args(argv)

    body_file = args.current_pr_body_file
    if not body_file:
        if args.require:
            print(
                "ai reconciliation audit: no PR body file provided; pass "
                "--current-pr-body-file or set ATLAS_CURRENT_PR_BODY_FILE",
                file=sys.stderr,
            )
            return 2
        print("ai reconciliation audit: no PR body file; skipped")
        return 0

    try:
        body = read_body(body_file)
    except OSError as exc:
        print(f"ai reconciliation audit: cannot read {body_file}: {exc}", file=sys.stderr)
        return 2

    errors = reconciliation_errors(body, require=args.require)

    print("ai reconciliation audit")
    print(f"pr body file: {body_file}")
    print("-" * 60)
    if errors:
        for error in errors:
            print(f"  - {error}")
        return 1
    print("OK: AI reconciliation record is resolved (or none required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
