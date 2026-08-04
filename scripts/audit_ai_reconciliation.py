#!/usr/bin/env python3
"""Validate the AI-finding reconciliation record in a PR body.

Mechanizes the AGENTS.md section 4a.1 / docs/REVIEWER_RULES.md rule that a PR
may not be LGTM'd until every Codex connector finding is either fixed or
explicitly waived with a reason.

Local tooling cannot read live GitHub bot threads (gh is not present in the
local/CI bundle, see local_pr_review.sh), so this audit enforces the half that
is mechanically checkable from the PR body itself: when the body declares an
"AI reconciliation" section, that record must be structured -- either
`no-findings` or one allowed disposition per finding, with evidence/reason text.
Fail closed on vague, contradictory, or empty reconciliation blocks so a
recorded reconciliation can be trusted. With --require, also fail when the
section is absent.

Cross-checking the recorded reconciliation against the live Codex threads is done
by the CI-side companion `scripts/check_ai_reconciliation_live.py` (needs
gh/API), which fails when this recorded reconciliation omits a still-open bot
finding. This local audit owns the body-shape half; that one owns reality.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
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
    r"|nothing\s+to\s+reconcile\b"
    r"|\bno-findings\b)",
    re.IGNORECASE,
)
# Markers that say the record is NOT resolved (any one fails the audit).
UNRESOLVED_RE = re.compile(
    r"(fixed\s+or\s+waived\s*:?\s*no\b"
    r"|findings\s+reviewed\s*:?\s*no\b"
    r"|all\s+(?:findings\s+)?fixed\s*:?\s*no\b"
    r"|findings?\s+(?:still\s+)?(?:open|outstanding|unaddressed|pending))",
    re.IGNORECASE,
)
_EMPTY_REASON_WORD = "to" "do"
# A waiver line that carries no rationale or only a placeholder after the colon.
WAIVER_NO_REASON_RE = re.compile(
    r"waiv(?:e|ed|er)\b[^:\n]*:\s*(?:" + _EMPTY_REASON_WORD + r"|tbd|\.*)?\s*$",
    re.IGNORECASE,
)
DISPOSITIONS = (
    "fixed-in",
    "waived-duplicate",
    "waived-out-of-scope",
    "waived-speculative",
    "waived-nit",
    "not-applicable",
)
DISPOSITION_RE = re.compile(
    r"\b(?P<name>"
    + "|".join(re.escape(name) for name in DISPOSITIONS)
    + r")\s*:\s*(?P<detail>\S.*)$",
    re.IGNORECASE,
)
DISPOSITION_TOKEN_RE = re.compile(
    r"\b(?P<name>"
    + "|".join(re.escape(name) for name in DISPOSITIONS)
    + r")\s*:",
    re.IGNORECASE,
)
NO_FINDINGS_RE = re.compile(r"^(?:[-*]\s*)?no-findings\s*\.?\s*$", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(?P<body>\S.*)$")
SUMMARY_LINE_RE = re.compile(
    r"^(?:[-*]\s*)?"
    r"(?:ai|codex|automated-review)?\s*findings\s+reviewed\s*:?\s*(?:yes|no)\s*\.?$"
    r"|^(?:[-*]\s*)?all\s+(?:findings\s+)?fixed\s+or\s+waived\s*:?\s*(?:yes|no)\s*\.?$"
    r"|^(?:[-*]\s*)?all\s+(?:findings\s+)?fixed\s*:?\s*(?:yes|no)\s*\.?$",
    re.IGNORECASE,
)
PLACEHOLDER_DETAIL_RE = re.compile(
    r"^(?:" + _EMPTY_REASON_WORD + r"|tbd|n/?a|none|\.*|-+)\s*\.?$",
    re.IGNORECASE,
)
FINDING_SEPARATOR_RE = re.compile(r"\s+(?:--|—|–)\s+")
FIXED_IN_EVIDENCE_RE = re.compile(
    r"(?i)(?:\b[0-9a-f]{7,40}\b|(?:^|[\s,;])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.(?:py|sh|md|yml|yaml|json|toml|txt)\b)"
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
    disposition_count = 0
    no_findings = False
    if UNRESOLVED_RE.search(section):
        errors.append(
            "reconciliation incomplete: an automated-review finding is "
            "unresolved (open/pending, or marked neither fixed nor waived); "
            "resolve or waive it with a reason before LGTM"
        )

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if SUMMARY_LINE_RE.fullmatch(line):
            continue
        if NO_FINDINGS_RE.fullmatch(line):
            no_findings = True
            continue

        bullet = BULLET_RE.match(raw_line)
        if bullet is None:
            errors.append(
                "reconciliation line is not structured; use 'no-findings' or "
                f"one allowed disposition per bullet: {line!r}"
            )
            continue

        item = bullet.group("body").strip()
        if WAIVER_NO_REASON_RE.search(item):
            errors.append(f"waived finding has no reason: {item!r}")
            continue
        tokens = list(DISPOSITION_TOKEN_RE.finditer(item))
        if len(tokens) != 1:
            errors.append(
                "reconciliation bullet must contain exactly one allowed disposition "
                f"({', '.join(DISPOSITIONS)}): {item!r}"
            )
            continue
        disposition = DISPOSITION_RE.search(item)
        if disposition is None:  # pragma: no cover - guarded by token count
            errors.append(
                "reconciliation bullet lacks an allowed disposition "
                f"({', '.join(DISPOSITIONS)}): {item!r}"
            )
            continue
        finding = item[: disposition.start()].strip()
        finding = FINDING_SEPARATOR_RE.sub(" ", finding).strip(" :-–—")
        if not finding:
            errors.append(
                "reconciliation bullet must name the finding/root decision before "
                f"the disposition: {item!r}"
            )
            continue
        detail = disposition.group("detail").strip()
        if PLACEHOLDER_DETAIL_RE.fullmatch(detail):
            errors.append(
                f"{disposition.group('name').lower()} disposition has no usable "
                f"evidence/reason: {item!r}"
            )
            continue
        if disposition.group("name").lower() == "fixed-in" and not FIXED_IN_EVIDENCE_RE.search(detail):
            errors.append(
                f"fixed-in disposition must cite a commit, file, or test path: {item!r}"
            )
            continue
        disposition_count += 1

    if no_findings and disposition_count:
        errors.append("'no-findings' cannot be mixed with finding disposition bullets")
    if not no_findings and disposition_count == 0:
        errors.append(
            "reconciliation section must include 'no-findings' or at least one "
            "finding bullet with an allowed disposition "
            f"({', '.join(DISPOSITIONS)})"
        )
    return errors


def read_body(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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
