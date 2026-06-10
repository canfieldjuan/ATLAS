#!/usr/bin/env python3
"""Audit a PR body against the AGENTS.md section 1b contract.

The PR body must lead with ``Plan: plans/PR-<Slice-Name>.md`` and a
``Slice phase: <phase>`` line, then carry these ``##`` sections in order:
Intentional, Deferred, Parked hardening, Verification, Diff size. The
referenced plan doc must exist in the checkout.

Intended for CI on ``pull_request`` events (the workflow writes
``github.event.pull_request.body`` to a file - no GitHub API call), and for
local use before opening a PR:

    python scripts/audit_pr_body.py /tmp/pr_body.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PLAN_LINE_RE = re.compile(r"^Plan:\s+(?P<plan>plans/PR-[A-Za-z0-9._-]+\.md)\s*$")
SLICE_PHASE_RE = re.compile(r"^Slice phase:\s*\S.*$")
HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")
REQUIRED_SECTIONS = (
    "Intentional",
    "Deferred",
    "Parked hardening",
    "Verification",
    "Diff size",
)


def audit_pr_body(body: str, *, root: Path = ROOT) -> list[str]:
    """Return a list of contract failures (empty means the body passes)."""

    failures: list[str] = []
    lines = body.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        return ["PR body is empty"]

    plan_match = PLAN_LINE_RE.match(nonempty[0])
    if plan_match is None:
        failures.append(
            "first non-empty line must be 'Plan: plans/PR-<Slice-Name>.md'"
        )
    else:
        plan_path = root / plan_match.group("plan")
        if not plan_path.is_file():
            failures.append(
                f"plan doc named in the PR body does not exist: {plan_match.group('plan')}"
            )

    first_heading_index = next(
        (index for index, line in enumerate(lines) if HEADING_RE.match(line)),
        len(lines),
    )
    lead_lines = lines[:first_heading_index]
    if not any(SLICE_PHASE_RE.match(line.strip()) for line in lead_lines):
        failures.append(
            "missing 'Slice phase: <phase>' line before the first '##' section"
        )

    headings = [
        match.group("title")
        for line in lines
        if (match := HEADING_RE.match(line))
    ]
    missing = [title for title in REQUIRED_SECTIONS if title not in headings]
    for title in missing:
        failures.append(f"missing required section: ## {title}")
    present_in_order = [title for title in headings if title in REQUIRED_SECTIONS]
    expected_order = [title for title in REQUIRED_SECTIONS if title in headings]
    if not missing and present_in_order != expected_order:
        failures.append(
            "required sections are out of order; expected "
            + " -> ".join(f"## {title}" for title in REQUIRED_SECTIONS)
        )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body_file", help="path to a file holding the PR body")
    args = parser.parse_args(argv)

    body_path = Path(args.body_file)
    if not body_path.is_file():
        print(f"pr body audit: body file not found: {body_path}", file=sys.stderr)
        return 2
    body = body_path.read_text(encoding="utf-8", errors="replace")

    failures = audit_pr_body(body)
    if failures:
        print("pr body audit: FAIL (AGENTS.md section 1b contract)")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("pr body audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
