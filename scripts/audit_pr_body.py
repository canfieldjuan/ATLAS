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
import subprocess
import sys
from typing import Callable, Sequence


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
DEPENDABOT_AUTHORS = frozenset(
    {
        "app/dependabot",
        "dependabot",
        "dependabot[bot]",
    }
)


def is_dependabot_author(author: str | None) -> bool:
    """Return true for Dependabot identities seen in GitHub PR events."""

    if author is None:
        return False
    return author.strip() in DEPENDABOT_AUTHORS


def resolve_git_ref(ref: str, *, repo_root: Path = ROOT) -> bool:
    """True when ``ref`` resolves to a commit in the local repo. The
    trusted-base workflow fetches the PR head before auditing; an
    unresolvable ref is an infrastructure failure, never a silent pass."""
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
    )
    return proc.returncode == 0


def plan_exists_at_ref(ref: str, *, repo_root: Path = ROOT) -> Callable[[str], bool]:
    """Plan-doc existence checker against a git ref (the fetched PR head).
    Trusted-base gate runs execute this script from the BASE checkout, but a
    new plan doc arrives WITH the PR -- so existence is asked of the PR head
    ref by inspection (git cat-file), never by executing PR code."""

    def _exists(plan: str) -> bool:
        # ls-tree mode, not cat-file existence: the entry must be a
        # REGULAR-FILE blob. cat-file -e also accepts a tree at the path
        # (plans/PR-Foo.md/child) and cat-file -t reports symlinks as
        # blobs (mode 120000, possibly dangling) -- neither is a plan
        # doc. Requiring mode 100644/100755 mirrors the working-tree
        # default's is_file().
        proc = subprocess.run(
            ["git", "ls-tree", ref, "--", plan],
            cwd=repo_root,
            capture_output=True,
        )
        fields = proc.stdout.split()
        return (
            proc.returncode == 0
            and len(fields) >= 2
            and fields[0] in (b"100644", b"100755")
            and fields[1] == b"blob"
        )

    return _exists


def audit_pr_body(
    body: str,
    *,
    root: Path = ROOT,
    plan_exists: Callable[[str], bool] | None = None,
) -> list[str]:
    """Return a list of contract failures (empty means the body passes)."""

    if plan_exists is None:
        def plan_exists(plan: str) -> bool:
            return (root / plan).is_file()

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
        if not plan_exists(plan_match.group("plan")):
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
    why_lines = [
        line.strip()
        for line in lead_lines
        if line.strip()
        and PLAN_LINE_RE.match(line.strip()) is None
        and SLICE_PHASE_RE.match(line.strip()) is None
    ]
    if not why_lines:
        failures.append(
            "missing the one-paragraph why between the lead lines and the "
            "first '##' section"
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
    parser.add_argument(
        "--pr-author",
        default="",
        help="GitHub PR author login; Dependabot PRs are exempt",
    )
    parser.add_argument(
        "--plan-git-ref",
        default="",
        help=(
            "check plan-doc existence against this git ref (the fetched PR "
            "head) instead of the working tree -- for trusted-base gate runs"
        ),
    )
    parser.add_argument("body_file", help="path to a file holding the PR body")
    args = parser.parse_args(argv)

    body_path = Path(args.body_file)
    if not body_path.is_file():
        print(f"pr body audit: body file not found: {body_path}", file=sys.stderr)
        return 2
    body = body_path.read_text(encoding="utf-8", errors="replace")

    if is_dependabot_author(args.pr_author):
        print("pr body audit: PASS (Dependabot PR body exempt)")
        return 0

    plan_exists = None
    if args.plan_git_ref:
        if not resolve_git_ref(args.plan_git_ref):
            print(
                f"pr body audit: plan ref not resolvable: {args.plan_git_ref} "
                "(fetch the PR head before auditing)",
                file=sys.stderr,
            )
            return 2  # infrastructure failure -- never a silent pass
        plan_exists = plan_exists_at_ref(args.plan_git_ref)

    failures = audit_pr_body(body, plan_exists=plan_exists)
    if failures:
        print("pr body audit: FAIL (AGENTS.md section 1b contract)")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("pr body audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
