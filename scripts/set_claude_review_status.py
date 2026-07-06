#!/usr/bin/env python3
"""Set the ``claude-review`` commit status that gates builder merges.

This is the reviewer half of the two-review merge gate. The Codex/bot half is
already machine-checkable via ``live-reconciliation`` (see
``scripts/check_ai_reconciliation_live.py``), which reds while unaccounted bot
threads exist. The reviewer (Claude Code) operates as the operator's GitHub
identity, so a Claude review is not a distinct machine signal on its own -- it
is prose review comments. This script turns the reviewer's verdict into a
per-SHA commit status named ``claude-review`` so the builder's merge condition
can require it next to ``live-reconciliation``.

Semantics (see docs/REVIEWER_MERGE_GATE.md and AGENTS.md 3c.1):
- ``success``  -> the reviewer reviewed *this exact head SHA* and found no
  BLOCKER (an LGTM, or only non-blocking MAJOR/NIT notes).
- ``failure``  -> the reviewer found a BLOCKER open at this head SHA.
- ``pending``  -> a review of this head SHA is in progress / not yet complete.
- absent       -> never reviewed at this SHA. A re-push produces a new SHA with
  no status, so a required ``claude-review`` gate is fail-closed by absence
  until the reviewer re-reviews and re-sets it.

The status ``context`` is hardcoded to ``claude-review`` and cannot be
overridden, so this tool can only ever set the reviewer gate, never spoof
another check.

Exit codes: 0 = status set (or would be set, under --dry-run); 2 = usage error
or a GitHub API failure (never a silent pass).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence

CONTEXT = "claude-review"
ALLOWED_STATES = ("success", "failure", "pending")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

_DEFAULT_DESCRIPTIONS = {
    "success": "Claude reviewed this head: no blocker.",
    "failure": "Claude reviewed this head: blocker open.",
    "pending": "Claude review of this head in progress.",
}


class UsageError(Exception):
    """Raised for operator-actionable argument or API failures (exit 2)."""


def _validate(repo: str, sha: str, state: str) -> None:
    if state not in ALLOWED_STATES:
        raise UsageError(
            f"state must be one of {', '.join(ALLOWED_STATES)}; got {state!r}"
        )
    if not _REPO_RE.match(repo):
        raise UsageError(f"repo must be owner/name; got {repo!r}")
    if not _SHA_RE.match(sha):
        raise UsageError(
            "sha must be a full 40-char hex commit SHA (the GitHub statuses API "
            f"rejects abbreviations); got {sha!r}"
        )


def build_gh_args(
    *, repo: str, sha: str, state: str, description: str, target_url: str | None
) -> list[str]:
    """Return the ``gh api`` argv that POSTs the claude-review status.

    Pure: no network. The context is always ``claude-review``.
    """
    _validate(repo, sha, state)
    args = [
        "gh",
        "api",
        "-X",
        "POST",
        f"repos/{repo}/statuses/{sha}",
        "-f",
        f"state={state}",
        "-f",
        f"context={CONTEXT}",
        "-f",
        f"description={description[:140]}",
    ]
    if target_url:
        args += ["-f", f"target_url={target_url}"]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name, e.g. canfieldjuan/ATLAS")
    parser.add_argument("--sha", required=True, help="head commit SHA to attach the status to")
    parser.add_argument("--state", required=True, help=f"one of {', '.join(ALLOWED_STATES)}")
    parser.add_argument("--description", default=None, help="short status description")
    parser.add_argument("--pr", default=None, help="PR number, used only to build the target_url")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the gh argv that would run and exit 0 without calling GitHub",
    )
    ns = parser.parse_args(argv)

    description = ns.description or _DEFAULT_DESCRIPTIONS[
        ns.state if ns.state in _DEFAULT_DESCRIPTIONS else "pending"
    ]
    target_url = (
        f"https://github.com/{ns.repo}/pull/{ns.pr}" if ns.pr else None
    )

    try:
        args = build_gh_args(
            repo=ns.repo,
            sha=ns.sha,
            state=ns.state,
            description=description,
            target_url=target_url,
        )
    except UsageError as exc:
        print(f"set_claude_review_status: {exc}", file=sys.stderr)
        return 2

    if ns.dry_run:
        print(" ".join(args))
        return 0

    try:
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        print(
            "set_claude_review_status: gh not found on PATH; install the GitHub CLI",
            file=sys.stderr,
        )
        return 2
    if proc.returncode != 0:
        print(
            f"set_claude_review_status: gh failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}",
            file=sys.stderr,
        )
        return 2
    print(f"set {CONTEXT}={ns.state} on {ns.repo}@{ns.sha[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
