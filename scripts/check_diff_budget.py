#!/usr/bin/env python3
"""CI enforcement of the AGENTS.md 400 LOC diff budget (codification slice 1).

The Fable 5 arc (#1934) blew the soft cap 6/6 times with after-the-fact prose
justifications; a rule living only in docs does not bind an autonomous
builder. This gate mechanizes the existing contract, mirroring
scripts/check_ai_reconciliation_live.py: a PR whose ADDED line count exceeds
the budget must carry a line-anchored `Diff-budget override: <reason>` in the
PR body; placeholder reasons (TODO/TBD/empty) are rejected. Additions only --
deleting code is never penalized.

Exit codes: 0 = pass (under budget, or reasoned override); 1 = over budget
without a valid override; 2 = usage/API error (retryable, never silent).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_BUDGET = 400

# The override must be an explicit line-anchored marker, not prose that
# happens to mention the budget. The reason is everything after the colon.
# At most 3 leading spaces: 4+ (or a tab) is a Markdown indented code block,
# i.e. documentation of the syntax, not a decision.
OVERRIDE_RE = re.compile(
    r"^ {0,3}(?:[-*>]\s*)?\**diff[ -]?budget\s+override\**\s*:\s*(?P<reason>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Reasons that do not count as justification: empty, whitespace, bare
# punctuation, classic placeholders, or a copied <template> from the gate's
# own failure message / docs.
PLACEHOLDER_REASON_RE = re.compile(
    r"^(?:todo|tbd|n/?a|\.+|-+|<[^<>]*>)?$", re.IGNORECASE
)

# Fenced code blocks (``` or ~~~) are documentation, not decisions: a marker
# inside a fence (e.g. this gate's own syntax example) must not count.
FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~).*?(?:^[ \t]*(?:```|~~~)[ \t]*$|\Z)",
                      re.MULTILINE | re.DOTALL)


def find_override_reason(body: str) -> str | None:
    """Return the first substantive override reason, or None when no marker
    line exists at all.

    Every marker line is considered (an early placeholder must not shadow a
    later real reason). A marker present but never substantive returns "" so
    callers can distinguish marker-missing (None) from reason-missing ("").
    Substantive requires at least one alphanumeric character -- bare
    punctuation ("!!!") is a placeholder, not a justification.
    """
    text = FENCE_RE.sub("", body or "")
    found: str | None = None
    for match in OVERRIDE_RE.finditer(text):
        found = ""
        reason = match.group("reason").strip().strip("*").strip()
        # Classify on a punctuation-normalized form so "TODO." or
        # "<template>." cannot dodge the placeholder check via a stray dot.
        trimmed = reason.strip(" \t.*_~`'\"!?,;:()[]{}")
        if PLACEHOLDER_REASON_RE.fullmatch(trimmed):
            continue
        if not re.search(r"[A-Za-z0-9]", trimmed):
            continue
        return reason
    return found


def evaluate(additions: int, body: str, budget: int) -> tuple[int, list[str]]:
    """Core decision (pure). Returns (exit_code, messages)."""
    if additions <= budget:
        messages = [f"OK: {additions} added lines is within the {budget} LOC budget."]
        if find_override_reason(body) is not None:
            messages.append(
                "note: an override marker is present but not needed at this size."
            )
        return 0, messages

    reason = find_override_reason(body)
    overage = f"{additions} added lines exceeds the {budget} LOC soft cap"
    if reason:
        return 0, [
            f"OK (override): {overage}.",
            f"override reason: {reason}",
            "This overage is recorded as a deliberate decision (AGENTS.md diff budget).",
        ]
    if reason == "":
        return 1, [
            f"{overage}, and the 'Diff-budget override:' line carries no real "
            "reason (empty or placeholder).",
            "Write the actual justification after the colon -- why this slice "
            "is genuinely indivisible -- or split the PR.",
        ]
    return 1, [
        f"{overage} and the PR body has no override marker.",
        "Either split the slice, or add a line to the PR body:",
        "  Diff-budget override: <why this slice is genuinely indivisible>",
        "The check re-runs when the body is edited (AGENTS.md diff budget).",
    ]


def _gh(args: Sequence[str], gh: str) -> str:
    proc = subprocess.run([gh, *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return proc.stdout


def fetch_pr(pr: int, repo: str, gh: str) -> tuple[int, str]:
    out = _gh(
        ["pr", "view", str(pr), "--repo", repo, "--json", "additions,body"], gh
    )
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh returned non-JSON output: {exc}") from exc
    additions = data.get("additions")
    if not isinstance(additions, int) or isinstance(additions, bool):
        # Fail closed: a missing/null count must be a retryable error (exit 2),
        # never a silent zero that passes an over-budget PR.
        raise RuntimeError(f"gh response missing numeric 'additions': {additions!r}")
    return additions, str(data.get("body") or "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--pr", type=int, help="PR number (live mode)")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name (defaults to $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--gh", default="gh", help="path to the gh CLI")
    parser.add_argument(
        "--additions", type=int, help="added-line count (offline/test mode)"
    )
    parser.add_argument(
        "--body-file", help="PR body file (offline/test mode; skips the live fetch)"
    )
    args = parser.parse_args(argv)

    if args.budget < 1:
        print("diff budget: --budget must be >= 1", file=sys.stderr)
        return 2

    try:
        if args.additions is not None:
            additions = args.additions
            body = (
                Path(args.body_file).read_text(encoding="utf-8")
                if args.body_file
                else ""
            )
        elif args.pr is not None and args.repo:
            additions, body = fetch_pr(args.pr, args.repo, args.gh)
        else:
            print(
                "diff budget: need --pr and --repo (or $GITHUB_REPOSITORY), "
                "or --additions for offline mode",
                file=sys.stderr,
            )
            return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"diff budget: GitHub API/read error: {exc}", file=sys.stderr)
        return 2

    code, messages = evaluate(additions, body, args.budget)
    print("diff budget check")
    print("-" * 60)
    for line in messages:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
