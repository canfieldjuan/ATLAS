#!/usr/bin/env python3
"""Advisory detector for AGENTS.md 3k.2 (convergence circuit-breaker).

3k.2 already defines the failure: when each push closes the reported review
threads but the next push opens a comparable count of same-class findings on the
same file, the builder is instance-patching a shared decision rather than fixing
it. The rule is mandatory and names the required response (a Decision-Seam
Analysis, and a ban on "another token, regex, vocabulary row, or oracle
fixture"). It has never fired on its own, because it lives in a 73 KB document
and needs a human to notice and count rounds.

This makes it fire. It buckets bot review findings into the push that preceded
them and trips when, over three consecutive pushes, the finding count is not
trending to zero and one file dominates the window. The dominant file is the
seam 3k.2 asks the builder to name.

Measured on ATLAS #2181 (94 findings over 20 pushes, dead flat, every round on
content_factory_copy_verification.py): this trips at push 3, with 17 pushes and
76 of the 94 findings still to come.

Deliberately advisory. It emits ``::warning`` annotations and exits 0, exactly
like scripts/check_guard_class_closure.py (the 3k.1 sibling). It is NOT a round
cap: it never blocks, never fails, and never stops a PR -- capping counts the
symptom, and on surfaces where the findings are real it would ship defects. It
only changes the class of fix the next push is allowed to make.

Exit codes: 0 = no trip, trip suppressed by a Decision-Seam Analysis, or a trip
in advisory mode; 1 = trip under --strict (reserved for a future promotion);
2 = usage error or a GitHub API failure (retryable, never a silent pass).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Same bot set and env override as scripts/check_ai_reconciliation_live.py, so
# the two checks cannot disagree about who a review bot is.
_DEFAULT_BOTS = ("copilot", "codex")

# The PR body marker that records the builder did what 3k.2 asks. Mirrors the
# WAIVER_MARKER convention in scripts/check_guard_class_closure.py.
SEAM_ANALYSIS_MARKER = "decision-seam analysis"

# Trip parameters. A window of 3 matches 3k.2's "over 3 consecutive pushes".
WINDOW = 3
# "Not trending to zero": the last push in the window still carries at least
# this share of the first push's findings. Halving across three pushes is
# convergence; anything flatter is not.
CONVERGENCE_RATIO = 0.5
# One file must account for more than this share of the window's findings for
# the threads to be "the same decision" rather than scattered review.
SEAM_DOMINANCE = 0.5

_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      commits(first:100){ nodes{ commit{ oid committedDate } } }
      reviewThreads(first:100, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          path
          comments(first:1){ nodes{ author{ login } createdAt } }
        }
      }
    }
  }
}
"""

# Defensive cap on pagination (100 threads/page) so a pathological PR can never
# loop unbounded; far above any real review.
_MAX_THREAD_PAGES = 50


@dataclass(frozen=True)
class _Commit:
    """A commit's push time and sha, named so ordering never indexes a tuple."""

    when: str
    sha: str


@dataclass
class PushRound:
    """One push and the bot findings that landed after it."""

    index: int
    sha: str
    pushed_at: str
    paths: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.paths)


def bot_findings(nodes: Sequence[dict], bot_logins: Sequence[str]) -> list[tuple[str, str]]:
    """Return (created_at, path) for each bot-authored review thread.

    Pure, so the decision is testable without touching GitHub. Unlike the
    reconciliation check this deliberately counts resolved and outdated threads
    too: a thread the builder already closed is exactly the instance-patch 3k.2
    is looking for, so excluding it would hide the pattern.
    """
    wanted = tuple(b.lower() for b in bot_logins)
    found: list[tuple[str, str]] = []
    for node in nodes or []:
        comments = ((node.get("comments") or {}).get("nodes")) or []
        first = next(iter(comments), None)
        if not isinstance(first, dict):
            continue
        author = ((first.get("author") or {}).get("login") or "").lower()
        created = first.get("createdAt") or ""
        if not created:
            continue
        if not any(w in author for w in wanted):
            continue
        found.append((created, node.get("path") or "?"))
    found.sort()
    return found


def assign_findings_to_pushes(
    commits: Sequence[dict],
    findings: Sequence[tuple[str, str]],
) -> list[PushRound]:
    """Bucket each finding into the push it followed.

    3k.2 counts "consecutive pushes", so pushes are the unit -- not wall-clock
    gaps, which are not deterministic. A finding belongs to the latest commit
    at or before its creation time; a finding predating every commit belongs to
    no push and is dropped (it cannot be a response to one).

    ISO-8601 UTC timestamps from the GitHub API are lexicographically ordered,
    so string comparison is exact here and avoids a parsing dependency.
    """
    dated: list[_Commit] = []
    for c in commits or []:
        blob = c.get("commit") or {}
        when = str(blob.get("committedDate") or "")
        if when:
            dated.append(_Commit(when=when, sha=str(blob.get("oid") or "")))
    dated.sort(key=lambda commit: commit.when)
    rounds = [
        PushRound(index=i + 1, sha=commit.sha, pushed_at=commit.when)
        for i, commit in enumerate(dated)
    ]
    if not rounds:
        return []
    for created, path in findings:
        target = None
        for r in rounds:
            if r.pushed_at <= created:
                target = r
            else:
                break
        if target is not None:
            target.paths.append(path)
    return rounds


def dominant_seam(paths: Sequence[str]) -> tuple[str, int] | None:
    """Return (path, count) when one file carries most of the findings."""
    if not paths:
        return None
    top = next(iter(Counter(paths).most_common(1)), None)
    if top is None:
        return None
    path, hits = top
    if hits / len(paths) > SEAM_DOMINANCE:
        return path, hits
    return None


def find_trip(rounds: Sequence[PushRound]) -> tuple[int, str, list[PushRound]] | None:
    """First window of WINDOW consecutive pushes that is not converging.

    Returns (trip_index, seam_path, window_rounds), or None. A push with zero
    findings breaks the streak: the bot finding nothing is convergence.
    """
    for i in range(len(rounds) - WINDOW + 1):
        window = list(rounds[i : i + WINDOW])
        if any(r.count == 0 for r in window):
            continue
        first_push = next(iter(window), None)
        last_push = next(iter(reversed(window)), None)
        if first_push is None or last_push is None:
            continue
        if last_push.count < first_push.count * CONVERGENCE_RATIO:
            continue
        seam = dominant_seam([p for r in window for p in r.paths])
        if seam is None:
            continue
        seam_path, _hits = seam
        return last_push.index, seam_path, window
    return None


def body_has_seam_analysis(body: str) -> bool:
    return SEAM_ANALYSIS_MARKER in (body or "").lower()


def evaluate(
    rounds: Sequence[PushRound],
    body: str,
) -> tuple[bool, str | None, list[str]]:
    """Core decision (pure). Returns (tripped, seam_path, messages)."""
    messages: list[str] = []
    with_findings = [r for r in rounds if r.count]
    if with_findings:
        messages.append("findings per push:")
        for r in rounds:
            if r.count:
                messages.append(
                    "  push %d (%s)  %d  %s"
                    % (r.index, r.pushed_at[:16], r.count, "#" * r.count)
                )
    trip = find_trip(rounds)
    if trip is None:
        messages.append(
            "OK: no window of %d consecutive pushes with same-seam findings that "
            "are not trending to zero." % WINDOW
        )
        return False, None, messages
    trip_index, seam, window = trip
    counts = ", ".join(str(r.count) for r in window)
    if body_has_seam_analysis(body):
        messages.append(
            "SATISFIED: non-convergence on %s at push %d (%s), but the PR body "
            "carries a Decision-Seam Analysis." % (seam, trip_index, counts)
        )
        return False, seam, messages
    messages.append(
        "AGENTS 3k.2 tripped at push %d: findings per push %s on %s are not "
        "trending to zero." % (trip_index, counts, seam)
    )
    return True, seam, messages


def annotation(seam: str, rounds: Sequence[PushRound]) -> str:
    trip = find_trip(rounds)
    counts = "?"
    if trip is not None:
        _index, _seam, window = trip
        counts = ", ".join(str(r.count) for r in window)
    return (
        "::warning file=%s::AGENTS 3k.2 convergence circuit-breaker: findings per "
        "push (%s) on this file are not trending to zero, so the threads are the "
        "same decision re-litigated. The next push may NOT add another token, "
        "regex, vocabulary row, or oracle fixture. It must carry a Decision-Seam "
        "Analysis in the PR body: name the one decision the threads share, state "
        "why it is wrong (over-broad, under-broad, or an open category it cannot "
        "enumerate), then do exactly one of -- fix the seam structurally with a "
        "stated default direction; waive the bounded residual in Deferred; or "
        "re-scope the slice. See AGENTS.md 3k.2 and docs/GUARD_CLASS_CLOSURE.md."
        % (seam, counts)
    )


def _gh(args: Sequence[str], gh: str) -> str:
    proc = subprocess.run([gh, *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return proc.stdout


def fetch(pr: int, owner: str, name: str, gh: str) -> tuple[list[dict], list[dict]]:
    """Fetch commits and ALL review threads, paginating threads."""
    commits: list[dict] = []
    threads: list[dict] = []
    cursor: str | None = None
    for _ in range(_MAX_THREAD_PAGES):
        args = [
            "api", "graphql",
            "-f", f"query={_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"name={name}",
            "-F", f"pr={pr}",
        ]
        if cursor:
            args += ["-F", f"cursor={cursor}"]
        try:
            data = json.loads(_gh(args, gh))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GitHub returned non-JSON: {exc}") from exc
        node = (((data.get("data") or {}).get("repository") or {}).get("pullRequest")) or {}
        if not commits:
            commits = ((node.get("commits") or {}).get("nodes")) or []
        block = (node.get("reviewThreads") or {})
        threads.extend(block.get("nodes") or [])
        page = block.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
    return commits, threads


def _pr_body(pr: int, repo: str, gh: str) -> str:
    """Prefer the CI-provided body file; fall back to the API."""
    path = os.environ.get("ATLAS_CURRENT_PR_BODY_FILE")
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    try:
        return _gh(
            ["pr", "view", str(pr), "--repo", repo, "--json", "body", "-q", ".body"],
            gh,
        )
    except RuntimeError as exc:
        # Not silent: an unreadable body means a Decision-Seam Analysis cannot be
        # detected, so the run falls toward reporting the trip. Say so.
        print(f"warning: could not read the PR body ({exc}); "
              "a Decision-Seam Analysis cannot be detected", file=sys.stderr)
        return ""


def main(argv: Sequence[str] | None = None) -> int:
    summary = next(iter((__doc__ or "").splitlines()), "seam-convergence breaker")
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument("--pr", type=int, required=True, help="pull request number")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--gh", default="gh", help="path to the gh CLI")
    parser.add_argument(
        "--bots",
        default=os.environ.get("ATLAS_REVIEW_BOTS", ",".join(_DEFAULT_BOTS)),
        help="comma-separated review-bot login substrings",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on a trip (for a future required enrollment)",
    )
    args = parser.parse_args(argv)

    if "/" not in args.repo:
        print("error: --repo must be owner/name", file=sys.stderr)
        return 2
    owner, name = args.repo.split("/", 1)
    bots = [b.strip() for b in args.bots.split(",") if b.strip()]

    try:
        commits, threads = fetch(args.pr, owner, name, args.gh)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: GitHub API failure: {exc}", file=sys.stderr)
        return 2

    rounds = assign_findings_to_pushes(commits, bot_findings(threads, bots))
    tripped, seam, messages = evaluate(rounds, _pr_body(args.pr, args.repo, args.gh))

    print("seam-convergence breaker (advisory) -- AGENTS 3k.2")
    print("-" * 60)
    for line in messages:
        print(line)
    if not tripped:
        return 0
    print(annotation(seam or "?", rounds))
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
