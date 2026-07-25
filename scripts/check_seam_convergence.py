#!/usr/bin/env python3
"""Advisory detector for AGENTS.md 3k.2 (convergence circuit-breaker).

3k.2 already defines the failure: when each push closes the reported review
threads but the next push opens a comparable count of same-class findings on the
same file, the builder is instance-patching a shared decision rather than fixing
it. The rule is mandatory and names the required response (a Decision-Seam
Analysis, and a ban on "another token, regex, vocabulary row, or oracle
fixture"). It has never fired on its own, because it lives in a 73 KB document
and needs a human to notice and count rounds.

This makes it fire. A round is one bot review submission -- the bot's response
to a push -- which is the unit 3k.2 actually means and which, unlike commits,
cannot be split by a multi-commit push or skewed by commit dates that differ
from push times. The breaker trips when, across three consecutive rounds, the
finding count is flat or rising rather than trending to zero AND one file leads
every one of those rounds. That file is the seam 3k.2 asks the builder to name.

Measured on ATLAS #2181 (18 bot review rounds, dead flat, the same file leading
every round): this trips at round 3, with 15 rounds still to come.

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
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Same bot set and env override as scripts/check_ai_reconciliation_live.py, so
# the two checks cannot disagree about who a review bot is.
_DEFAULT_BOTS = ("copilot", "codex")

# A Decision-Seam Analysis suppresses the trip, so the marker must be a real
# section rather than any mention of the phrase: a body saying "no
# Decision-Seam Analysis yet" must NOT read as satisfied.
_SEAM_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*decision[-\s]seam\s+analysis\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# 3k.2 step 3: the disposition must be one of fix / waive / re-scope.
_DISPOSITION_RE = re.compile(
    r"\b(fix(es|ed)?|waiv(e|ed|er)|re-?scope[ds]?|park(ed)?)\b", re.IGNORECASE
)
# 3k.2 step 1: the section must actually name a seam / decision.
_SEAM_NAMED_RE = re.compile(r"\b(seam|decision)\b", re.IGNORECASE)
_MIN_SEAM_SECTION_CHARS = 80

# Trip parameters. A window of 3 matches 3k.2's "over 3 consecutive pushes".
WINDOW = 3
# "Not trending to zero": the final round still carries at least this share of
# the window mean. Compared against the mean rather than the first round so one
# noisy round cannot disguise a collapse (for example 4, 100, 2).
CONVERGENCE_RATIO = 0.5
# One file must exceed this share of the window's findings to be "the same
# decision" rather than scattered review breadth.
SEAM_DOMINANCE = 0.5

_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      reviews(first:50, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          submittedAt
          author{ login }
          comments(first:100){ nodes{ path } }
        }
      }
    }
  }
}
"""

# Defensive cap on pagination so a pathological PR can never loop unbounded;
# far above any real review.
_MAX_REVIEW_PAGES = 50


@dataclass(frozen=True)
class _Submission:
    """One bot review's timestamp and paths, named so sorting never indexes."""

    submitted_at: str
    paths: list[str]


@dataclass
class ReviewRound:
    """One bot review submission and the file paths it raised findings on."""

    index: int
    submitted_at: str
    paths: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.paths)


def bot_review_rounds(
    nodes: Sequence[dict],
    bot_logins: Sequence[str],
) -> list[ReviewRound]:
    """Return one round per bot review that raised at least one finding.

    Pure, so the decision is testable without touching GitHub. A review is the
    bot's response to a push, which is what 3k.2 counts; deriving rounds from
    commits would split one push into several synthetic rounds whenever a push
    carries more than one commit, and commit dates can differ from push times.

    Reviews with no inline comments are skipped rather than recorded as empty:
    an approval or summary-only submission is not a round of findings, and
    treating it as a zero would silently break a real streak.
    """
    wanted = tuple(b.lower() for b in bot_logins)
    dated: list[_Submission] = []
    for node in nodes or []:
        author = ((node.get("author") or {}).get("login") or "").lower()
        if not any(w in author for w in wanted):
            continue
        submitted = str(node.get("submittedAt") or "")
        if not submitted:
            continue
        comments = ((node.get("comments") or {}).get("nodes")) or []
        paths = [
            str(c.get("path"))
            for c in comments
            if isinstance(c, dict) and c.get("path")
        ]
        if paths:
            dated.append(_Submission(submitted_at=submitted, paths=paths))
    # ISO-8601 UTC from the GitHub API sorts lexicographically, so this is exact
    # and needs no parsing dependency.
    dated.sort(key=lambda item: item.submitted_at)
    return [
        ReviewRound(index=i + 1, submitted_at=item.submitted_at, paths=list(item.paths))
        for i, item in enumerate(dated)
    ]


def dominant_seam(paths: Sequence[str]) -> tuple[str, int] | None:
    """Return (path, count) when one file carries a majority of the findings."""
    if not paths:
        return None
    top = next(iter(Counter(paths).most_common(1)), None)
    if top is None:
        return None
    path, hits = top
    if hits / len(paths) > SEAM_DOMINANCE:
        return path, hits
    return None


def leading_path(paths: Sequence[str]) -> str | None:
    """The single most-reported file in one round, or None on a tie or empty.

    Plurality, not majority: on a real spiral the seam leads every round but
    need not exceed half of it (ATLAS #2181 round 3 is exactly 50%). Requiring a
    majority per round would miss the case this detector exists for.
    """
    ranked = list(Counter(paths).most_common(2))
    leader = next(iter(ranked), None)
    if leader is None:
        return None
    leader_path, leader_hits = leader
    for _path, hits in ranked[1:]:
        if hits == leader_hits:
            return None
    return leader_path


def _is_strictly_decreasing(counts: Sequence[int]) -> bool:
    return all(b < a for a, b in zip(counts, counts[1:]))


def window_is_flat_or_rising(counts: Sequence[int]) -> bool:
    """3k.2's "flat or rising ... not trending to zero".

    A strictly decreasing run is converging, however slowly, so it never trips.
    Otherwise the final round must still carry at least CONVERGENCE_RATIO of the
    window mean.
    """
    if not counts:
        return False
    if _is_strictly_decreasing(counts):
        return False
    mean = sum(counts) / len(counts)
    last = next(iter(reversed(counts)), 0)
    return last >= CONVERGENCE_RATIO * mean


def find_trip(rounds: Sequence[ReviewRound]) -> tuple[int, str, list[ReviewRound]] | None:
    """First window of WINDOW consecutive rounds that is not converging.

    Returns (trip_round_index, seam_path, window_rounds), or None. The seam must
    lead every round in the window and hold a majority across it, so a window
    whose latest round has moved to a different file is not "the same decision
    re-litigated" and does not trip.
    """
    for i in range(len(rounds) - WINDOW + 1):
        window = list(rounds[i : i + WINDOW])
        if not window_is_flat_or_rising([r.count for r in window]):
            continue
        seam = dominant_seam([p for r in window for p in r.paths])
        if seam is None:
            continue
        seam_path, _hits = seam
        if any(leading_path(r.paths) != seam_path for r in window):
            continue
        last_round = next(iter(reversed(window)), None)
        if last_round is None:
            continue
        return last_round.index, seam_path, window
    return None


def body_declares_seam_analysis(body: str) -> bool:
    """True only for a real Decision-Seam Analysis section.

    Fails closed: a body that merely mentions the phrase, or promises one later,
    does not suppress the warning. Requires the heading, a section long enough
    to hold an argument, a named seam/decision, and one of 3k.2's dispositions.
    """
    text = body or ""
    match = _SEAM_HEADING_RE.search(text)
    if match is None:
        return False
    section = text[match.end():]
    next_heading = re.search(r"^\s{0,3}#{1,6}\s", section, re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]
    if len(section.strip()) < _MIN_SEAM_SECTION_CHARS:
        return False
    if _SEAM_NAMED_RE.search(section) is None:
        return False
    return _DISPOSITION_RE.search(section) is not None


def evaluate(
    rounds: Sequence[ReviewRound],
    body: str,
) -> tuple[bool, str | None, list[str]]:
    """Core decision (pure). Returns (tripped, seam_path, messages)."""
    messages: list[str] = []
    if rounds:
        messages.append("findings per bot review round:")
        for r in rounds:
            messages.append(
                "  round %d (%s)  %d  %s"
                % (r.index, r.submitted_at[:16], r.count, "#" * r.count)
            )
    trip = find_trip(rounds)
    if trip is None:
        messages.append(
            "OK: no window of %d consecutive rounds that is flat or rising on a "
            "single leading seam." % WINDOW
        )
        return False, None, messages
    trip_index, seam, window = trip
    counts = ", ".join(str(r.count) for r in window)
    if body_declares_seam_analysis(body):
        messages.append(
            "SATISFIED: non-convergence on %s at round %d (%s), but the PR body "
            "carries a Decision-Seam Analysis." % (seam, trip_index, counts)
        )
        return False, seam, messages
    messages.append(
        "AGENTS 3k.2 tripped at round %d: findings per round %s on %s are flat "
        "or rising, not trending to zero." % (trip_index, counts, seam)
    )
    return True, seam, messages


def annotation(seam: str, rounds: Sequence[ReviewRound]) -> str:
    trip = find_trip(rounds)
    counts = "?"
    if trip is not None:
        _index, _seam, window = trip
        counts = ", ".join(str(r.count) for r in window)
    return (
        "::warning file=%s::AGENTS 3k.2 convergence circuit-breaker: findings per "
        "review round (%s) on this file are flat or rising, so the threads are the "
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


def fetch_reviews(pr: int, owner: str, name: str, gh: str) -> list[dict]:
    """Fetch ALL reviews, paginating so a long review history cannot be cut."""
    nodes: list[dict] = []
    cursor: str | None = None
    for _ in range(_MAX_REVIEW_PAGES):
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
        pull = (((data.get("data") or {}).get("repository") or {}).get("pullRequest")) or {}
        block = pull.get("reviews") or {}
        nodes.extend(block.get("nodes") or [])
        page = block.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
    return nodes


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
        print(
            f"warning: could not read the PR body ({exc}); "
            "a Decision-Seam Analysis cannot be detected",
            file=sys.stderr,
        )
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
        reviews = fetch_reviews(args.pr, owner, name, args.gh)
    except RuntimeError as exc:
        print(f"error: GitHub API failure: {exc}", file=sys.stderr)
        return 2

    rounds = bot_review_rounds(reviews, bots)
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
