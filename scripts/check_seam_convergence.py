#!/usr/bin/env python3
"""Advisory detector for AGENTS.md 3k.2 (convergence circuit-breaker).

3k.2 already defines the failure: when each push closes the reported review
threads but the next push opens a comparable count of same-class findings on the
same file, the builder is instance-patching a shared decision rather than fixing
it. The rule is mandatory and names the required response (a Decision-Seam
Analysis, and a ban on "another token, regex, vocabulary row, or oracle
fixture"). It has never fired on its own, because it lives in a 73 KB document
and needs a human to notice and count rounds.

This makes it fire, and it is deliberately built as evidence rather than as
tuned thresholds -- see the Decision-Seam Analysis in
plans/PR-Seam-Convergence-Breaker.md. A round is one bot review submission. The
breaker trips only on four facts, none of which is a tunable number:

1. three consecutive rounds,
2. none of them empty (a bot review that raises nothing is convergence),
3. the same file leads all three, and
4. that file's finding count does not decrease across them.

Because the check is advisory and can never block, its error costs are
lopsided: a false alarm wastes a builder's time and burns trust, while silence
merely reproduces today's status quo. It therefore fails toward silence, and
every ambiguous case is resolved by not speaking.

Measured on live history: ATLAS #2181 trips at round 6 of 18, #2158 at round 8
of 19, #2161 at round 3 of 10; #2133, #2174, #2175 and #2199 stay silent.

It emits ``::warning`` annotations and exits 0, exactly like
scripts/check_guard_class_closure.py (the 3k.1 sibling). It is NOT a round cap:
it never blocks, never fails, and never stops a PR -- capping counts the
symptom, and on surfaces where the findings are real it would ship defects. It
only changes the class of fix the next push is allowed to make.

Exit codes: 0 = no trip, trip suppressed by a recorded Decision-Seam Analysis,
or a trip in advisory mode; 1 = trip under --strict (reserved for a future
promotion); 2 = usage error or a GitHub API failure (retryable, never a silent
pass).
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

# The recorded Decision-Seam Analysis marker. A machine token, not prose:
# judging whether a paragraph "really" analyses a seam is itself an open
# category, and parsing it would repeat the mistake this tool exists to catch.
# Mirrors the WAIVER_MARKER convention in scripts/check_guard_class_closure.py.
SEAM_MARKER_RE = re.compile(
    r"^\s*decision-seam-analysis:\s*(fix|waive|rescope)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Three consecutive rounds, per 3k.2's "over 3 consecutive pushes". This is the
# only constant in the trip decision, and it is quoted from the rule rather than
# tuned.
WINDOW = 3

_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      reviews(first:50, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          submittedAt
          author{ login }
          comments(first:100){ pageInfo{ hasNextPage } nodes{ path } }
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

    def seam_count(self, seam: str) -> int:
        return self.paths.count(seam)


def bot_review_rounds(
    nodes: Sequence[dict],
    bot_logins: Sequence[str],
) -> list[ReviewRound]:
    """Return one round per bot review, in submission order.

    Pure, so the decision is testable without touching GitHub. A review is the
    bot's response to a push, which is what 3k.2 counts; deriving rounds from
    commits would split one push into several synthetic rounds whenever a push
    carries more than one commit, and commit dates can differ from push times.

    A bot review that raises no inline findings is kept as an empty round on
    purpose: it is positive evidence that the loop converged, and dropping it
    would silently splice two non-adjacent rounds together.
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
        dated.append(_Submission(submitted_at=submitted, paths=paths))
    # ISO-8601 UTC from the GitHub API sorts lexicographically, so this is exact
    # and needs no parsing dependency.
    dated.sort(key=lambda item: item.submitted_at)
    return [
        ReviewRound(index=i + 1, submitted_at=item.submitted_at, paths=list(item.paths))
        for i, item in enumerate(dated)
    ]


def leading_path(paths: Sequence[str]) -> str | None:
    """The single most-reported file in one round, or None on a tie or empty.

    Plurality, not a share threshold: "which file did this round mostly argue
    about" is a fact about the round, whereas any percentage would be a knob.
    A tie is ambiguous, so it yields None and the window cannot trip.
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


def find_trip(rounds: Sequence[ReviewRound]) -> tuple[int, str, list[int]] | None:
    """First window of WINDOW consecutive rounds that shows non-convergence.

    Returns (trip_round_index, seam_path, seam_counts), or None. Four facts must
    all hold, and none of them is a tunable threshold:

    * every round in the window raised at least one finding (an empty bot review
      is convergence),
    * the same file leads every round,
    * that file's finding count does not decrease from the first round to the
      last -- 3k.2's "flat or rising", read literally, and measured on the seam
      rather than on unrelated findings that happen to share the round.
    """
    for i in range(len(rounds) - WINDOW + 1):
        window = list(rounds[i : i + WINDOW])
        if any(r.count == 0 for r in window):
            continue
        leaders = {leading_path(r.paths) for r in window}
        if len(leaders) != 1:
            continue
        seam = next(iter(leaders))
        if seam is None:
            continue
        counts = [r.seam_count(seam) for r in window]
        first = next(iter(counts), 0)
        last = next(iter(reversed(counts)), 0)
        if last < first:
            continue
        last_round = next(iter(reversed(window)), None)
        if last_round is None:
            continue
        return last_round.index, seam, counts
    return None


def recorded_seam_analysis(*texts: str) -> str | None:
    """The disposition from a recorded Decision-Seam Analysis marker, if any.

    Accepts the marker from any supplied source -- the PR body or the plan doc --
    because 3k.2 asks for the analysis "in the plan / PR body".
    """
    for text in texts:
        match = SEAM_MARKER_RE.search(text or "")
        if match is not None:
            return match.group(1).lower()
    return None


def evaluate(
    rounds: Sequence[ReviewRound],
    *texts: str,
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
            "OK: no window of %d consecutive rounds led by one file whose finding "
            "count does not decrease." % WINDOW
        )
        return False, None, messages
    trip_index, seam, counts = trip
    shown = ", ".join(str(c) for c in counts)
    disposition = recorded_seam_analysis(*texts)
    if disposition is not None:
        messages.append(
            "SATISFIED: non-convergence on %s at round %d (%s), and a "
            "Decision-Seam Analysis is recorded with disposition '%s'."
            % (seam, trip_index, shown, disposition)
        )
        return False, seam, messages
    messages.append(
        "AGENTS 3k.2 tripped at round %d: findings on %s across rounds %s are not "
        "decreasing." % (trip_index, seam, shown)
    )
    return True, seam, messages


def annotation(seam: str, rounds: Sequence[ReviewRound]) -> str:
    trip = find_trip(rounds)
    shown = "?"
    if trip is not None:
        _index, _seam, counts = trip
        shown = ", ".join(str(c) for c in counts)
    return (
        "::warning file=%s::AGENTS 3k.2 convergence circuit-breaker: this file led "
        "three consecutive bot review rounds and its finding count did not decrease "
        "(%s), so the threads are the same decision re-litigated. The next push may "
        "NOT add another token, regex, vocabulary row, or oracle fixture. Write a "
        "Decision-Seam Analysis in the plan or PR body -- name the one decision the "
        "threads share, state why it is wrong (over-broad, under-broad, or an open "
        "category it cannot enumerate), then do exactly one of: fix the seam "
        "structurally with a stated default direction, waive the bounded residual "
        "in Deferred, or re-scope the slice -- and record the outcome as a line "
        "reading 'decision-seam-analysis: fix' (or waive, or rescope). "
        "See AGENTS.md 3k.2 and docs/GUARD_CLASS_CLOSURE.md."
        % (seam, shown)
    )


def _gh(args: Sequence[str], gh: str) -> str:
    proc = subprocess.run([gh, *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return proc.stdout


def fetch_reviews(pr: int, owner: str, name: str, gh: str) -> list[dict]:
    """Fetch ALL reviews, paginating so a long review history cannot be cut.

    A single review carrying more than one page of inline comments raises rather
    than returning a truncated prefix: the round count and the leading file would
    both be computed from partial evidence, which could suppress a real trip or
    name the wrong seam. Exit code 2 is retryable; a silent partial read is not.
    """
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
        page_nodes = block.get("nodes") or []
        for node in page_nodes:
            comment_page = ((node.get("comments") or {}).get("pageInfo")) or {}
            if comment_page.get("hasNextPage"):
                raise RuntimeError(
                    "a review carries more than one page of inline comments; "
                    "refusing to judge convergence from a truncated prefix"
                )
        nodes.extend(page_nodes)
        page = block.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
    return nodes


def _read_text(path: Path) -> str:
    """Read a marker source, reporting rather than swallowing a read failure.

    An unreadable source means a recorded Decision-Seam Analysis cannot be seen,
    which biases toward reporting the trip. That is the safe direction, but it
    must not be silent.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not read {path} ({exc})", file=sys.stderr)
        return ""


def _pr_body(pr: int, repo: str, gh: str) -> str:
    """Prefer the CI-provided body file; fall back to the API."""
    path = os.environ.get("ATLAS_CURRENT_PR_BODY_FILE")
    if path and Path(path).exists():
        return _read_text(Path(path))
    try:
        return _gh(
            ["pr", "view", str(pr), "--repo", repo, "--json", "body", "-q", ".body"],
            gh,
        )
    except RuntimeError as exc:
        # Not silent: an unreadable body means a recorded Decision-Seam Analysis
        # cannot be seen, so the run falls toward reporting the trip. Say so.
        print(
            f"warning: could not read the PR body ({exc}); "
            "a recorded Decision-Seam Analysis cannot be detected",
            file=sys.stderr,
        )
        return ""


def _plan_texts(root: Path) -> list[str]:
    """Plan docs touched by this branch, so a marker there is honoured too."""
    plans = root / "plans"
    if not plans.is_dir():
        return []
    return [_read_text(p) for p in sorted(plans.glob("PR-*.md"))]


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
    root = Path(__file__).resolve().parent.parent
    texts = [_pr_body(args.pr, args.repo, args.gh), *_plan_texts(root)]
    tripped, seam, messages = evaluate(rounds, *texts)

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
