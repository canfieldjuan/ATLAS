#!/usr/bin/env python3
"""Live, CI-side enforcement of the AI-finding reconciliation rule (#1328 Phase 5).

The local audit (scripts/audit_ai_reconciliation.py) can only check that the PR
body's reconciliation record is internally well-formed; it cannot see the live
GitHub bot threads. This check closes that half: it fetches the real
Codex connector review threads and fails when the recorded reconciliation
*omits a genuinely open finding* -- i.e. the body claims all-clear (or carries
no reconciliation record at all) while unresolved bot threads still exist.

It deliberately does NOT require every thread to be GitHub-"resolved": that is
self-resolvable by the PR author and would be a gameable rigor gate. It catches
the specific contradiction between a "resolved" body and open reality, which is
exactly the deferred spec from plans/archive/PR-Reviewer-Reconciliation-Audit.md.

Codex findings are review-gate inputs, not auto-applied commands: nothing here
auto-resolves or auto-applies. It only enforces that the PR body accounts for
what Codex raised.

Exit codes: 0 = clean (no open bot threads, or the body honestly acknowledges
open findings); 1 = contradiction (open bot threads + an all-clear/absent
record); 2 = usage error or a GitHub API failure (retryable, never a silent
pass).

The body classifier reuses scripts/audit_ai_reconciliation.py by import so the
local and live checks cannot disagree on what a "resolved" record looks like.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_DEFAULT_BOTS = ("chatgpt-codex-connector", "chatgpt-codex-connector[bot]")
_CLEAN_CODEX_REVIEW_TEXT = "didn't find any major issues"
_REVIEWED_COMMIT_RE = re.compile(r"\*\*Reviewed commit:\*\*\s*`(?P<sha>[0-9a-f]{10,40})`", re.IGNORECASE)
_LEGACY_BOT_ALIASES = frozenset(
    {
        "bot",
        "chatgpt",
        "chatgpt-codex",
        "claude",
        "codex",
        "copilot",
        "copilot-pull-request-reviewer",
    }
)

_THREADS_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      reviewThreads(first:100, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          isResolved
          isOutdated
          path
          line
          comments(first:1){ nodes{ author{ login } bodyText } }
        }
      }
    }
  }
}
"""

_REVIEWS_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      headRefOid
      reviews(first:100, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          author{ login }
          commit{ oid }
          state
        }
      }
    }
  }
}
"""

_COMMENTS_QUERY = """
query($owner:String!,$name:String!,$pr:Int!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequest(number:$pr){
      headRefOid
      comments(first:100, after:$cursor){
        pageInfo{ hasNextPage endCursor }
        nodes{
          author{ login }
          body
          bodyText
        }
      }
    }
  }
}
"""

# Defensive cap on pagination (100 threads/page) so a pathological PR can never
# loop unbounded; far above any real review.
_MAX_THREAD_PAGES = 50
_MAX_REVIEW_PAGES = 50
_MAX_COMMENT_PAGES = 50


def _load_phase2():
    """Import the local reconciliation auditor so the body classifier matches."""
    path = Path(__file__).resolve().parent / "audit_ai_reconciliation.py"
    spec = importlib.util.spec_from_file_location("audit_ai_reconciliation", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_body(body: str) -> str:
    """Return 'absent' | 'acknowledges_open' | 'claims_clear' | 'unmarked'.

    Uses the Phase-2 section extractor + markers, so "what counts as a resolved
    record" stays defined in exactly one place.
    """
    p2 = _load_phase2()
    section = p2.extract_section(body)
    if section is None:
        return "absent"
    if p2.UNRESOLVED_RE.search(section):
        return "acknowledges_open"
    if p2.RESOLVED_RE.search(section):
        return "claims_clear"
    return "unmarked"


def open_bot_threads(nodes: Sequence[dict], bot_logins: Sequence[str]) -> list[dict]:
    """Return unresolved review threads authored by a known bot.

    `nodes` is the GraphQL reviewThreads node list; pure so it is unit-testable
    without touching GitHub.
    """
    wanted = frozenset(b.lower() for b in bot_logins)
    found: list[dict] = []
    for node in nodes or []:
        if node.get("isResolved"):
            continue
        comments = ((node.get("comments") or {}).get("nodes")) or []
        author = ""
        snippet = ""
        if comments:
            author = (((comments[0] or {}).get("author") or {}).get("login")) or ""
            snippet = ((comments[0] or {}).get("bodyText") or "").strip().replace("\n", " ")
        if author.lower() not in wanted:
            continue
        found.append(
            {
                "path": node.get("path") or "?",
                "line": node.get("line"),
                "author": author or "?",
                "snippet": (snippet[:120] + "...") if len(snippet) > 120 else snippet,
            }
        )
    return found


def _current_head_bot_reviews_with_states(
    reviews: Sequence[dict],
    *,
    head_sha: str,
    bot_logins: Sequence[str],
    states: frozenset[str],
) -> list[dict]:
    wanted = frozenset(b.lower() for b in bot_logins)
    found: list[dict] = []
    for review in reviews or []:
        author = (((review.get("author") or {}).get("login")) or "").lower()
        commit = ((review.get("commit") or {}).get("oid")) or ""
        state = review.get("state") or ""
        if author not in wanted or commit != head_sha:
            continue
        if state not in states:
            continue
        found.append(review)
    return found


def current_head_bot_reviews(
    reviews: Sequence[dict],
    *,
    head_sha: str,
    bot_logins: Sequence[str],
) -> list[dict]:
    """Return satisfactory Codex connector reviews attached to the current PR head."""

    return _current_head_bot_reviews_with_states(
        reviews,
        head_sha=head_sha,
        bot_logins=bot_logins,
        states=frozenset({"COMMENTED", "APPROVED"}),
    )


def current_head_clean_review_comments(
    comments: Sequence[dict],
    *,
    head_sha: str,
    bot_logins: Sequence[str],
) -> list[dict]:
    """Return Codex clean-review PR comments that name the current PR head."""

    wanted = frozenset(b.lower() for b in bot_logins)
    found: list[dict] = []
    for comment in comments or []:
        author = (((comment.get("author") or {}).get("login")) or "").lower()
        body = comment.get("body") or comment.get("bodyText") or ""
        if author not in wanted or _CLEAN_CODEX_REVIEW_TEXT not in body.lower():
            continue
        match = _REVIEWED_COMMIT_RE.search(body)
        if match is None:
            continue
        reviewed_sha = match.group("sha").lower()
        if not head_sha.lower().startswith(reviewed_sha):
            continue
        found.append(comment)
    return found


def current_head_change_requests(
    reviews: Sequence[dict],
    *,
    head_sha: str,
    bot_logins: Sequence[str],
) -> list[dict]:
    """Return current-head Codex connector reviews that request changes."""

    return _current_head_bot_reviews_with_states(
        reviews,
        head_sha=head_sha,
        bot_logins=bot_logins,
        states=frozenset({"CHANGES_REQUESTED"}),
    )


def review_attestation_generation(
    reviews: Sequence[dict],
    comments: Sequence[dict] | None = None,
    *,
    head_sha: str,
    bot_logins: Sequence[str],
) -> tuple[tuple[str, str, str], ...]:
    """Return the current-head Codex review generation used to prove freshness."""

    wanted = frozenset(b.lower() for b in bot_logins)
    generation = []
    for review in reviews or []:
        author = (((review.get("author") or {}).get("login")) or "").lower()
        commit = ((review.get("commit") or {}).get("oid")) or ""
        state = review.get("state") or ""
        if author in wanted and commit == head_sha:
            generation.append(("review", author, commit, state))
    for comment in comments or []:
        author = (((comment.get("author") or {}).get("login")) or "").lower()
        body = comment.get("body") or comment.get("bodyText") or ""
        if author not in wanted:
            continue
        match = _REVIEWED_COMMIT_RE.search(body)
        if match is None:
            continue
        reviewed_sha = match.group("sha").lower()
        if head_sha.lower().startswith(reviewed_sha):
            generation.append(("comment", author, reviewed_sha, body))
    return tuple(sorted(generation))


def _generation_value(value: object) -> tuple[int, str]:
    """Return a consistently comparable representation for GraphQL nullable fields."""

    if value is None:
        return (0, "")
    return (1, str(value))


def review_thread_generation(nodes: Sequence[dict]) -> tuple[tuple[object, ...], ...]:
    """Return a comparable thread generation for consistency checks."""

    generation = []
    for node in nodes or []:
        comments = ((node.get("comments") or {}).get("nodes")) or []
        author = ""
        body = ""
        if comments:
            author = (((comments[0] or {}).get("author") or {}).get("login")) or ""
            body = ((comments[0] or {}).get("bodyText")) or ""
        generation.append(
            (
                _generation_value(node.get("isResolved")),
                _generation_value(node.get("isOutdated")),
                _generation_value(node.get("path")),
                _generation_value(node.get("line")),
                _generation_value(author),
                _generation_value(body),
            )
        )
    return tuple(sorted(generation))


def parse_bot_logins(raw: str | None) -> list[str]:
    """Parse exact GitHub bot logins and reject legacy substring aliases."""

    bots = [b.strip() for b in (raw or "").split(",") if b.strip()]
    if not bots:
        raise ValueError("no bot logins configured")
    invalid = [
        bot
        for bot in bots
        if bot.lower() in _LEGACY_BOT_ALIASES or "*" in bot or any(ch.isspace() for ch in bot)
    ]
    if invalid:
        raise ValueError(
            "bot identities must be exact GitHub logins, not legacy aliases or patterns: "
            + ", ".join(invalid)
        )
    return bots


def evaluate(
    nodes: Sequence[dict],
    body: str,
    bot_logins: Sequence[str],
    *,
    reviews: Sequence[dict] | None = None,
    comments: Sequence[dict] | None = None,
    head_sha: str | None = None,
) -> tuple[int, list[str]]:
    """Core decision (pure). Returns (exit_code, messages)."""
    messages: list[str] = []
    if head_sha is not None:
        change_requests = current_head_change_requests(
            reviews or [],
            head_sha=head_sha,
            bot_logins=bot_logins,
        )
        if change_requests:
            messages.append(
                "current-head Codex connector review requested changes: live "
                f"reconciliation cannot pass PR head {head_sha} until the "
                "changes-requested review is superseded or resolved."
            )
        elif not current_head_bot_reviews(
            reviews or [],
            head_sha=head_sha,
            bot_logins=bot_logins,
        ) and not current_head_clean_review_comments(
            comments or [],
            head_sha=head_sha,
            bot_logins=bot_logins,
        ):
            messages.append(
                "missing current-head Codex connector review: live reconciliation "
                f"requires one scoped Codex review or clean review comment on PR head {head_sha} before merge."
            )

    open_threads = open_bot_threads(nodes, bot_logins)
    if not open_threads:
        if messages:
            return 1, messages
        return 0, ["OK: current-head Codex review attestation is present and no open scoped Codex review threads remain."]

    body_class = classify_body(body)
    if body_class == "claims_clear":
        lead = (
            "reconciliation contradicts reality: the PR body records the "
            "automated-review findings as all fixed/waived, but these bot threads "
            "are still open:"
        )
    elif body_class == "acknowledges_open":
        lead = (
            "AI reconciliation acknowledges open findings, and these scoped Codex "
            "threads are still open:"
        )
    elif body_class == "unmarked":
        lead = (
            "AI reconciliation record is present but does not mark findings fixed "
            "or waived, and these scoped Codex threads are still open:"
        )
    else:  # absent
        lead = (
            "no AI reconciliation record found, but these automated-review (bot) "
            "threads are still open and unaccounted for:"
        )
    messages.append(lead)
    for t in open_threads:
        loc = t["path"] if t["line"] is None else f"{t['path']}:{t['line']}"
        messages.append(f"  - [{t['author']}] {loc}: {t['snippet']}")
    messages.append(
        "Resolve or explicitly waive (with a reason in the PR body) each finding "
        "before merge (AGENTS.md 4a.1)."
    )
    return 1, messages


def _gh(args: Sequence[str], gh: str) -> str:
    proc = subprocess.run(
        [gh, *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return proc.stdout


def _expect_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"GitHub GraphQL response malformed: {label} is missing or not an object")
    return value


def fetch_threads(pr: int, owner: str, name: str, gh: str) -> list[dict]:
    """Fetch ALL review threads, paginating so a PR with >100 threads cannot

    hide an unresolved finding past the first page and pass as clear.
    """
    nodes: list[dict] = []
    cursor: str | None = None
    for page_number in range(1, _MAX_THREAD_PAGES + 1):
        args = [
            "api", "graphql",
            "-f", f"query={_THREADS_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"name={name}",
            "-F", f"pr={pr}",
        ]
        if cursor:
            args += ["-F", f"cursor={cursor}"]
        data = json.loads(_gh(args, gh))
        if data.get("errors"):
            raise RuntimeError(f"GitHub GraphQL returned errors on reviewThreads page {page_number}")

        envelope = _expect_mapping(data.get("data"), "data")
        repository = _expect_mapping(envelope.get("repository"), "repository")
        pull_request = _expect_mapping(repository.get("pullRequest"), "pullRequest")
        threads = _expect_mapping(pull_request.get("reviewThreads"), "reviewThreads")
        page = _expect_mapping(threads.get("pageInfo"), "reviewThreads.pageInfo")

        page_nodes = threads.get("nodes")
        if not isinstance(page_nodes, list):
            raise RuntimeError("GitHub GraphQL response malformed: reviewThreads.nodes is missing or not a list")
        nodes.extend(page_nodes)

        has_next = page.get("hasNextPage")
        if not isinstance(has_next, bool):
            raise RuntimeError(
                "GitHub GraphQL response malformed: reviewThreads.pageInfo.hasNextPage is missing or not a bool"
            )
        if not has_next:
            break
        next_cursor = page.get("endCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            raise RuntimeError(
                "GitHub GraphQL response malformed: reviewThreads.pageInfo.endCursor is required for pagination"
            )
        cursor = next_cursor
    else:
        raise RuntimeError(f"reviewThreads pagination exceeded {_MAX_THREAD_PAGES} pages")
    return nodes


def fetch_review_attestation(pr: int, owner: str, name: str, gh: str) -> tuple[str, list[dict]]:
    """Fetch PR head SHA and all review records for current-head Codex attestation."""

    reviews: list[dict] = []
    head_sha = ""
    cursor: str | None = None
    for page_number in range(1, _MAX_REVIEW_PAGES + 1):
        args = [
            "api", "graphql",
            "-f", f"query={_REVIEWS_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"name={name}",
            "-F", f"pr={pr}",
        ]
        if cursor:
            args += ["-F", f"cursor={cursor}"]
        data = json.loads(_gh(args, gh))
        if data.get("errors"):
            raise RuntimeError(f"GitHub GraphQL returned errors on reviews page {page_number}")

        envelope = _expect_mapping(data.get("data"), "data")
        repository = _expect_mapping(envelope.get("repository"), "repository")
        pull_request = _expect_mapping(repository.get("pullRequest"), "pullRequest")
        observed_head = pull_request.get("headRefOid")
        if not isinstance(observed_head, str) or not observed_head:
            raise RuntimeError("GitHub GraphQL response malformed: pullRequest.headRefOid is missing")
        if head_sha and observed_head != head_sha:
            raise RuntimeError("GitHub GraphQL response changed PR head during review pagination")
        head_sha = observed_head

        review_connection = _expect_mapping(pull_request.get("reviews"), "reviews")
        page = _expect_mapping(review_connection.get("pageInfo"), "reviews.pageInfo")
        page_nodes = review_connection.get("nodes")
        if not isinstance(page_nodes, list):
            raise RuntimeError("GitHub GraphQL response malformed: reviews.nodes is missing or not a list")
        reviews.extend(page_nodes)

        has_next = page.get("hasNextPage")
        if not isinstance(has_next, bool):
            raise RuntimeError(
                "GitHub GraphQL response malformed: reviews.pageInfo.hasNextPage is missing or not a bool"
            )
        if not has_next:
            break
        next_cursor = page.get("endCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            raise RuntimeError("GitHub GraphQL response malformed: reviews.pageInfo.endCursor is required")
        cursor = next_cursor
    else:
        raise RuntimeError(f"reviews pagination exceeded {_MAX_REVIEW_PAGES} pages")
    return head_sha, reviews


def fetch_comment_attestation(pr: int, owner: str, name: str, gh: str) -> tuple[str, list[dict]]:
    """Fetch PR head SHA and all PR comments for clean Codex attestation."""

    comments: list[dict] = []
    head_sha = ""
    cursor: str | None = None
    for page_number in range(1, _MAX_COMMENT_PAGES + 1):
        args = [
            "api", "graphql",
            "-f", f"query={_COMMENTS_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"name={name}",
            "-F", f"pr={pr}",
        ]
        if cursor:
            args += ["-F", f"cursor={cursor}"]
        data = json.loads(_gh(args, gh))
        if data.get("errors"):
            raise RuntimeError(f"GitHub GraphQL returned errors on comments page {page_number}")

        envelope = _expect_mapping(data.get("data"), "data")
        repository = _expect_mapping(envelope.get("repository"), "repository")
        pull_request = _expect_mapping(repository.get("pullRequest"), "pullRequest")
        observed_head = pull_request.get("headRefOid")
        if not isinstance(observed_head, str) or not observed_head:
            raise RuntimeError("GitHub GraphQL response malformed: pullRequest.headRefOid is missing")
        if head_sha and observed_head != head_sha:
            raise RuntimeError("GitHub GraphQL response changed PR head during comment pagination")
        head_sha = observed_head

        comment_connection = _expect_mapping(pull_request.get("comments"), "comments")
        page = _expect_mapping(comment_connection.get("pageInfo"), "comments.pageInfo")
        page_nodes = comment_connection.get("nodes")
        if not isinstance(page_nodes, list):
            raise RuntimeError("GitHub GraphQL response malformed: comments.nodes is missing or not a list")
        comments.extend(page_nodes)

        has_next = page.get("hasNextPage")
        if not isinstance(has_next, bool):
            raise RuntimeError(
                "GitHub GraphQL response malformed: comments.pageInfo.hasNextPage is missing or not a bool"
            )
        if not has_next:
            break
        next_cursor = page.get("endCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            raise RuntimeError("GitHub GraphQL response malformed: comments.pageInfo.endCursor is required")
        cursor = next_cursor
    else:
        raise RuntimeError(f"comments pagination exceeded {_MAX_COMMENT_PAGES} pages")
    return head_sha, comments


def fetch_body(pr: int, repo: str, gh: str) -> str:
    out = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "body", "-q", ".body"], gh)
    return out


def fetch_consistent_review_thread_snapshot(
    pr: int,
    owner: str,
    name: str,
    gh: str,
    bot_logins: Sequence[str],
) -> tuple[list[dict], str, list[dict], list[dict]]:
    """Fetch reviews/threads twice and fail closed if either generation moves."""

    head_before, reviews_before = fetch_review_attestation(pr, owner, name, gh)
    comment_head_before, comments_before = fetch_comment_attestation(pr, owner, name, gh)
    nodes_before = fetch_threads(pr, owner, name, gh)
    head_middle, reviews_middle = fetch_review_attestation(pr, owner, name, gh)
    comment_head_middle, comments_middle = fetch_comment_attestation(pr, owner, name, gh)
    nodes_after = fetch_threads(pr, owner, name, gh)
    head_after, reviews_after = fetch_review_attestation(pr, owner, name, gh)
    comment_head_after, comments_after = fetch_comment_attestation(pr, owner, name, gh)
    if len({head_before, head_middle, head_after, comment_head_before, comment_head_middle, comment_head_after}) != 1:
        raise RuntimeError("GitHub PR head changed during review/thread snapshot fetch")
    before_generation = review_attestation_generation(
        reviews_before,
        comments_before,
        head_sha=head_before,
        bot_logins=bot_logins,
    )
    after_generation = review_attestation_generation(
        reviews_after,
        comments_after,
        head_sha=head_after,
        bot_logins=bot_logins,
    )
    middle_generation = review_attestation_generation(
        reviews_middle,
        comments_middle,
        head_sha=head_middle,
        bot_logins=bot_logins,
    )
    if before_generation != middle_generation or middle_generation != after_generation:
        raise RuntimeError("GitHub Codex review generation changed during review/thread snapshot fetch")
    if review_thread_generation(nodes_before) != review_thread_generation(nodes_after):
        raise RuntimeError("GitHub review thread generation changed during review/thread snapshot fetch")
    return nodes_after, head_after, reviews_after, comments_after


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, help="PR number")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name (defaults to $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--bots",
        default=os.environ.get("ATLAS_REVIEW_BOTS", ",".join(_DEFAULT_BOTS)),
        help="comma-separated exact bot logins (default: Codex connector)",
    )
    parser.add_argument("--gh", default="gh", help="path to the gh CLI")
    parser.add_argument(
        "--threads-file",
        help="JSON file of reviewThreads nodes (test/dry-run; skips the live fetch)",
    )
    parser.add_argument(
        "--body-file",
        help="PR body file (test/dry-run; skips fetching the live body)",
    )
    parser.add_argument(
        "--reviews-file",
        help="JSON file of review nodes (test/dry-run; skips fetching live reviews)",
    )
    parser.add_argument(
        "--comments-file",
        help="JSON file of PR comment nodes (test/dry-run; skips fetching live review comments)",
    )
    parser.add_argument(
        "--head-sha",
        help="PR head SHA for current-head Codex review attestation in test/dry-run mode",
    )
    args = parser.parse_args(argv)

    try:
        bots = parse_bot_logins(args.bots)
    except ValueError as exc:
        print(f"live reconciliation: {exc}", file=sys.stderr)
        return 2

    try:
        head_sha = args.head_sha
        reviews: Sequence[dict] | None = None
        comments: Sequence[dict] | None = None

        if args.threads_file:
            nodes = json.loads(Path(args.threads_file).read_text(encoding="utf-8"))
        else:
            if args.pr is None or not args.repo:
                print(
                    "live reconciliation: need --pr and --repo (or $GITHUB_REPOSITORY) "
                    "when not using --threads-file",
                    file=sys.stderr,
                )
                return 2
            owner, _, name = args.repo.partition("/")
            nodes, head_sha, reviews, comments = fetch_consistent_review_thread_snapshot(
                args.pr,
                owner,
                name,
                args.gh,
                bots,
            )

        if args.reviews_file:
            reviews = json.loads(Path(args.reviews_file).read_text(encoding="utf-8"))
        if args.comments_file:
            comments = json.loads(Path(args.comments_file).read_text(encoding="utf-8"))

        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
        elif args.pr is not None and args.repo:
            body = fetch_body(args.pr, args.repo, args.gh)
        else:
            body = ""
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"live reconciliation: GitHub API/read error: {exc}", file=sys.stderr)
        return 2

    code, messages = evaluate(nodes, body, bots, reviews=reviews, comments=comments, head_sha=head_sha)
    print("live AI reconciliation check")
    print("-" * 60)
    for line in messages:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
