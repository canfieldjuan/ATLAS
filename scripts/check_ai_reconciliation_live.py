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
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

_DEFAULT_BOTS = ("chatgpt-codex-connector", "chatgpt-codex-connector[bot]")
_CLEAN_CODEX_REVIEW_TEXT = "didn't find any major issues"
_DEFAULT_CODEX_REVIEW_GRACE_SECONDS = 300
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


class ChangedFileProof:
    def __init__(
        self,
        *,
        base_sha: str,
        head_sha: str,
        merge_base_sha: str,
        expected_count: int,
        files: list[dict],
    ) -> None:
        self.base_sha = base_sha
        self.head_sha = head_sha
        self.merge_base_sha = merge_base_sha
        self.expected_count = expected_count
        self.files = files


class PrRefSnapshot:
    def __init__(
        self,
        *,
        base_ref_name: str,
        base_sha: str,
        head_sha: str,
        changed_files: int,
    ) -> None:
        self.base_ref_name = base_ref_name
        self.base_sha = base_sha
        self.head_sha = head_sha
        self.changed_files = changed_files


def _load_phase2():
    """Import the local reconciliation auditor so the body classifier matches."""
    path = Path(__file__).resolve().parent / "audit_ai_reconciliation.py"
    spec = importlib.util.spec_from_file_location("audit_ai_reconciliation", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pr_body_audit():
    """Import the PR-body auditor so docs-only marker parsing is canonical."""
    path = Path(__file__).resolve().parent / "audit_pr_body.py"
    spec = importlib.util.spec_from_file_location("audit_pr_body", path)
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


def is_docs_only_body(body: str) -> bool:
    """Return true when the PR body uses the explicit docs-only exemption."""

    return bool(_load_pr_body_audit().is_docs_only_body(body))


def _is_markdown_only_path(path: str) -> bool:
    """Return true when a changed path has `.md` as its only suffix."""

    return PurePosixPath(path).suffixes == [".md"]


def _is_non_executable_regular_blob(item: dict, prefix: str) -> bool:
    return item.get(f"{prefix}_mode") == "100644" and item.get(f"{prefix}_type") == "blob"


_HEAD_SIDE_STATUSES = frozenset({"added", "changed", "copied", "modified", "renamed"})
_BASE_SIDE_STATUSES = frozenset({"removed", "renamed"})
_ALLOWED_CHANGED_FILE_STATUSES = _HEAD_SIDE_STATUSES | _BASE_SIDE_STATUSES


def changed_file_shape_is_valid(item: dict) -> bool:
    """Return true when a GitHub changed-file row has the fields its status needs."""

    filename = item.get("filename")
    status = item.get("status")
    if not isinstance(filename, str) or not filename:
        return False
    if not isinstance(status, str) or status not in _ALLOWED_CHANGED_FILE_STATUSES:
        return False
    previous_filename = item.get("previous_filename")
    if status == "renamed":
        return isinstance(previous_filename, str) and bool(previous_filename)
    return previous_filename is None or isinstance(previous_filename, str)


def changed_files_are_docs_only(files: Sequence[dict]) -> bool:
    """Return true only when the live changed-file list proves a docs-only diff."""

    if not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        if not changed_file_shape_is_valid(item):
            return False
        filename = item.get("filename")
        if not isinstance(filename, str) or not _is_markdown_only_path(filename):
            return False
        previous_filename = item.get("previous_filename")
        if previous_filename is not None and (
            not isinstance(previous_filename, str) or not _is_markdown_only_path(previous_filename)
        ):
            return False
        status = item.get("status")
        if status == "removed":
            if not _is_non_executable_regular_blob(item, "base"):
                return False
        elif status in _HEAD_SIDE_STATUSES:
            if not _is_non_executable_regular_blob(item, "head"):
                return False
        if previous_filename is not None and not _is_non_executable_regular_blob(item, "base"):
            return False
    return True


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


def _parse_github_timestamp(raw: str) -> datetime:
    """Parse a GitHub API timestamp into an aware UTC datetime."""

    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def updated_within_review_grace(
    updated_at: str | None,
    *,
    review_grace_seconds: int,
    now: datetime | None = None,
) -> bool:
    """Return true while a fresh PR update is still inside the Codex review window."""

    return review_grace_remaining_seconds(
        updated_at,
        review_grace_seconds=review_grace_seconds,
        now=now,
    ) > 0


def review_grace_remaining_seconds(
    updated_at: str | None,
    *,
    review_grace_seconds: int,
    now: datetime | None = None,
) -> float:
    """Return seconds left in the fresh-head Codex review window."""

    if review_grace_seconds <= 0 or not updated_at:
        return 0.0
    reference_time = now or datetime.now(UTC)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)
    updated_time = _parse_github_timestamp(updated_at)
    expires_at = updated_time + timedelta(seconds=review_grace_seconds)
    return max(0.0, (expires_at - reference_time.astimezone(UTC)).total_seconds())


def missing_codex_activity_inside_review_grace(
    nodes: Sequence[dict],
    bot_logins: Sequence[str],
    *,
    reviews: Sequence[dict] | None,
    comments: Sequence[dict] | None,
    head_sha: str | None,
    pr_updated_at: str | None,
    review_grace_seconds: int,
    now: datetime | None = None,
) -> bool:
    """Return true only for the quiet fresh-head race the required job must wait out."""

    if head_sha is None or open_bot_threads(nodes, bot_logins):
        return False
    if current_head_change_requests(reviews or [], head_sha=head_sha, bot_logins=bot_logins):
        return False
    if current_head_bot_reviews(reviews or [], head_sha=head_sha, bot_logins=bot_logins):
        return False
    if current_head_clean_review_comments(comments or [], head_sha=head_sha, bot_logins=bot_logins):
        return False
    return updated_within_review_grace(
        pr_updated_at,
        review_grace_seconds=review_grace_seconds,
        now=now,
    )


def parse_review_grace_seconds(raw: str | int | None) -> int:
    """Parse the review-window duration from CLI/env without argparse tracebacks."""

    if raw is None:
        return _DEFAULT_CODEX_REVIEW_GRACE_SECONDS
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("review grace seconds must be an integer") from exc
    if value < 0:
        raise ValueError("review grace seconds must be non-negative")
    return value


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
    changed_files: Sequence[dict] | None = None,
    head_sha: str | None = None,
    pr_updated_at: str | None = None,
    review_grace_seconds: int = 0,
    now: datetime | None = None,
) -> tuple[int, list[str]]:
    """Core decision (pure). Returns (exit_code, messages)."""
    messages: list[str] = []
    open_threads = open_bot_threads(nodes, bot_logins)
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
        elif not open_threads and not current_head_bot_reviews(
            reviews or [],
            head_sha=head_sha,
            bot_logins=bot_logins,
        ) and not current_head_clean_review_comments(
            comments or [],
            head_sha=head_sha,
            bot_logins=bot_logins,
        ) and updated_within_review_grace(
            pr_updated_at,
            review_grace_seconds=review_grace_seconds,
            now=now,
        ):
            messages.append(
                "waiting for Codex connector review window: no scoped Codex "
                f"activity is recorded on PR head {head_sha} yet, and the PR "
                f"was updated less than {review_grace_seconds} seconds ago."
            )
        elif not open_threads and is_docs_only_body(body) and changed_files_are_docs_only(changed_files or []):
            messages.append(
                "OK: docs-only PR diff has no open scoped Codex review threads; "
                "current-head Codex review attestation is not required."
            )

    if not open_threads:
        if messages:
            if all(message.startswith("OK:") for message in messages):
                return 0, messages
            return 1, messages
        return 0, ["OK: no open scoped Codex review threads remain."]

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


def docs_only_exemption_needs_file_proof(
    nodes: Sequence[dict],
    body: str,
    bot_logins: Sequence[str],
    *,
    reviews: Sequence[dict] | None,
    comments: Sequence[dict] | None,
    head_sha: str | None,
) -> bool:
    """Return true only when file proof is needed to emit the docs-only watcher signal."""

    if head_sha is None or not is_docs_only_body(body):
        return False
    if open_bot_threads(nodes, bot_logins):
        return False
    if current_head_change_requests(reviews or [], head_sha=head_sha, bot_logins=bot_logins):
        return False
    if current_head_bot_reviews(reviews or [], head_sha=head_sha, bot_logins=bot_logins):
        return False
    if current_head_clean_review_comments(comments or [], head_sha=head_sha, bot_logins=bot_logins):
        return False
    return True


def _gh(args: Sequence[str], gh: str) -> str:
    proc = subprocess.run(
        [gh, *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip())
    return proc.stdout


def _git_stdout(args: Sequence[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        raise RuntimeError(detail)
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


def fetch_pr_updated_at(pr: int, repo: str, gh: str) -> str:
    out = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "updatedAt", "-q", ".updatedAt"], gh)
    updated_at = out.strip()
    _parse_github_timestamp(updated_at)
    return updated_at


def fetch_pr_ref_snapshot(pr: int, repo: str, gh: str) -> PrRefSnapshot:
    out = _gh(
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "baseRefName,baseRefOid,changedFiles,headRefOid",
        ],
        gh,
    )
    data = json.loads(out)
    base_ref_name = data.get("baseRefName")
    base_sha = data.get("baseRefOid")
    head_sha = data.get("headRefOid")
    changed_files = data.get("changedFiles")
    if not isinstance(base_ref_name, str) or not base_ref_name:
        raise RuntimeError("GitHub PR response malformed: baseRefName is missing")
    if base_ref_name.startswith("-") or ".." in base_ref_name:
        raise RuntimeError("GitHub PR response malformed: baseRefName is unsafe")
    if not isinstance(base_sha, str) or not base_sha:
        raise RuntimeError("GitHub PR response malformed: baseRefOid is missing")
    if not isinstance(head_sha, str) or not head_sha:
        raise RuntimeError("GitHub PR response malformed: headRefOid is missing")
    if not isinstance(changed_files, int) or changed_files < 0:
        raise RuntimeError("GitHub PR response malformed: changedFiles is missing")
    return PrRefSnapshot(
        base_ref_name=base_ref_name,
        base_sha=base_sha,
        head_sha=head_sha,
        changed_files=changed_files,
    )


def fetch_pr_refs(pr: int, repo: str, gh: str) -> tuple[str, str, int]:
    snapshot = fetch_pr_ref_snapshot(pr, repo, gh)
    return snapshot.base_sha, snapshot.head_sha, snapshot.changed_files


def fetch_merge_base(repo: str, base_sha: str, head_sha: str, gh: str) -> str:
    out = _gh(["api", f"repos/{repo}/compare/{base_sha}...{head_sha}", "--jq", ".merge_base_commit.sha"], gh)
    merge_base = out.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", merge_base):
        raise RuntimeError("GitHub compare response malformed: merge_base_commit.sha is missing")
    return merge_base


def fetch_tree_entries(repo: str, ref: str, gh: str) -> dict[str, dict]:
    out = _gh(["api", f"repos/{repo}/git/trees/{ref}?recursive=1"], gh)
    data = json.loads(out)
    if data.get("truncated"):
        raise RuntimeError(f"GitHub tree response truncated for {ref}")
    tree = data.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError("GitHub tree response malformed: tree is missing or not a list")
    entries: dict[str, dict] = {}
    for entry in tree:
        if not isinstance(entry, dict):
            raise RuntimeError("GitHub tree response malformed: tree entry is not an object")
        path = entry.get("path")
        if isinstance(path, str):
            entries[path] = entry
    return entries


def _fetch_pr_git_refs(pr: int, base_ref_name: str) -> None:
    _git_stdout(
        [
            "fetch",
            "--no-tags",
            "origin",
            f"+refs/heads/{base_ref_name}:refs/remotes/origin/{base_ref_name}",
            f"+pull/{pr}/head:refs/remotes/origin/pr-{pr}",
        ]
    )


def _assert_commit_available(ref: str, label: str) -> None:
    _git_stdout(["cat-file", "-e", f"{ref}^{{commit}}"])


def _git_tree_entry(ref: str, path: str) -> dict[str, str]:
    entry = _git_stdout(["ls-tree", ref, "--", path]).strip()
    if not entry:
        return {}
    parts = entry.split(None, 3)
    if len(parts) < 4:
        raise RuntimeError(f"git ls-tree response malformed for {path} at {ref}")
    return {"mode": parts[0], "type": parts[1]}


def _attach_tree_entry(item: dict, *, prefix: str, path: str, entries: dict[str, dict]) -> None:
    entry = entries.get(path)
    if not isinstance(entry, dict):
        return
    item[f"{prefix}_mode"] = entry.get("mode")
    item[f"{prefix}_type"] = entry.get("type")


def _attach_git_tree_entry(item: dict, *, prefix: str, path: str, ref: str) -> None:
    entry = _git_tree_entry(ref, path)
    if not entry:
        return
    item[f"{prefix}_mode"] = entry.get("mode")
    item[f"{prefix}_type"] = entry.get("type")


def _local_changed_files_from_refs(merge_base: str, head_sha: str) -> list[dict]:
    payload = _git_stdout(
        [
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            f"{merge_base}...{head_sha}",
        ]
    )
    parts = [part for part in payload.split("\0") if part]
    if len(parts) % 2 != 0:
        raise RuntimeError("git diff --name-status response malformed")
    files: list[dict] = []
    status_map = {
        "A": "added",
        "D": "removed",
        "M": "modified",
    }
    for index in range(0, len(parts), 2):
        raw_status = parts[index]
        path = parts[index + 1]
        status = status_map.get(raw_status)
        if status is None:
            status = f"unsupported:{raw_status}"
        item = {"filename": path, "status": status, "previous_filename": None}
        if status == "removed":
            _attach_git_tree_entry(item, prefix="base", path=path, ref=merge_base)
        elif status in _HEAD_SIDE_STATUSES:
            _attach_git_tree_entry(item, prefix="head", path=path, ref=head_sha)
        files.append(item)
    return files


def fetch_changed_file_proof(
    pr: int,
    repo: str,
    gh: str,
    *,
    head_sha: str | None = None,
    base_sha: str | None = None,
) -> ChangedFileProof:
    """Derive PR changed files from immutable git refs, not the mutable PR files API."""

    snapshot = fetch_pr_ref_snapshot(pr, repo, gh)
    if base_sha is not None and snapshot.base_sha != base_sha:
        raise RuntimeError("GitHub PR base changed before changed-file fetch")
    if head_sha is not None and snapshot.head_sha != head_sha:
        raise RuntimeError("GitHub PR head changed before changed-file fetch")
    merge_base = fetch_merge_base(repo, snapshot.base_sha, snapshot.head_sha, gh)
    _fetch_pr_git_refs(pr, snapshot.base_ref_name)
    for ref, label in (
        (snapshot.base_sha, "base"),
        (snapshot.head_sha, "head"),
        (merge_base, "merge base"),
    ):
        try:
            _assert_commit_available(ref, label)
        except RuntimeError as exc:
            raise RuntimeError(f"{label} commit {ref} is unavailable after git fetch") from exc
    fetched_head = _git_stdout(["rev-parse", "--verify", f"refs/remotes/origin/pr-{pr}^{{commit}}"]).strip()
    if fetched_head != snapshot.head_sha:
        raise RuntimeError("fetched PR ref does not match observed head SHA")
    files = _local_changed_files_from_refs(merge_base, snapshot.head_sha)
    return ChangedFileProof(
        base_sha=snapshot.base_sha,
        head_sha=snapshot.head_sha,
        merge_base_sha=merge_base,
        expected_count=snapshot.changed_files,
        files=files,
    )


def fetch_changed_files(pr: int, repo: str, gh: str, head_sha: str | None = None) -> list[dict]:
    """Fetch PR changed files from GitHub's trusted PR file list."""

    proof = fetch_changed_file_proof(pr, repo, gh, head_sha=head_sha)
    return proof.files


def _assert_stable_changed_file_proof(
    *,
    proof: ChangedFileProof,
    after_refs: tuple[str, str, int],
) -> None:
    after_base, after_head, after_count = after_refs
    if proof.base_sha != after_base:
        raise RuntimeError("GitHub PR base changed during body/file proof fetch")
    if proof.head_sha != after_head:
        raise RuntimeError("GitHub PR head changed during body/file proof fetch")
    if proof.expected_count != after_count:
        raise RuntimeError("GitHub PR changed-file count changed during body/file proof fetch")


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


def _assert_stable_review_thread_state(
    *,
    before: tuple[list[dict], str, list[dict], list[dict]],
    after: tuple[list[dict], str, list[dict], list[dict]],
    bot_logins: Sequence[str],
) -> None:
    before_nodes, before_head, before_reviews, before_comments = before
    after_nodes, after_head, after_reviews, after_comments = after
    if before_head != after_head:
        raise RuntimeError("GitHub PR head changed during body/file proof fetch")
    before_generation = review_attestation_generation(
        before_reviews,
        before_comments,
        head_sha=before_head,
        bot_logins=bot_logins,
    )
    after_generation = review_attestation_generation(
        after_reviews,
        after_comments,
        head_sha=after_head,
        bot_logins=bot_logins,
    )
    if before_generation != after_generation:
        raise RuntimeError("GitHub Codex review generation changed during body/file proof fetch")
    if review_thread_generation(before_nodes) != review_thread_generation(after_nodes):
        raise RuntimeError("GitHub review thread generation changed during body/file proof fetch")


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
        "--changed-files-file",
        help="JSON file of PR changed-file objects (test/dry-run; skips fetching live changed files)",
    )
    parser.add_argument(
        "--head-sha",
        help="PR head SHA for current-head Codex review state in test/dry-run mode",
    )
    parser.add_argument(
        "--pr-updated-at",
        help="GitHub PR updatedAt timestamp for fresh-head Codex review-window checks",
    )
    parser.add_argument(
        "--review-grace-seconds",
        default=os.environ.get("ATLAS_CODEX_REVIEW_GRACE_SECONDS", str(_DEFAULT_CODEX_REVIEW_GRACE_SECONDS)),
        help="seconds after a PR update to wait for Codex activity before allowing a quiet no-thread pass",
    )
    parser.add_argument(
        "--wait-for-review-window",
        action="store_true",
        help="wait/refetch once when a live PR is still inside the fresh-head review window",
    )
    args = parser.parse_args(argv)

    try:
        bots = parse_bot_logins(args.bots)
        review_grace_seconds = parse_review_grace_seconds(args.review_grace_seconds)
    except ValueError as exc:
        print(f"live reconciliation: {exc}", file=sys.stderr)
        return 2

    try:
        head_sha = args.head_sha
        reviews: Sequence[dict] | None = None
        comments: Sequence[dict] | None = None
        changed_files: Sequence[dict] | None = None
        changed_file_proof: ChangedFileProof | None = None
        pr_updated_at = args.pr_updated_at
        if pr_updated_at is not None:
            _parse_github_timestamp(pr_updated_at)

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
            before_snapshot = fetch_consistent_review_thread_snapshot(
                args.pr,
                owner,
                name,
                args.gh,
                bots,
            )
            nodes, head_sha, reviews, comments = before_snapshot

        if args.reviews_file:
            reviews = json.loads(Path(args.reviews_file).read_text(encoding="utf-8"))
        if args.comments_file:
            comments = json.loads(Path(args.comments_file).read_text(encoding="utf-8"))
        if args.changed_files_file:
            changed_files = json.loads(Path(args.changed_files_file).read_text(encoding="utf-8"))

        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
        elif args.pr is not None and args.repo:
            body = fetch_body(args.pr, args.repo, args.gh)
        else:
            body = ""

        if pr_updated_at is None and args.pr is not None and args.repo and not args.threads_file:
            pr_updated_at = fetch_pr_updated_at(args.pr, args.repo, args.gh)

        if (
            args.wait_for_review_window
            and not args.threads_file
            and args.pr is not None
            and args.repo
            and missing_codex_activity_inside_review_grace(
                nodes,
                bots,
                reviews=reviews,
                comments=comments,
                head_sha=head_sha,
                pr_updated_at=pr_updated_at,
                review_grace_seconds=review_grace_seconds,
            )
        ):
            remaining = review_grace_remaining_seconds(
                pr_updated_at,
                review_grace_seconds=review_grace_seconds,
            )
            wait_seconds = max(1, math.ceil(remaining))
            print(
                "live reconciliation: waiting "
                f"{wait_seconds} second(s) for the Codex review window to close",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
            before_snapshot = fetch_consistent_review_thread_snapshot(
                args.pr,
                owner,
                name,
                args.gh,
                bots,
            )
            nodes, head_sha, reviews, comments = before_snapshot
            body = fetch_body(args.pr, args.repo, args.gh)
            pr_updated_at = fetch_pr_updated_at(args.pr, args.repo, args.gh)
            changed_files = None
            changed_file_proof = None

        needs_live_changed_file_proof = (
            changed_files is None
            and args.pr is not None
            and args.repo
            and docs_only_exemption_needs_file_proof(
                nodes,
                body,
                bots,
                reviews=reviews,
                comments=comments,
                head_sha=head_sha,
            )
        )
        if needs_live_changed_file_proof:
            changed_file_proof = fetch_changed_file_proof(args.pr, args.repo, args.gh, head_sha=head_sha)
            changed_files = changed_file_proof.files
        if needs_live_changed_file_proof and not args.threads_file and args.pr is not None and args.repo:
            body_after = fetch_body(args.pr, args.repo, args.gh)
            after_snapshot = fetch_consistent_review_thread_snapshot(
                args.pr,
                owner,
                name,
                args.gh,
                bots,
            )
            body_final = fetch_body(args.pr, args.repo, args.gh)
            after_refs = fetch_pr_refs(args.pr, args.repo, args.gh)
            if body != body_after or body != body_final:
                raise RuntimeError("GitHub PR body changed during body/file proof fetch")
            if changed_file_proof is None:
                raise RuntimeError("changed-file proof missing during body/file proof fetch")
            _assert_stable_changed_file_proof(
                proof=changed_file_proof,
                after_refs=after_refs,
            )
            _assert_stable_review_thread_state(
                before=before_snapshot,
                after=after_snapshot,
                bot_logins=bots,
            )
            nodes, head_sha, reviews, comments = after_snapshot
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"live reconciliation: GitHub API/read error: {exc}", file=sys.stderr)
        return 2

    code, messages = evaluate(
        nodes,
        body,
        bots,
        reviews=reviews,
        comments=comments,
        changed_files=changed_files,
        head_sha=head_sha,
        pr_updated_at=pr_updated_at,
        review_grace_seconds=review_grace_seconds,
    )
    print("live AI reconciliation check")
    print("-" * 60)
    for line in messages:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
