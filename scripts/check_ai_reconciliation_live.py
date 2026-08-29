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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

_DEFAULT_BOTS = ("chatgpt-codex-connector", "chatgpt-codex-connector[bot]")
_CLEAN_CODEX_REVIEW_TEXT = "didn't find any major issues"
_DEFAULT_CODEX_REVIEW_GRACE_SECONDS = 300
_REVIEWED_COMMIT_RE = re.compile(r"\*\*Reviewed commit:\*\*\s*`(?P<sha>[0-9a-f]{10,40})`", re.IGNORECASE)
_DEFINED_REVIEW_RULE_IDS = tuple(f"R{number}" for number in range(1, 15))
_DEFINED_RULE_ID_RE = "(?:" + "|".join(
    sorted((re.escape(rule_id) for rule_id in _DEFINED_REVIEW_RULE_IDS), key=len, reverse=True)
) + ")"
_RULE_REFERENCE_RE = rf"{_DEFINED_RULE_ID_RE}(?:/{_DEFINED_RULE_ID_RE})*"
_NUMERIC_POTENTIAL_RULE_REFERENCE_RE = r"[Rr]\d+(?:/[Rr]\d+)*"
_PARTIAL_INITIAL_RULE_REFERENCE_RE = r"[Rr](?=\s*(?:\(|:|/|[-—]))"
_LEADING_PARTIAL_RULE_REFERENCE_RE = r"[Rr](?=$|\s+\S|\s*(?:\(|:|/|[-—]))"
_POTENTIAL_RULE_REFERENCE_RE = (
    rf"(?:{_NUMERIC_POTENTIAL_RULE_REFERENCE_RE}|{_PARTIAL_INITIAL_RULE_REFERENCE_RE})"
)
_LEADING_POTENTIAL_RULE_REFERENCE_RE = (
    rf"(?:{_NUMERIC_POTENTIAL_RULE_REFERENCE_RE}|{_LEADING_PARTIAL_RULE_REFERENCE_RE})"
)
_RULE_SEVERITY_RE = r"\([A-Z][A-Z0-9 _-]*\)"
_LEGACY_COMPLETE_RULE_LABEL_RE = (
    rf"{_RULE_REFERENCE_RE}(?:"
    rf"\s+{_RULE_SEVERITY_RE}(?:\s+[—-]\s+\S|\s*:\s+\S|\s+(?![:—-])\S)"
    rf"|\s+[—-]\s+\S"
    rf")"
)
_COMPLETE_RULE_LABEL_RE = (
    rf"(?:{_LEGACY_COMPLETE_RULE_LABEL_RE}|{_RULE_REFERENCE_RE}:\s+\S)"
)
_REVIEW_TITLE_STOP_RE = re.compile(rf"\s+{_LEGACY_COMPLETE_RULE_LABEL_RE}")
_REVIEW_RULE_LABEL_RE = re.compile(rf"^{_COMPLETE_RULE_LABEL_RE}")
_POTENTIAL_RULE_EVIDENCE_RE = re.compile(
    rf"(?<![^\W_])(?>{_POTENTIAL_RULE_REFERENCE_RE})(?!\d)"
    rf"(?=\S|\s*(?:\(|:|/|[-—]))"
)
_LEADING_RULE_REFERENCE_RE = re.compile(
    rf"^[\W_]*(?P<reference>(?>{_LEADING_POTENTIAL_RULE_REFERENCE_RE}))(?!\d)"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_UNPARSEABLE_THREAD_DECISION = "<unparseable trusted-bot review title>"
_MIN_REVIEW_TITLE_CHARS = 24
_MIN_REVIEW_TITLE_TOKENS = 4
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
      headRefOid
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

    Uses the canonical PR-body section selection plus Phase-2 markers, so
    "what counts as a resolved record" stays defined by the same top-level,
    unfenced section shape as the PR-body contract.
    """
    p2 = _load_phase2()
    section = canonical_reconciliation_section(body)
    if section is None:
        return "absent"
    if p2.UNRESOLVED_RE.search(section):
        return "acknowledges_open"
    if p2.RESOLVED_RE.search(section):
        return "claims_clear"
    return "unmarked"


def body_uses_no_findings(body: str) -> bool:
    """Return true when the reconciliation record claims no findings existed."""

    p2 = _load_phase2()
    section = canonical_reconciliation_section(body)
    if section is None:
        return False
    return any(
        p2.NO_FINDINGS_RE.fullmatch(line.strip())
        for line in section.splitlines()
        if line.strip()
    )


def _normalized_decision(text: str) -> str:
    return " ".join(_NON_ALNUM_RE.sub(" ", text.lower()).split())


def _first_display_line(body_text: str) -> str:
    """Return the first nonblank review line for diagnostic output only."""

    for line in body_text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _has_bounded_decision_evidence(text: str) -> bool:
    normalized = _normalized_decision(text)
    return (
        len(normalized) >= _MIN_REVIEW_TITLE_CHARS
        and len(normalized.split()) >= _MIN_REVIEW_TITLE_TOKENS
    )


def _bounded_title_root(line: str) -> str:
    """Return a full review-title root, never an ambiguous label fragment."""

    match = _REVIEW_TITLE_STOP_RE.search(line)
    if match:
        root = line[: match.start()].strip()
    elif (
        _POTENTIAL_RULE_EVIDENCE_RE.search(line)
        or _LEADING_RULE_REFERENCE_RE.search(line)
    ):
        return ""
    else:
        root = line.strip()
    if not _has_bounded_decision_evidence(root):
        return ""
    return root


def _has_unvalidated_rule_evidence(line: str) -> bool:
    """Return whether a potential title contains non-complete rule evidence."""

    leading_match = _LEADING_RULE_REFERENCE_RE.search(line)
    if leading_match is not None:
        prefix = line[: leading_match.start("reference")]
        reference = leading_match.group("reference")
        remainder = line[leading_match.end("reference") :]
        is_canonical_reference = re.fullmatch(_RULE_REFERENCE_RE, reference) is not None
        if (
            prefix
            or not is_canonical_reference
            or (remainder.strip() and not _REVIEW_RULE_LABEL_RE.match(line))
        ):
            return True
    for match in _POTENTIAL_RULE_EVIDENCE_RE.finditer(line):
        candidate = line[match.start() :]
        if not _REVIEW_RULE_LABEL_RE.match(candidate):
            return True
    return False


def _evidenced_root_decision(body_text: str) -> str:
    """Return a bounded title with complete adjacent or inline rule evidence."""

    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if _has_unvalidated_rule_evidence(line):
            return ""
        root = _bounded_title_root(line)
        if root and _REVIEW_TITLE_STOP_RE.search(line):
            return root
        if (
            index + 1 < len(lines)
            and _REVIEW_RULE_LABEL_RE.match(lines[index + 1])
            and not _REVIEW_RULE_LABEL_RE.match(line)
            and not _REVIEW_TITLE_STOP_RE.search(line)
            and _has_bounded_decision_evidence(line)
        ):
            return line
        if not root and (
            _REVIEW_RULE_LABEL_RE.match(line)
            or _REVIEW_TITLE_STOP_RE.search(line)
            or _POTENTIAL_RULE_EVIDENCE_RE.search(line)
            or _LEADING_RULE_REFERENCE_RE.search(line)
        ):
            return ""
    return ""


def _thread_root_decision(thread_summary: dict) -> str:
    return str(thread_summary.get("decision") or "").strip()


def canonical_reconciliation_section(body: str) -> str | None:
    """Return the canonical top-level, unfenced AI reconciliation section.

    The PR-body contract intentionally ignores fenced or indented heading-like
    examples. The live history correlator must use that same section selection
    or a non-canonical heading can satisfy live correlation while the body audit
    validates a different record.
    """

    audit = _load_pr_body_audit()
    lines = audit.unfenced_lines(body)
    section_lines = audit.section_body_lines(lines, "AI reconciliation")
    if section_lines is None:
        return None
    return "## AI reconciliation\n" + "\n".join(section_lines)


def reconciliation_disposition_roots(body: str) -> list[str]:
    """Return normalized finding/root-decision names from structured dispositions."""

    p2 = _load_phase2()
    section = canonical_reconciliation_section(body)
    if section is None:
        return []
    roots: list[str] = []
    for raw_line in section.splitlines():
        bullet = p2.BULLET_RE.match(raw_line)
        if bullet is None:
            continue
        item = bullet.group("body").strip()
        tokens = list(p2.DISPOSITION_TOKEN_RE.finditer(item))
        if len(tokens) != 1:
            continue
        root = item[: tokens[0].start()].strip()
        root = p2.FINDING_SEPARATOR_RE.sub(" ", root).strip(" :-–—")
        normalized = _normalized_decision(root)
        if normalized:
            roots.append(normalized)
    return roots


def root_decision_matches_thread(root: str, decision: str) -> bool:
    """Return true when a structured disposition names the thread decision."""

    if not _has_bounded_decision_evidence(root) or not _has_bounded_decision_evidence(decision):
        return False
    if root == decision:
        return True
    return root in decision or decision in root


def _covered_thread_decisions(roots: Sequence[str], decisions: Sequence[str]) -> set[str]:
    """Return distinct decisions covered without ambiguous root reuse."""

    distinct_decisions = tuple(dict.fromkeys(decisions))
    covered: set[str] = set()
    for root in roots:
        candidates = tuple(
            decision
            for decision in distinct_decisions
            if root_decision_matches_thread(root, decision)
        )
        if root in candidates:
            covered.add(root)
        elif len(candidates) == 1:
            covered.add(candidates[0])
    return covered


def missing_thread_dispositions(
    thread_summaries: Sequence[dict],
    body: str,
) -> list[dict]:
    """Return bot-thread summaries not named by any structured disposition."""

    roots = reconciliation_disposition_roots(body)
    prepared = [
        (summary, _thread_root_decision(summary)) for summary in thread_summaries
    ]
    normalized_decisions = [
        _normalized_decision(decision) for _, decision in prepared if decision
    ]
    covered = _covered_thread_decisions(roots, normalized_decisions)
    missing: list[dict] = []
    for summary, decision in prepared:
        normalized = _normalized_decision(decision)
        if not normalized:
            copy = dict(summary)
            copy["decision"] = _UNPARSEABLE_THREAD_DECISION
            missing.append(copy)
            continue
        if normalized not in covered:
            copy = dict(summary)
            copy["decision"] = decision
            missing.append(copy)
    return missing


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
        summary = _bot_thread_summary(node, wanted)
        if summary is None:
            continue
        found.append(summary)
    return found


def bot_review_threads(nodes: Sequence[dict], bot_logins: Sequence[str]) -> list[dict]:
    """Return all review threads authored by a known bot, resolved or not."""

    wanted = frozenset(b.lower() for b in bot_logins)
    found: list[dict] = []
    for node in nodes or []:
        summary = _bot_thread_summary(node, wanted)
        if summary is not None:
            found.append(summary)
    return found


def _bot_thread_summary(node: dict, wanted: frozenset[str]) -> dict | None:
    comments = ((node.get("comments") or {}).get("nodes")) or []
    author = ""
    title = ""
    snippet = ""
    if comments:
        author = (((comments[0] or {}).get("author") or {}).get("login")) or ""
        body_text = ((comments[0] or {}).get("bodyText") or "").strip()
        title = _first_display_line(body_text)
        snippet = " ".join(body_text.split())
    if author.lower() not in wanted:
        return None
    return {
        "path": node.get("path") or "?",
        "line": node.get("line"),
        "author": author or "?",
        "title": title,
        "decision": _evidenced_root_decision(body_text),
        "snippet": (snippet[:120] + "...") if len(snippet) > 120 else snippet,
    }


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

    if not open_threads:
        prior_threads = bot_review_threads(nodes, bot_logins)
        body_class = classify_body(body)
        if body_uses_no_findings(body):
            if prior_threads:
                messages.append(
                    "AI reconciliation records no-findings, but scoped Codex "
                    "review-thread history contains findings:"
                )
                for t in prior_threads:
                    loc = t["path"] if t["line"] is None else f"{t['path']}:{t['line']}"
                    messages.append(f"  - [{t['author']}] {loc}: {t['snippet']}")
                messages.append(
                    "Replace no-findings with fixed-in/waived/not-applicable "
                    "dispositions for the resolved review findings."
                )
                return 1, messages
        has_structured_dispositions = bool(reconciliation_disposition_roots(body))
        missing_dispositions = []
        if prior_threads and (body_class == "claims_clear" or has_structured_dispositions):
            missing_dispositions = missing_thread_dispositions(prior_threads, body)
        if missing_dispositions:
            messages.append(
                "AI reconciliation ledger is missing dispositions for scoped "
                "Codex review-thread history:"
            )
            for t in missing_dispositions:
                loc = t["path"] if t["line"] is None else f"{t['path']}:{t['line']}"
                messages.append(f"  - [{t['author']}] {loc}: {t['decision']}")
            if any(t["decision"] == _UNPARSEABLE_THREAD_DECISION for t in missing_dispositions):
                messages.append(
                    "A trusted bot review thread has no normalizable title and cannot be "
                    "reconciled by a generic disposition; restore or obtain named review "
                    "evidence before merge."
                )
            else:
                messages.append(
                    "Add one fixed-in/waived/not-applicable disposition naming each "
                    "review-thread root decision before merge."
                )
            return 1, messages
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


def fetch_thread_snapshot(pr: int, owner: str, name: str, gh: str) -> tuple[str, list[dict]]:
    """Fetch the PR head and ALL review threads from the thread query only.

    The required live gate is thread-only, so review/comment attestation API
    failures must not block a clear thread snapshot.
    """
    nodes: list[dict] = []
    head_sha = ""
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
        observed_head = pull_request.get("headRefOid")
        if not isinstance(observed_head, str) or not observed_head:
            raise RuntimeError("GitHub GraphQL response malformed: pullRequest.headRefOid is missing")
        if head_sha and observed_head != head_sha:
            raise RuntimeError("GitHub GraphQL response changed PR head during reviewThreads pagination")
        head_sha = observed_head
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
    return head_sha, nodes


def fetch_threads(pr: int, owner: str, name: str, gh: str) -> list[dict]:
    """Fetch ALL review threads, paginating so a PR with >100 threads cannot

    hide an unresolved finding past the first page and pass as clear.
    """
    _, nodes = fetch_thread_snapshot(pr, owner, name, gh)
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


def fetch_consistent_review_thread_snapshot(
    pr: int,
    owner: str,
    name: str,
    gh: str,
    bot_logins: Sequence[str],
) -> tuple[list[dict], str, list[dict], list[dict]]:
    """Fetch threads twice and fail closed only if head or thread state moves."""

    head_before, nodes_before = fetch_thread_snapshot(pr, owner, name, gh)
    head_after, nodes_after = fetch_thread_snapshot(pr, owner, name, gh)
    if head_before != head_after:
        raise RuntimeError("GitHub PR head changed during review/thread snapshot fetch")
    if review_thread_generation(nodes_before) != review_thread_generation(nodes_after):
        raise RuntimeError("GitHub review thread generation changed during review/thread snapshot fetch")
    return nodes_after, head_after, [], []


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
        help="Deprecated compatibility input; live reconciliation no longer waits on PR updatedAt.",
    )
    parser.add_argument(
        "--review-grace-seconds",
        default=os.environ.get("ATLAS_CODEX_REVIEW_GRACE_SECONDS", str(_DEFAULT_CODEX_REVIEW_GRACE_SECONDS)),
        help="Deprecated compatibility input; ignored by the open-thread-only gate.",
    )
    parser.add_argument(
        "--wait-for-review-window",
        action="store_true",
        help="Deprecated compatibility flag; accepted but ignored.",
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
        changed_files: Sequence[dict] | None = None
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
    )
    print("live AI reconciliation check")
    print("-" * 60)
    for line in messages:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
