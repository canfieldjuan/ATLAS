#!/usr/bin/env python3
"""Capture the finetunelab Zendesk dev tickets into a sanitized product-proof corpus.

This wraps the existing live export client
(:func:`extracted_content_pipeline.support_ticket_zendesk_export.export_zendesk_full_thread_artifact`)
and projects the raw ``{ticket, comments}`` entries down to a committed,
PII-scrubbed fixture in the same ``tickets + comments`` shape as
``tests/fixtures/zendesk_full_thread_seed_sample.json``.

The operator runs this with their own ``ATLAS_CONTENT_OPS_ZENDESK_*`` credentials;
the script never writes credentials, tokens, requester identities, or raw author
fields into the committed artifact. Sanitization is a pure function
(:func:`sanitize_zendesk_export`) so it is unit-tested without live credentials.

Usage (operator):

    # preview only -- fetch + sanitize, print summary, write nothing
    .venv/bin/python scripts/capture_zendesk_product_proof_corpus.py --dry-run

    # write the sanitized fixture
    .venv/bin/python scripts/capture_zendesk_product_proof_corpus.py \\
        --run-tag atlas_product_proof_20260614 \\
        --out docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json

Expected-outcome labels (cluster_theme / should_publish_answer / unresolved) are
left null for a human labeling pass; ``has_private_note`` is derived from the
data because it is a fact, not a judgment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_macro_writeback_zendesk import (  # noqa: E402
    ZendeskMacroCredentials,
)
from extracted_content_pipeline.support_ticket_zendesk_export import (  # noqa: E402
    DEFAULT_ZENDESK_EXPORT_LIMIT,
    export_zendesk_full_thread_artifact,
)

# Whitelist projection: only these keys ever leave the raw ticket. Everything
# else (submitter_id, assignee_id, organization_id, tags, custom_fields,
# collaborators, url, ...) is dropped by construction. Raw requester/author IDs
# are NOT carried through -- they are replaced by role pseudonyms so the
# importer's requester-vs-agent split still works without committing real IDs.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
_URL_TOKEN_RE = re.compile(r"(?i)\b(?:access_token|token|api_key|secret|password)=[^\s&]+")

# Role pseudonyms. The Zendesk importer splits customer wording from agent
# resolution by comparing comment.author_id to ticket.requester_id, so we keep
# those two field names but replace the raw IDs with stable role tokens.
_REQUESTER_ROLE = "requester"
_AGENT_ROLE = "agent"
_SYSTEM_ROLE = "system"
_AUTOMATION_CHANNELS = frozenset({"rule", "system", "automation", "trigger"})


def _clean(value: Any) -> str:
    return str(value or "").strip()


def scrub_text(text: Any) -> str:
    """Remove emails, phone-shaped runs, long identifier numbers, and URL tokens.

    Keeps short numbers (amounts like ``$49``, years like ``2024``) so the
    support content stays readable; only 6+ digit runs are treated as
    identifier-shaped.
    """

    cleaned = _clean(text)
    if not cleaned:
        return ""
    cleaned = _URL_TOKEN_RE.sub("[redacted]", cleaned)
    cleaned = _EMAIL_RE.sub("[email]", cleaned)
    cleaned = _PHONE_RE.sub("[phone]", cleaned)
    cleaned = _LONG_NUMBER_RE.sub("[number]", cleaned)
    return cleaned


def _comment_body(raw_comment: Mapping[str, Any]) -> str:
    body = raw_comment.get("plain_body")
    if not _clean(body):
        body = raw_comment.get("body")
    return scrub_text(body)


def _comment_role(raw_comment: Mapping[str, Any], *, raw_requester_id: str) -> str:
    """Derive a role pseudonym from raw author identity, then drop the raw ID."""

    author_id = _clean(raw_comment.get("author_id"))
    if raw_requester_id and author_id == raw_requester_id:
        return _REQUESTER_ROLE
    via = raw_comment.get("via")
    channel = via.get("channel") if isinstance(via, Mapping) else None
    if _clean(channel).lower() in _AUTOMATION_CHANNELS:
        return _SYSTEM_ROLE
    return _AGENT_ROLE


def _sanitized_comments(raw_comments: Any, *, raw_requester_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_comments, Sequence) or isinstance(raw_comments, (str, bytes, bytearray)):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_comments:
        if not isinstance(raw, Mapping):
            continue
        role = _comment_role(raw, raw_requester_id=raw_requester_id)
        if role == _SYSTEM_ROLE:
            # Automation/system comments are neither the customer's question nor
            # an agent resolution. Drop them so a non-boilerplate automation
            # comment cannot reach the importer and become resolution_text.
            continue
        out.append({
            "public": bool(raw.get("public", True)),
            "author_id": role,
            "body": _comment_body(raw),
        })
    return out


def _satisfaction_score(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return raw.get("score")
    if isinstance(raw, str):
        return raw or None
    return None


def _expected_metadata(comments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the factual label (private note present); leave judgments null."""

    has_private_note = any(not bool(c.get("public", True)) for c in comments)
    return {
        "cluster_theme": None,           # human label
        "should_publish_answer": None,   # human judgment
        "has_private_note": has_private_note,  # derived fact
        "reopened": None,                # human judgment
        "unresolved": None,              # human judgment
    }


def sanitize_zendesk_export(
    raw_artifact: Mapping[str, Any],
    *,
    subdomain: str,
    run_tag: str,
) -> dict[str, Any]:
    """Project a raw export into the committed, PII-scrubbed corpus fixture.

    Input is the shape returned by ``export_zendesk_full_thread_artifact``:
    ``{"tickets": [{"ticket": {...}, "comments": [...]}, ...]}``.
    """

    entries = raw_artifact.get("tickets")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        entries = []
    tickets: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw_ticket = entry.get("ticket")
        if not isinstance(raw_ticket, Mapping):
            continue
        raw_requester_id = _clean(raw_ticket.get("requester_id"))
        comments = _sanitized_comments(entry.get("comments"), raw_requester_id=raw_requester_id)
        tickets.append({
            # Local stable ID -- never the raw Zendesk ticket id.
            "id": f"zd-proof-{len(tickets) + 1:03d}",
            # Role pseudonym so the importer's author_id == requester_id split works.
            "requester_id": _REQUESTER_ROLE,
            "subject": scrub_text(raw_ticket.get("subject") or raw_ticket.get("raw_subject")),
            "description": scrub_text(raw_ticket.get("description")),
            "status": _clean(raw_ticket.get("status")) or None,
            "satisfaction_rating": _satisfaction_score(raw_ticket.get("satisfaction_rating")),
            "comments": comments,
            "expected": _expected_metadata(comments),
        })
    return {
        "source": "zendesk_trial_api",
        "subdomain": _clean(subdomain),
        "run_tag": _clean(run_tag),
        "ticket_count": len(tickets),
        "tickets": tickets,
    }


def _credentials_from_settings() -> ZendeskMacroCredentials:
    # The content_ops_zendesk_* fields live on the b2b_campaign sub-config, not
    # top-level settings. Reuse the canonical builder so the access path stays
    # in one place (it returns None when the credentials are incomplete).
    from atlas_brain.config import settings
    from atlas_brain._content_ops_macro_writeback import (
        zendesk_macro_credentials_from_config,
    )

    credentials = zendesk_macro_credentials_from_config(settings.b2b_campaign)
    if credentials is None:
        raise SystemExit(
            "Zendesk credentials are incomplete. Set ATLAS_CONTENT_OPS_ZENDESK_EMAIL, "
            "ATLAS_CONTENT_OPS_ZENDESK_API_TOKEN, and ATLAS_CONTENT_OPS_ZENDESK_SUBDOMAIN "
            "(or _BASE_URL) in .env before running the capture."
        )
    return credentials


async def capture_corpus(*, limit: int, run_tag: str) -> dict[str, Any]:
    credentials = _credentials_from_settings()
    raw = await export_zendesk_full_thread_artifact(credentials, limit=limit)
    subdomain = credentials.normalized_base_url().removeprefix("https://").removesuffix(".zendesk.com")
    return sanitize_zendesk_export(raw, subdomain=subdomain, run_tag=run_tag)


def _summary(corpus: Mapping[str, Any]) -> str:
    tickets = corpus.get("tickets") or []
    private = sum(1 for t in tickets if t.get("expected", {}).get("has_private_note"))
    comments = sum(len(t.get("comments") or []) for t in tickets)
    return (
        f"captured tickets={corpus.get('ticket_count')} comments={comments} "
        f"with_private_note={private} subdomain={corpus.get('subdomain')!r} "
        f"run_tag={corpus.get('run_tag')!r}"
    )


def _assert_no_secrets(corpus: Mapping[str, Any]) -> None:
    blob = json.dumps(corpus)
    for marker in ("authorization", "api_token", "Basic ", "/token:"):
        if marker.lower() in blob.lower():
            raise SystemExit(f"refusing to write: sanitized corpus contains '{marker}'")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture sanitized Zendesk product-proof corpus.")
    parser.add_argument("--limit", type=int, default=DEFAULT_ZENDESK_EXPORT_LIMIT)
    parser.add_argument("--run-tag", default="atlas_product_proof")
    parser.add_argument("--out", type=Path, help="Path to write the sanitized fixture JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force preview: fetch + sanitize + print summary, never write (even if --out is set).",
    )
    args = parser.parse_args(argv)

    corpus = asyncio.run(capture_corpus(limit=int(args.limit), run_tag=_clean(args.run_tag)))
    _assert_no_secrets(corpus)
    print(_summary(corpus))

    # Writing is opt-in: it happens only with --out and not --dry-run. With no
    # --out (or --dry-run) the run is a preview, so a stray invocation never
    # writes a fixture.
    if args.dry_run or not args.out:
        reason = "dry-run" if args.dry_run else "no --out"
        print(f"{reason}: preview only, no file written")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
