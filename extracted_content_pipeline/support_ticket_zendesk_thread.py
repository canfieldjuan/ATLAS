"""Zendesk full-thread export normalization for support-ticket deflection."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .support_ticket_clustering import support_ticket_plain_text


_WHITESPACE_RE = re.compile(r"\s+")
_AUTO_ACK_PATTERNS = (
    re.compile(
        r"^a member of (the )?support team will get back to you"
        r"(?: within [^.]+)?[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^we (received|got) your (ticket|request)"
        r"(?: and (will|we'?ll) get back to you(?: soon| within [^.]+)?)?[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(thanks?|thank you) for (contacting|reaching out)"
        r"(?: to support)?[.!;]?"
        r"(?: we (received|got) your (ticket|request)[.!]?)?"
        r"(?: we'?ll get back to you(?: soon| within [^.]+)?[.!]?)?$",
        re.IGNORECASE,
    ),
)
_UNRATED_ZENDESK_SCORES = frozenset({"unoffered", "offered"})


@dataclass(frozen=True)
class ZendeskThreadImportResult:
    rows: list[dict[str, Any]]
    warnings: tuple[dict[str, Any], ...] = ()


def load_zendesk_full_thread_rows_from_json_bytes(
    data: bytes,
) -> ZendeskThreadImportResult:
    """Parse Zendesk thread JSON bytes into support-ticket rows."""

    try:
        artifact = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Zendesk full-thread JSON could not be parsed.") from exc
    return rows_from_zendesk_full_thread(artifact)


def rows_from_zendesk_full_thread(artifact: Any) -> ZendeskThreadImportResult:
    """Normalize Zendesk `{ticket, comments}` records into flat ticket rows."""

    ticket_entries = _ticket_entries(artifact)
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if ticket_entries is None:
        return ZendeskThreadImportResult(
            rows=[],
            warnings=({
                "code": "zendesk_thread_invalid_shape",
                "message": "Zendesk full-thread artifact must be an object or list.",
            },),
        )
    if not ticket_entries:
        return ZendeskThreadImportResult(
            rows=[],
            warnings=({
                "code": "zendesk_thread_empty",
                "message": "Zendesk full-thread artifact did not include tickets.",
            },),
        )
    for index, entry in enumerate(ticket_entries, start=1):
        normalized = _row_from_entry(entry, row_index=index)
        if normalized is None:
            warnings.append({
                "code": "zendesk_thread_ticket_missing",
                "row_index": index,
                "message": "Skipped Zendesk thread row because ticket was missing.",
            })
            continue
        row, row_warnings = normalized
        rows.append(row)
        warnings.extend(row_warnings)
    return ZendeskThreadImportResult(rows=rows, warnings=tuple(warnings))


def _ticket_entries(artifact: Any) -> Sequence[Any] | None:
    if isinstance(artifact, Mapping):
        tickets = artifact.get("tickets")
        if isinstance(tickets, Sequence) and not isinstance(
            tickets, (str, bytes, bytearray)
        ):
            return tickets
        return (artifact,)
    if isinstance(artifact, Sequence) and not isinstance(
        artifact, (str, bytes, bytearray)
    ):
        return artifact
    return None


def _row_from_entry(
    entry: Any,
    *,
    row_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not isinstance(entry, Mapping):
        return None
    ticket = entry.get("ticket") if isinstance(entry.get("ticket"), Mapping) else entry
    if not isinstance(ticket, Mapping):
        return None
    requester_id = _id_text(ticket.get("requester_id"))
    ticket_id = _clean(ticket.get("id")) or f"zendesk-ticket-{row_index}"
    subject = _plain(ticket.get("subject") or ticket.get("raw_subject"))
    customer_parts: list[str] = []
    resolution_parts: list[str] = []
    warnings: list[dict[str, Any]] = []
    comments = entry.get("comments")
    if comments in (None, "", [], {}):
        comments = ()
    if isinstance(comments, Mapping):
        comments = (comments,)
    if not isinstance(comments, Sequence) or isinstance(
        comments, (str, bytes, bytearray)
    ):
        warnings.append({
            "code": "zendesk_thread_comments_invalid",
            "row_index": row_index,
            "source_id": ticket_id,
            "message": "Ignored Zendesk comments because they were not a list.",
        })
        comments = ()
    private_comment_keys = {
        _text_key(_comment_text(comment))
        for comment in comments
        if isinstance(comment, Mapping) and comment.get("public") is False
    }
    description = _plain(ticket.get("description"))
    if description and _text_key(description) not in private_comment_keys:
        _append_unique(customer_parts, description)
    for comment in comments:
        if not isinstance(comment, Mapping):
            warnings.append({
                "code": "zendesk_thread_comment_not_object",
                "row_index": row_index,
                "source_id": ticket_id,
                "message": "Ignored Zendesk comment because it was not an object.",
            })
            continue
        if comment.get("public") is False:
            continue
        text = _comment_text(comment)
        if not text:
            continue
        author_id = _id_text(comment.get("author_id"))
        if requester_id and author_id == requester_id:
            _append_unique(customer_parts, text)
            continue
        if _looks_like_auto_ack(text):
            continue
        _append_unique(resolution_parts, text)
    if not (subject or customer_parts or resolution_parts):
        return None

    row: dict[str, Any] = {
        "ticket_id": ticket_id,
        "source_id": ticket_id,
        "source_type": "support_ticket",
    }
    if subject:
        row["subject"] = subject
    if customer_parts:
        row["description"] = "\n".join(customer_parts)
    if resolution_parts:
        row["resolution_text"] = "\n".join(resolution_parts)
    for source_key, target_key in (
        ("status", "ticket_status"),
        ("created_at", "created_at"),
        ("url", "source_url"),
    ):
        value = _clean(ticket.get(source_key))
        if value:
            row[target_key] = value
    csat = _satisfaction_rating(ticket.get("satisfaction_rating"))
    if csat:
        row["satisfaction_rating"] = csat
    return row, warnings


def _comment_text(comment: Mapping[str, Any]) -> str:
    for key in ("plain_body", "body", "html_body"):
        text = _plain(comment.get(key))
        if text:
            return text
    return ""


def _satisfaction_rating(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("score")
    text = _clean(value)
    if not text or text.lower() in _UNRATED_ZENDESK_SCORES:
        return ""
    return text


def _append_unique(parts: list[str], value: str) -> None:
    text = _plain(value)
    if not text:
        return
    key = text.lower()
    if any(existing.lower() == key for existing in parts):
        return
    parts.append(text)


def _looks_like_auto_ack(value: str) -> bool:
    text = _plain(value)
    return any(pattern.fullmatch(text) for pattern in _AUTO_ACK_PATTERNS)


def _plain(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", support_ticket_plain_text(_clean(value))).strip()


def _text_key(value: Any) -> str:
    return _plain(value).lower()


def _id_text(value: Any) -> str:
    return _clean(value)


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "ZendeskThreadImportResult",
    "load_zendesk_full_thread_rows_from_json_bytes",
    "rows_from_zendesk_full_thread",
]
