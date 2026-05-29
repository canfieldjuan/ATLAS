"""Adapt generated FAQ output into campaign source rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


FAQ_OUTPUT_SOURCE_TYPE = "faq_output"

_FAQ_OUTPUT_MARKER_KEYS = (
    "generated",
    "markdown",
    "output_checks",
    "ticket_source_count",
    "saved_ids",
)
_FAQ_ITEM_ID_KEYS = (
    "faq_id",
    "faq_draft_id",
    "draft_id",
    "article_id",
    "id",
)
_FAQ_ITEM_URL_KEYS = ("url", "source_url", "article_url", "faq_url")
_SLUG_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def is_faq_output_bundle(value: Any) -> bool:
    """Return whether a mapping looks like a generated FAQ output bundle."""

    if not isinstance(value, Mapping):
        return False
    items = value.get("items")
    if not _is_sequence(items):
        return False
    return any(key in value for key in _FAQ_OUTPUT_MARKER_KEYS)


def faq_output_to_source_rows(faq_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert generated FAQ output into source rows for campaign ingestion."""

    if not is_faq_output_bundle(faq_output):
        return []

    saved_ids = _text_values(faq_output.get("saved_ids"))
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(faq_output.get("items") or (), start=1):
        if not isinstance(item, Mapping):
            continue
        saved_id = saved_ids[index - 1] if index <= len(saved_ids) else ""
        row = faq_item_to_source_row(item, rank=index, saved_id=saved_id)
        if row:
            rows.append(row)
    return rows


def faq_item_to_source_row(
    item: Mapping[str, Any],
    *,
    rank: int,
    saved_id: str = "",
) -> dict[str, Any]:
    """Convert one generated FAQ item into one source-material row."""

    text = _faq_item_text(item)
    if not text:
        return {}

    topic = _clean_text(item.get("topic"))
    question = _clean_text(item.get("question"))
    summary = _clean_text(item.get("summary"))
    steps = _text_values(item.get("steps") or item.get("action_items"))
    evidence_quotes = _text_values(item.get("evidence_quotes"))
    source_ids = _text_values(item.get("source_ids"))
    title = question or topic or f"FAQ item {rank}"
    source_id = _source_id(item, rank=rank, saved_id=saved_id, title=title)

    row: dict[str, Any] = {
        "source_type": FAQ_OUTPUT_SOURCE_TYPE,
        "source_id": source_id,
        "source_title": title,
        "text": text,
        "faq_rank": rank,
        "faq_question": question,
        "faq_topic": topic,
        "faq_summary": summary,
        "faq_steps": steps,
        "faq_customer_language": _customer_language(question, evidence_quotes),
        "faq_source_ticket_ids": source_ids,
        "faq_answer_evidence_status": _clean_text(
            item.get("answer_evidence_status")
        ),
        "question_source": _clean_text(item.get("question_source")),
        "pain_points": [topic] if topic else [],
    }
    for key in (
        "frequency",
        "weighted_frequency",
        "failure_risk_score",
        "opportunity_score",
        "ticket_count",
        "evidence_count",
        "resolution_source_count",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            row[key] = value
    url = _first_text(item, _FAQ_ITEM_URL_KEYS)
    if url:
        row["source_url"] = url
    return _drop_empty(row)


def _faq_item_text(item: Mapping[str, Any]) -> str:
    parts: list[str] = []
    question = _clean_text(item.get("question"))
    topic = _clean_text(item.get("topic"))
    summary = _clean_text(item.get("summary"))
    answer = _clean_text(item.get("answer"))
    steps = _text_values(item.get("steps") or item.get("action_items"))
    evidence_quotes = _text_values(item.get("evidence_quotes"))
    answer_evidence_status = _clean_text(item.get("answer_evidence_status"))

    if question:
        parts.append(f"FAQ question: {question}")
    if topic:
        parts.append(f"Topic: {topic}")
    if summary:
        parts.append(f"Summary: {summary}")
    if answer:
        parts.append(f"Answer: {answer}")
    if steps:
        numbered_steps = " ".join(
            f"{index}. {step}" for index, step in enumerate(steps, start=1)
        )
        parts.append(f"Steps: {numbered_steps}")
    if evidence_quotes:
        parts.append("Customer wording: " + " ".join(evidence_quotes))
    if answer_evidence_status:
        parts.append(f"Answer evidence status: {answer_evidence_status}")
    return "\n".join(parts)


def _source_id(
    item: Mapping[str, Any],
    *,
    rank: int,
    saved_id: str,
    title: str,
) -> str:
    candidate = _clean_text(saved_id) or _first_text(item, _FAQ_ITEM_ID_KEYS)
    if candidate:
        return candidate
    slug = _SLUG_SEPARATOR_RE.sub("-", title.lower()).strip("-")
    suffix = slug[:64].strip("-") or str(rank)
    return f"faq-output-{rank}-{suffix}"


def _customer_language(question: str, evidence_quotes: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in (question, *evidence_quotes):
        text = _clean_text(value)
        if text and text not in out:
            out.append(text)
    return out


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = _clean_text(row.get(key))
        if text:
            return text
    return ""


def _text_values(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif _is_sequence(value):
        values = value
    else:
        values = (value,)
    out: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _drop_empty(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if value not in (None, "", [], {})
    }


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


__all__ = [
    "FAQ_OUTPUT_SOURCE_TYPE",
    "faq_item_to_source_row",
    "faq_output_to_source_rows",
    "is_faq_output_bundle",
]
