"""Support-ticket input packages for AI Content Ops generation.

This module is the adapter between host-owned ticket ingestion and the generic
Content Ops input-provider contract. It accepts already-loaded ticket rows or a
ticket bundle, keeps customer wording intact, and returns request inputs that
the existing FAQ, landing-page, and blog planners already understand.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .campaign_source_adapters import source_material_to_source_rows
from .content_ops_input_provider import ContentOpsInputPackage


DEFAULT_SUPPORT_TICKET_OUTPUTS: tuple[str, ...] = (
    "faq_markdown",
    "landing_page",
    "blog_post",
)
DEFAULT_FAQ_REPORT_CAMPAIGN_NAME = "FAQ Report"
DEFAULT_FAQ_REPORT_AUDIENCE = "Small teams answering repeat support questions"
DEFAULT_FAQ_REPORT_OFFER = (
    "Turn repeat support tickets into clear FAQ answers customers can use before "
    "they email support"
)
DEFAULT_FAQ_REPORT_CTA_LABEL = "Upload Ticket CSV -- Free Analysis"
DEFAULT_FAQ_REPORT_CTA_URL = "/systems/ai-content-ops/intake"
DEFAULT_FAQ_REPORT_TARGET_KEYWORD = "support ticket FAQ report"

_QUESTION_RE = re.compile(
    r"\b(can|could|do|does|how|is|should|what|when|where|why)\b[^?]*\?",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_QUESTION_STARTS = (
    "can ",
    "could ",
    "do ",
    "does ",
    "how ",
    "is ",
    "should ",
    "what ",
    "when ",
    "where ",
    "why ",
)

_SOURCE_ID_KEYS = ("source_id", "ticket_id", "id", "case_id", "conversation_id")
_SOURCE_TITLE_KEYS = ("source_title", "subject", "title", "summary")
_TEXT_KEYS = (
    "text",
    "body",
    "description",
    "message",
    "content",
    "complaint",
    "notes",
    "summary",
)
_DATE_KEYS = ("created_at", "submitted_at", "updated_at", "date")
_URL_KEYS = ("source_url", "ticket_url", "url", "link")
_COMPANY_KEYS = ("company_name", "account_name", "company", "account")
_VENDOR_KEYS = ("vendor_name", "product_name", "vendor", "product")
_CONTACT_EMAIL_KEYS = ("contact_email", "email", "customer_email")
_PAIN_KEYS = ("pain_category", "category", "intent", "topic")


def build_support_ticket_input_package(
    source_material: Any,
    *,
    provider: str = "support_ticket_upload",
    outputs: Sequence[str] = DEFAULT_SUPPORT_TICKET_OUTPUTS,
    window_days: int = 90,
    max_rows: int = 1000,
    campaign_name: str = DEFAULT_FAQ_REPORT_CAMPAIGN_NAME,
    audience: str = DEFAULT_FAQ_REPORT_AUDIENCE,
    offer: str = DEFAULT_FAQ_REPORT_OFFER,
    target_keyword: str = DEFAULT_FAQ_REPORT_TARGET_KEYWORD,
    cta_label: str = DEFAULT_FAQ_REPORT_CTA_LABEL,
    cta_url: str = DEFAULT_FAQ_REPORT_CTA_URL,
    secondary_keywords: Sequence[str] | None = None,
    objections: Sequence[str] | None = None,
    internal_links: Sequence[str] | None = None,
) -> ContentOpsInputPackage:
    """Build a Content Ops input package from support-ticket source material."""

    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")

    raw_rows = _rows_from_source_material(source_material)
    normalized_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not raw_rows:
        warnings.append({
            "code": "source_material_empty",
            "message": "No support-ticket source rows were provided.",
        })
    for index, row in enumerate(raw_rows[:max_rows], start=1):
        if not isinstance(row, Mapping):
            warnings.append({
                "code": "ticket_row_not_object",
                "row_index": index,
                "message": "Skipped ticket row because it was not an object.",
            })
            continue
        normalized = _normalize_ticket_row(row, row_index=index)
        if normalized:
            normalized_rows.append(normalized)
            continue
        warnings.append({
            "code": "ticket_row_missing_text",
            "row_index": index,
            "message": "Skipped ticket row because it did not include customer wording.",
        })
    if len(raw_rows) > max_rows:
        warnings.append({
            "code": "ticket_rows_truncated",
            "message": f"Used first {max_rows} ticket rows out of {len(raw_rows)}.",
            "row_count": len(raw_rows),
            "max_rows": max_rows,
            "truncated_row_count": len(raw_rows) - max_rows,
        })
    truncated_row_count = max(0, len(raw_rows) - max_rows)

    source_period = f"Last {window_days} days of support tickets"
    faq_questions = _ticket_questions(normalized_rows)
    resolved_secondary_keywords = tuple(secondary_keywords or (
        "customer support FAQ",
        "reduce repeat support tickets",
        "help center answers from support tickets",
    ))
    resolved_objections = tuple(objections or (
        "Will this publish automatically?",
        "Do we need a full-time docs person?",
        "Can our team review the answers first?",
    ))

    inputs = {
        "source_material": normalized_rows,
        "faq_window_days": window_days,
        "faq_source_types": ["support_ticket"],
        "faq_title": campaign_name,
        "topic": "Support-ticket questions customers keep asking",
        "filters": {"topic_type": "content_ops_support_ticket_faq"},
        "campaign_name": campaign_name,
        "offer": offer,
        "audience": audience,
        "target_keyword": target_keyword,
        "secondary_keywords": list(resolved_secondary_keywords),
        "search_intent": (
            "Small teams looking for a practical way to turn repeat support "
            "questions into help-center answers."
        ),
        "primary_entity": campaign_name,
        "audience_entity": audience,
        "competitors": ["manual help-center cleanup", "chatbot setup"],
        "objections": list(resolved_objections),
        "faq_questions": faq_questions,
        "source_period": source_period,
        "internal_links": list(internal_links or (DEFAULT_FAQ_REPORT_CTA_URL,)),
        "cta_label": cta_label,
        "cta_url": cta_url,
    }

    return ContentOpsInputPackage(
        provider=_clean(provider) or "support_ticket_upload",
        outputs=_normalize_outputs(outputs),
        target_mode="vendor_retention",
        ingestion_profile="existing_evidence",
        inputs=inputs,
        metadata={
            "source": "support_ticket_input_package",
            "source_row_count": len(raw_rows),
            "included_row_count": len(normalized_rows),
            "skipped_row_count": len(raw_rows) - len(normalized_rows),
            "truncated_row_count": truncated_row_count,
            "source_period": source_period,
        },
        warnings=tuple(warnings),
    )


def _rows_from_source_material(source_material: Any) -> list[Any]:
    if isinstance(source_material, str):
        text = _clean(source_material)
        return [{"text": text, "source_type": "support_ticket"}] if text else []
    return source_material_to_source_rows(source_material)


def _normalize_ticket_row(row: Any, *, row_index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    source_title = _first_text(row, _SOURCE_TITLE_KEYS)
    text = _ticket_text(row, source_title=source_title)
    if not text:
        return {}
    source_id = _first_text(row, _SOURCE_ID_KEYS) or f"ticket-{row_index}"
    normalized: dict[str, Any] = {
        "source_id": source_id,
        "source_type": _first_text(row, ("source_type", "type")) or "support_ticket",
        "source_title": source_title or source_id,
        "text": text,
    }
    for key, keys in (
        ("created_at", _DATE_KEYS),
        ("source_url", _URL_KEYS),
        ("company_name", _COMPANY_KEYS),
        ("vendor_name", _VENDOR_KEYS),
        ("contact_email", _CONTACT_EMAIL_KEYS),
        ("pain_category", _PAIN_KEYS),
    ):
        value = _first_value(row, keys)
        if value not in (None, "", [], {}):
            normalized[key] = value
    return normalized


def _ticket_text(row: Mapping[str, Any], *, source_title: str) -> str:
    parts: list[str] = []
    if source_title:
        parts.append(source_title)
    body = _first_text(row, _TEXT_KEYS)
    if body and body.lower() != source_title.lower():
        parts.append(body)
    comments = _comments_text(row)
    if comments:
        parts.append(comments)
    return _compact("\n".join(parts))


def _comments_text(row: Mapping[str, Any]) -> str:
    comments = _first_value(row, ("comments", "messages", "thread"))
    if isinstance(comments, Mapping):
        comments = (comments,)
    if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes, bytearray)):
        return ""
    parts: list[str] = []
    for item in comments:
        if isinstance(item, Mapping):
            text = _first_text(item, ("body", "message", "text", "content", "description"))
        else:
            text = _clean(item)
        if text:
            parts.append(text)
    return _compact("\n".join(parts))


def _ticket_questions(rows: Sequence[Mapping[str, Any]], *, limit: int = 6) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for candidate in (
            _first_question(row.get("text")),
            _question_like(row.get("source_title")),
        ):
            question = _compact(candidate)
            key = question.lower()
            if question and key not in seen:
                questions.append(question)
                seen.add(key)
            if len(questions) >= limit:
                return questions
    return questions


def _first_question(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    match = _QUESTION_RE.search(text)
    return match.group(0).strip() if match else _question_like(text)


def _question_like(value: Any) -> str:
    text = _compact(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith(_QUESTION_STARTS):
        return text if text.endswith("?") else f"{text}?"
    return ""


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _first_value(row, keys)
    return _clean(value)


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        normalized_key = _key(key)
        for raw_key, value in row.items():
            if _key(raw_key) == normalized_key and value not in (None, "", [], {}):
                return value
    return None


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _compact(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_outputs(outputs: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(outputs, str):
        return tuple(item.strip() for item in outputs.split(",") if item.strip())
    return tuple(str(output).strip() for output in outputs if str(output).strip())


__all__ = [
    "DEFAULT_FAQ_REPORT_AUDIENCE",
    "DEFAULT_FAQ_REPORT_CAMPAIGN_NAME",
    "DEFAULT_FAQ_REPORT_CTA_LABEL",
    "DEFAULT_FAQ_REPORT_CTA_URL",
    "DEFAULT_FAQ_REPORT_OFFER",
    "DEFAULT_FAQ_REPORT_TARGET_KEYWORD",
    "DEFAULT_SUPPORT_TICKET_OUTPUTS",
    "build_support_ticket_input_package",
]
