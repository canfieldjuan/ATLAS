"""Shared support-ticket context markers for Content Ops generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SUPPORT_TICKET_TOPIC_TYPE = "content_ops_support_ticket_faq"
SUPPORT_TICKET_SOURCE = "support_ticket_provider"
SUPPORT_TICKET_SOURCE_MARKER = "support_ticket"
SUPPORT_TICKET_CATEGORY = "support tickets"
SUPPORT_TICKET_CATEGORY_MARKER = "support ticket"
SUPPORT_TICKET_DEFAULT_TOPIC = "Support-ticket questions customers keep asking"
UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD = "Uploaded support tickets"
UPLOADED_TICKETS_REVIEW_PERIOD = "uploaded tickets"
SUPPORT_TICKET_LAST_90_DAYS_REVIEW_PERIOD = "last 90 days"

SUPPORT_TICKET_COUNT_KEYS = (
    "included_ticket_row_count",
    "question_like_ticket_count",
)
SUPPORT_TICKET_SOURCE_COUNT_KEYS = (
    "source_row_count",
    *SUPPORT_TICKET_COUNT_KEYS,
)
SUPPORT_TICKET_CLUSTER_KEYS = ("top_ticket_clusters", "top_clusters")

_SUPPORT_TICKET_PERIOD_MARKERS = ("support-ticket", "support ticket")


def support_ticket_topic_filter() -> dict[str, str]:
    return {"topic_type": SUPPORT_TICKET_TOPIC_TYPE}


def is_support_ticket_topic_type(value: Any) -> bool:
    return str(value or "").strip() == SUPPORT_TICKET_TOPIC_TYPE


def is_uploaded_ticket_context(context: Mapping[str, Any]) -> bool:
    source_period = str(context.get("source_period") or "").strip().lower()
    review_period = str(context.get("review_period") or "").strip().lower()
    return (
        source_period == UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD.lower()
        or review_period == UPLOADED_TICKETS_REVIEW_PERIOD.lower()
    )


def is_support_ticket_context(context: Mapping[str, Any]) -> bool:
    source = str(context.get("source") or context.get("provider") or "").lower()
    source_period = str(context.get("source_period") or "").lower()
    category = str(context.get("category") or context.get("topic") or "").lower()
    return (
        SUPPORT_TICKET_SOURCE_MARKER in source
        or any(marker in source_period for marker in _SUPPORT_TICKET_PERIOD_MARKERS)
        or SUPPORT_TICKET_CATEGORY_MARKER in category
        or any(_positive_int(context.get(key)) is not None for key in SUPPORT_TICKET_COUNT_KEYS)
        or any(_has_cluster_rows(context.get(key)) for key in SUPPORT_TICKET_CLUSTER_KEYS)
    )


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _has_cluster_rows(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if not isinstance(value, Sequence):
        return False
    return any(isinstance(item, Mapping) for item in value)
