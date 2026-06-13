"""Date parsing helpers for support-ticket source rows."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

_US_DATE_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%m-%d-%Y",
    "%m-%d-%y",
)


def parse_support_ticket_source_date(value: Any) -> date | None:
    """Parse source dates from support-ticket exports.

    Existing ISO-style inputs stay accepted. Common US SaaS CSV export dates are
    accepted explicitly; natural-language and locale-ambiguous text is not.
    """

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    for fmt in _US_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
