"""Date parsing helpers for support-ticket source rows.

Numeric two-field dates (02/01/2026) are locale-ambiguous: US exports are
month-first, most other locales are day-first. A single value cannot decide,
but an UPLOAD can: any value with a first field over 12 proves day-first,
any value with a second field over 12 proves month-first, and conflicting
evidence means the convention is unknowable -- those values stay unparsed
rather than silently transposed.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable

DATE_CONVENTION_MONTH_FIRST = "month_first"
DATE_CONVENTION_DAY_FIRST = "day_first"
DATE_CONVENTION_AMBIGUOUS = "ambiguous"
DATE_CONVENTION_UNKNOWN = "unknown"

_MONTH_FIRST_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%m-%d-%Y",
    "%m-%d-%y",
)
_DAY_FIRST_FORMATS = (
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
)
_NUMERIC_DATE_RE = re.compile(
    r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$"
)


def infer_support_ticket_date_convention(values: Iterable[Any]) -> str:
    """Infer the numeric-date convention for one upload.

    Returns ``month_first`` / ``day_first`` when the upload's own values
    prove it, ``ambiguous`` when they contradict each other, and
    ``unknown`` when no value is decisive (all fields <= 12).
    """

    day_first_evidence = 0
    month_first_evidence = 0
    for value in values:
        text = _clean(value)
        match = _NUMERIC_DATE_RE.match(text)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        # Evidence must PARSE under the convention it implies. Malformed
        # cells (99/01, 00/13, mixed separators 13/01-2026, impossible days
        # like 30/02) prove nothing and must not decide the upload.
        if 12 < first <= 31 and 1 <= second <= 12:
            if _parses_with(text, _DAY_FIRST_FORMATS):
                day_first_evidence += 1
        elif 12 < second <= 31 and 1 <= first <= 12:
            if _parses_with(text, _MONTH_FIRST_FORMATS):
                month_first_evidence += 1
    if day_first_evidence and month_first_evidence:
        return DATE_CONVENTION_AMBIGUOUS
    if day_first_evidence:
        return DATE_CONVENTION_DAY_FIRST
    if month_first_evidence:
        return DATE_CONVENTION_MONTH_FIRST
    return DATE_CONVENTION_UNKNOWN


def parse_support_ticket_source_date(
    value: Any,
    *,
    convention: str = DATE_CONVENTION_UNKNOWN,
) -> date | None:
    """Parse source dates from support-ticket exports.

    ISO-style inputs always parse. Numeric two-field dates parse under the
    upload's inferred ``convention``: day-first when proven, month-first
    when proven or unknown (preserving the historical US default), and NOT
    AT ALL when the upload's evidence is contradictory (``ambiguous``) --
    a silently transposed date is worse than a missing one.
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
    if _NUMERIC_DATE_RE.match(text):
        if convention == DATE_CONVENTION_AMBIGUOUS:
            return None
        formats = (
            _DAY_FIRST_FORMATS
            if convention == DATE_CONVENTION_DAY_FIRST
            else _MONTH_FIRST_FORMATS
        )
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def _parses_with(text: str, formats: tuple[str, ...]) -> bool:
    for fmt in formats:
        try:
            datetime.strptime(text, fmt)
        except ValueError:
            continue
        return True
    return False


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


__all__ = [
    "DATE_CONVENTION_AMBIGUOUS",
    "DATE_CONVENTION_DAY_FIRST",
    "DATE_CONVENTION_MONTH_FIRST",
    "DATE_CONVENTION_UNKNOWN",
    "infer_support_ticket_date_convention",
    "parse_support_ticket_source_date",
]
