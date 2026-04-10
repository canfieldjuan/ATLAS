"""Shared reviewer-identity normalization helpers for B2B review ingestion."""

from __future__ import annotations

import re
from typing import Any

_SYNTHETIC_REVIEWER_TITLE_PATTERNS = (
    re.compile(r"^repeat churn signal", re.I),
    re.compile(r"score:\s*\d", re.I),
)


def is_synthetic_reviewer_title(value: Any) -> bool:
    title = str(value or "").strip()
    if not title:
        return False
    lowered = title.lower()
    return any(pattern.search(lowered) for pattern in _SYNTHETIC_REVIEWER_TITLE_PATTERNS)


def sanitize_reviewer_title(value: Any) -> str | None:
    title = str(value or "").strip()
    if not title or is_synthetic_reviewer_title(title):
        return None
    return title
