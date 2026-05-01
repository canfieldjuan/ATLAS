from __future__ import annotations

from typing import Any


def pain_category_for_bucket(bucket: str | None) -> str:
    return str(bucket or "other").strip().lower() or "other"


def quote_grade_phrases(*args: Any, **kwargs: Any) -> list[str]:
    return []


def resolve_pain_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
