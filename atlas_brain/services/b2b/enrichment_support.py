from __future__ import annotations

import json
import re
from typing import Any


def normalize_text_list(values: Any) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        if value:
            normalized.append(str(value))
    return normalized


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = text.lower()
    return any(needle in haystack for needle in needles)


def normalize_compare_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_name_tokens(value: Any, *, low_fidelity_token_stopwords: set[str]) -> list[str]:
    normalized = normalize_compare_text(value)
    if not normalized:
        return []
    return [
        token for token in normalized.split()
        if len(token) >= 3 and token not in low_fidelity_token_stopwords
    ]


def text_mentions_name(haystack: str, needle: Any, *, low_fidelity_token_stopwords: set[str]) -> bool:
    normalized = normalize_compare_text(needle)
    if not normalized:
        return False
    wrapped = f" {haystack} "
    if f" {normalized} " in wrapped:
        return True
    compact_haystack = haystack.replace(" ", "")
    compact_needle = normalized.replace(" ", "")
    if compact_needle and compact_needle in compact_haystack:
        return True
    return any(
        re.search(rf"\b{re.escape(token)}\b", haystack)
        for token in normalized_name_tokens(needle, low_fidelity_token_stopwords=low_fidelity_token_stopwords)
    )


def dedupe_reason_codes(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def has_commercial_context(text: str, *, low_fidelity_commercial_markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in low_fidelity_commercial_markers)


def has_strong_commercial_context(
    text: str,
    *,
    low_fidelity_strong_commercial_markers: tuple[str, ...],
) -> bool:
    return any(marker in text for marker in low_fidelity_strong_commercial_markers)


def has_technical_context(
    summary_text: str,
    combined_text: str,
    *,
    low_fidelity_technical_patterns: tuple[str, ...],
) -> bool:
    if summary_text.endswith("?"):
        return True
    haystack = f"{summary_text} {combined_text}".strip()
    return any(re.search(pattern, haystack) for pattern in low_fidelity_technical_patterns)


def has_consumer_context(text: str, *, low_fidelity_consumer_patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in low_fidelity_consumer_patterns)


def normalized_low_fidelity_noisy_sources(
    configured_raw: Any,
    default_raw: Any,
) -> set[str]:
    configured = {
        item.strip().lower()
        for item in str(configured_raw or "").split(",")
        if item.strip()
    }
    default_values = {
        item.strip().lower()
        for item in str(default_raw or "").split(",")
        if item.strip()
    }
    return configured | default_values


def coerce_bool(value: Any) -> bool | None:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0", "null", "none"):
            return False
    return None


def combined_source_text(source_row: dict[str, Any] | None) -> str:
    if not isinstance(source_row, dict):
        return ""
    parts = [
        str(source_row.get("summary") or ""),
        str(source_row.get("review_text") or ""),
        str(source_row.get("pros") or ""),
        str(source_row.get("cons") or ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def is_unknownish(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "unknown", "none", "null", "n/a", "na"}


def coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
