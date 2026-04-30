from __future__ import annotations

import json
from typing import Any

from ...config import settings
from ...services.scraping.sources import parse_source_allowlist

MIN_REVIEW_TEXT_LENGTH = 80
_TIER2_INSIDER_SECTION_HEADER = "### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)"
_TIER2_OUTPUT_SECTION_HEADER = "## Output"


def coerce_int_value(raw_value: Any, fallback: int) -> int:
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if raw_value != raw_value:
            return fallback
        return int(raw_value)
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return fallback
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return fallback
    return fallback


def config_allowlist(
    raw_value: Any,
    fallback: str | list[str] | tuple[str, ...] | set[str] | frozenset[str] = "",
) -> list[str]:
    candidate = raw_value if isinstance(raw_value, (str, list, tuple, set, frozenset)) else fallback
    return list(parse_source_allowlist(candidate))


def smart_truncate(text: str, max_len: int = 3000) -> str:
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


def build_classify_payload(
    row: dict[str, Any],
    *,
    truncate_length: int = 3000,
    smart_truncate: Any,
) -> dict[str, Any]:
    review_text = smart_truncate(row["review_text"] or "", max_len=truncate_length)

    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}

    payload: dict[str, Any] = {
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"] or "",
        "product_category": row["product_category"] or "",
        "source_name": row.get("source") or "",
        "source_weight": raw_meta.get("source_weight", 0.7),
        "source_type": raw_meta.get("source_type", "unknown"),
        "content_type": row.get("content_type") or "review",
        "rating": float(row["rating"]) if row["rating"] is not None else None,
        "rating_max": int(row["rating_max"]),
        "summary": row["summary"] or "",
        "review_text": review_text,
    }
    for key, value in (
        ("pros", row["pros"]),
        ("cons", row["cons"]),
        ("reviewer_title", row["reviewer_title"]),
        ("reviewer_company", row["reviewer_company"]),
        ("company_size_raw", row["company_size_raw"]),
        ("reviewer_industry", row["reviewer_industry"]),
    ):
        if isinstance(value, str) and value.strip():
            payload[key] = value
    if not payload["product_name"]:
        payload.pop("product_name", None)
    if not payload["product_category"]:
        payload.pop("product_category", None)
    if payload.get("rating") is None:
        payload.pop("rating", None)
    if not payload["summary"]:
        payload.pop("summary", None)
    if not payload["review_text"]:
        payload.pop("review_text", None)
    return payload


def tier2_system_prompt_for_content_type(prompt: str, content_type: str | None) -> str:
    if str(content_type or "").strip().lower() == "insider_account":
        return prompt
    before, marker, after = prompt.partition(_TIER2_INSIDER_SECTION_HEADER)
    if not marker:
        return prompt
    _insider_body, output_marker, output_tail = after.partition(_TIER2_OUTPUT_SECTION_HEADER)
    if not output_marker:
        return prompt
    return f"{before.rstrip()}\n\n{_TIER2_OUTPUT_SECTION_HEADER}{output_tail}"


def combined_review_text_length(row: dict[str, Any] | None) -> int:
    row = row or {}
    return sum(
        len(str(row.get(field) or ""))
        for field in ("review_text", "pros", "cons")
    )


def effective_min_review_text_length(row: dict[str, Any] | None) -> int:
    source = str((row or {}).get("source") or "").strip().lower()
    if source == "capterra":
        try:
            return min(
                MIN_REVIEW_TEXT_LENGTH,
                max(
                    20,
                    int(
                        getattr(
                            settings.b2b_scrape,
                            "capterra_min_enrichable_text_len",
                            40,
                        ) or 40
                    ),
                ),
            )
        except (TypeError, ValueError):
            return 40
    return MIN_REVIEW_TEXT_LENGTH


def effective_enrichment_skip_sources() -> set[str]:
    configured = config_allowlist(getattr(settings.b2b_churn, "enrichment_skip_sources", ""), "")
    deprecated = config_allowlist(
        getattr(settings.b2b_churn, "deprecated_review_sources", ""),
        "capterra,software_advice,trustpilot,trustradius",
    )
    return {
        source
        for source in [*configured, *deprecated]
        if source
    }


def tier1_has_extraction_gaps(tier1: dict[str, Any], *, source: str | None = None) -> bool:
    complaints = tier1.get("specific_complaints") or []
    quotes = tier1.get("quotable_phrases") or []
    competitors = tier1.get("competitors_mentioned") or []
    pricing = tier1.get("pricing_phrases") or []
    rec_lang = tier1.get("recommendation_language") or []
    churn = tier1.get("churn_signals") or {}
    has_churn = any(bool(v) for v in churn.values())
    has_evidence = bool(complaints or quotes or competitors or pricing or rec_lang)
    source_norm = str(source or "").strip().lower()
    strict_sources = set(
        config_allowlist(
            getattr(settings.b2b_churn, "enrichment_tier2_strict_sources", ""),
            "",
        )
    )
    if source_norm in strict_sources:
        complaint_count = len(complaints) if isinstance(complaints, list) else 0
        quote_count = len(quotes) if isinstance(quotes, list) else 0
        evidence_groups = sum(
            1
            for present in (
                bool(complaints),
                bool(quotes),
                bool(competitors),
                bool(pricing),
                bool(rec_lang),
            )
            if present
        )
        has_strong_structured_evidence = (
            bool(competitors)
            or bool(pricing)
            or complaint_count >= coerce_int_value(
                getattr(settings.b2b_churn, "enrichment_tier2_strict_min_complaints", 2),
                2,
            )
            or (
                quote_count >= coerce_int_value(
                    getattr(settings.b2b_churn, "enrichment_tier2_strict_min_quotes", 2),
                    2,
                )
                and evidence_groups >= 2
            )
        )
        return has_churn or has_strong_structured_evidence
    return has_churn or has_evidence
