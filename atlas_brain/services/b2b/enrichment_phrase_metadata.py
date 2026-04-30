from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")

PHRASE_METADATA_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
    "recommendation_language",
    "positive_aspects",
)
PHRASE_SUBJECT_VALUES: tuple[str, ...] = (
    "subject_vendor", "alternative", "self", "third_party", "unclear",
)
PHRASE_POLARITY_VALUES: tuple[str, ...] = (
    "negative", "positive", "mixed", "unclear",
)
PHRASE_ROLE_VALUES: tuple[str, ...] = (
    "primary_driver", "supporting_context", "passing_mention", "unclear",
)
PHRASE_UNCLEAR = "unclear"


@dataclass(frozen=True)
class EnrichmentPhraseMetadataDeps:
    check_phrase_grounded: Any


def coerce_legacy_phrase_arrays(result: dict[str, Any]) -> None:
    for field in PHRASE_METADATA_FIELDS:
        value = result.get(field)
        if not isinstance(value, list):
            result[field] = []
            continue
        coerced: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    coerced.append(text)
            elif isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                if text:
                    coerced.append(text)
        result[field] = coerced


def normalize_tag_value(value: Any, allowed: tuple[str, ...]) -> tuple[str, bool]:
    raw = value if isinstance(value, str) else None
    if raw is None:
        return PHRASE_UNCLEAR, False
    normalized = raw.strip().lower()
    if not normalized:
        return PHRASE_UNCLEAR, False
    if normalized in allowed:
        return normalized, False
    return PHRASE_UNCLEAR, True


def normalize_phrase_metadata(
    result: dict[str, Any],
    source_row: dict[str, Any] | None = None,
    *,
    deps: EnrichmentPhraseMetadataDeps,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_metadata = result.get("phrase_metadata")
    if not isinstance(source_metadata, list):
        source_metadata = []

    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in source_metadata:
        if not isinstance(entry, dict):
            continue
        field = str(entry.get("field") or "")
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if field and idx >= 0:
            source_by_key[(field, idx)] = entry

    counters = {
        "llm_provided_rows": 0,
        "llm_missing_rows": 0,
        "text_mismatch_rows": 0,
        "unknown_tag_coercions": 0,
        "verbatim_grounding_failures": 0,
        "verbatim_grounding_wins": 0,
    }
    canonical: list[dict[str, Any]] = []

    summary_text = (source_row or {}).get("summary") if source_row else None
    review_text = (source_row or {}).get("review_text") if source_row else None
    grounding_enabled = source_row is not None

    for field in PHRASE_METADATA_FIELDS:
        legacy_array = result.get(field) or []
        for index, legacy_text in enumerate(legacy_array):
            llm_row = source_by_key.get((field, index))
            if llm_row is None:
                counters["llm_missing_rows"] += 1
                canonical.append({
                    "field": field,
                    "index": index,
                    "text": str(legacy_text),
                    "subject": PHRASE_UNCLEAR,
                    "polarity": PHRASE_UNCLEAR,
                    "role": PHRASE_UNCLEAR,
                    "verbatim": False,
                    "category_hint": None,
                })
                continue

            counters["llm_provided_rows"] += 1
            subject, sub_coerced = normalize_tag_value(llm_row.get("subject"), PHRASE_SUBJECT_VALUES)
            polarity, pol_coerced = normalize_tag_value(llm_row.get("polarity"), PHRASE_POLARITY_VALUES)
            role, role_coerced = normalize_tag_value(llm_row.get("role"), PHRASE_ROLE_VALUES)
            counters["unknown_tag_coercions"] += int(sub_coerced) + int(pol_coerced) + int(role_coerced)

            verbatim = llm_row.get("verbatim") is True
            hint = llm_row.get("category_hint")
            category_hint = str(hint).strip() if isinstance(hint, str) and str(hint).strip() else None

            llm_text = str(llm_row.get("text") or "").strip()
            if llm_text != str(legacy_text).strip():
                counters["text_mismatch_rows"] += 1
                verbatim = False

            if verbatim and grounding_enabled:
                if deps.check_phrase_grounded(str(legacy_text), summary=summary_text, review_text=review_text):
                    counters["verbatim_grounding_wins"] += 1
                else:
                    counters["verbatim_grounding_failures"] += 1
                    verbatim = False

            canonical.append({
                "field": field,
                "index": index,
                "text": str(legacy_text),
                "subject": subject,
                "polarity": polarity,
                "role": role,
                "verbatim": verbatim,
                "category_hint": category_hint,
            })

    return canonical, counters


def apply_phrase_metadata_contract(
    result: dict[str, Any],
    source_row: dict[str, Any],
    *,
    deps: EnrichmentPhraseMetadataDeps,
) -> None:
    if isinstance(result.get("phrase_metadata"), list):
        canonical_metadata, metadata_counters = normalize_phrase_metadata(
            result, source_row, deps=deps,
        )
        total_legacy_phrases = sum(len(result.get(field) or []) for field in PHRASE_METADATA_FIELDS)
        llm_genuinely_tried = (
            metadata_counters["llm_provided_rows"] > 0
            or total_legacy_phrases == 0
        )
        if llm_genuinely_tried:
            result["phrase_metadata"] = canonical_metadata
            result["enrichment_schema_version"] = 4
            logger.info(
                "phrase_metadata normalized for %s: %s",
                source_row.get("id"),
                metadata_counters,
            )
        else:
            result.pop("phrase_metadata", None)
            result["enrichment_schema_version"] = 3
            logger.warning(
                "phrase_metadata present but unusable for %s "
                "(legacy_phrases=%d, provided_rows=0); falling back to v3",
                source_row.get("id"),
                total_legacy_phrases,
            )
    else:
        result.pop("phrase_metadata", None)
        result["enrichment_schema_version"] = 3
