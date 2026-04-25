"""Use-case-specific contract for reading B2B enrichment rows.

This module is the supported entry point for downstream consumers
(reports, dashboards, MCP tools, alerts) to ask narrative questions
about an enrichment dict without reaching into the JSONB themselves.

It wraps two existing internal pieces:

  - ``atlas_brain.autonomous.tasks._b2b_phrase_metadata`` -- low-level
    accessors for the v2 ``phrase_metadata`` parallel field. Re-exported
    here so consumers have one import path.

  - ``_ensure_pain_confidence`` from ``_b2b_witnesses.py`` -- the legacy
    fallback that recomputes pain_confidence for v3 rows missing the
    field. Reimplemented here as ``resolve_pain_confidence`` and the
    witness module imports it back via a tiny shim.

The contract distinguishes three use cases for pain_category:

  - HEADLINE (battle card titles, blog headlines, briefing primary
    drivers): strict ``confidence='strong'`` and non-generic by default.
  - BUCKET (vendor pain rollups, dashboard counts, company signals):
    permits ``'weak'`` and generic categories so aggregation surfaces
    do not lose rows.
  - DISPLAY (admin/debug): returns whatever resolved, including the
    'unknown' tier for malformed rows.

Phrase reads are split:

  - QUOTE-GRADE: subject_vendor + negative/mixed + verbatim. Only v4
    rows can produce these. Use for any pull-quote rendered in UI,
    email, blog, or report.
  - THEME-GRADE: subject_vendor + negative/mixed (verbatim NOT required).
    Use for theme aggregation, topic clustering, headline generation.
    Optionally includes legacy v3 phrases stamped with
    ``legacy=True, verbatim=False, subject='unknown', polarity='unknown'``
    so the pre-v4 corpus is not lost as theme material.
"""

from __future__ import annotations

from typing import Any, Literal

from ...autonomous.tasks._b2b_phrase_metadata import (
    enrichment_schema_version,
    is_v2_tagged,
    phrase_metadata_by_field,
    phrase_metadata_map,
    phrase_tag,
)

ConfidenceTier = Literal["strong", "weak", "none", "unknown"]

# Categories that indicate "we have a pain signal but no specific driver".
# Headlines, blog post titles, and similar surfaces should NOT treat these
# as authoritative narrative claims. ``overall_dissatisfaction`` is the
# canonical v4 catch-all; older names appear in v3 and pre-rewrite rows.
GENERIC_PAIN_CATEGORIES: frozenset[str] = frozenset({
    "overall_dissatisfaction",
    "general_dissatisfaction",
    "other",
    "unknown",
})

_LEGACY_PHRASE_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
    "recommendation_language",
    "positive_aspects",
)

_NEGATIVE_POLARITIES: frozenset[str] = frozenset({"negative", "mixed"})


# ---------------------------------------------------------------------------
# Pain-category helpers
# ---------------------------------------------------------------------------


def is_generic_pain(category: str | None) -> bool:
    """Return True iff ``category`` is one of GENERIC_PAIN_CATEGORIES or empty.

    Use to gate against treating the catch-all as a specific narrative
    claim. Consumers building "Pricing pain" or "Reliability pain" surfaces
    should refuse to render anything where this returns True.
    """
    if category is None:
        return True
    text = str(category).strip().lower()
    if not text:
        return True
    return text in GENERIC_PAIN_CATEGORIES


def resolve_pain_confidence(enrichment: dict[str, Any] | None) -> ConfidenceTier:
    """Return resolved pain_confidence with legacy fallback for v3 rows.

    Reads ``enrichment['pain_confidence']`` when present and one of the
    canonical values ('strong'/'weak'/'none'). Otherwise lazy-imports the
    deterministic recompute from ``b2b_enrichment`` so older v3 rows missing
    the field still produce a usable tier.

    Returns 'unknown' only when the recompute is unavailable (missing
    enrichment dict, malformed structure, or import failure). 'unknown'
    means "we cannot judge", not "legacy".
    """
    if not isinstance(enrichment, dict):
        return "unknown"
    existing = enrichment.get("pain_confidence")
    if isinstance(existing, str) and existing in ("strong", "weak", "none"):
        return existing  # type: ignore[return-value]
    try:
        # Lazy import: avoids circular dep with b2b_enrichment, which itself
        # imports witness helpers. Same pattern as _ensure_pain_confidence.
        from ...autonomous.tasks.b2b_enrichment import (
            _compute_pain_confidence,
            _normalize_pain_category,
        )
    except Exception:  # pragma: no cover - defensive
        return "unknown"
    pain = _normalize_pain_category(enrichment.get("pain_category"))
    try:
        recomputed = _compute_pain_confidence(enrichment, pain)
    except Exception:  # pragma: no cover - defensive
        return "unknown"
    if isinstance(recomputed, str) and recomputed in ("strong", "weak", "none"):
        return recomputed  # type: ignore[return-value]
    return "unknown"


def pain_category_for_headline(
    enrichment: dict[str, Any] | None,
    *,
    allow_generic: bool = False,
) -> str | None:
    """Strict gate for top-line claims.

    Returns ``pain_category`` iff:
      - resolve_pain_confidence == 'strong'
      - allow_generic=True OR NOT is_generic_pain(pain_category)

    Headlines NEVER accept the 'unknown' tier or weak confidence, so this
    helper deliberately has no allow_unknown_confidence knob. If a row
    cannot resolve to 'strong', it is not headline-worthy. Callers must
    not synthesize a generic placeholder when this returns None.
    """
    if not isinstance(enrichment, dict):
        return None
    confidence = resolve_pain_confidence(enrichment)
    if confidence != "strong":
        return None
    raw = enrichment.get("pain_category")
    if raw is None:
        return None
    category = str(raw).strip()
    if not category:
        return None
    if not allow_generic and is_generic_pain(category):
        return None
    return category


def pain_category_for_bucket(
    enrichment: dict[str, Any] | None,
    *,
    min_confidence: Literal["strong", "weak"] = "weak",
    allow_generic: bool = True,
    allow_unknown_confidence: bool = False,
) -> str | None:
    """Looser gate for aggregation surfaces.

    Returns ``pain_category`` iff:
      - resolved confidence meets ``min_confidence`` floor (default 'weak')
      - allow_generic=True OR NOT is_generic_pain(pain_category)
      - allow_unknown_confidence=True OR confidence != 'unknown'

    Defaults permit weak signal and generic categories so vendor pain
    rollups, company signals, and dashboard counts do not lose rows.
    Tighten via ``min_confidence='strong'`` for surfaces that need it.
    """
    if not isinstance(enrichment, dict):
        return None
    confidence = resolve_pain_confidence(enrichment)
    raw = enrichment.get("pain_category")
    if raw is None:
        return None
    category = str(raw).strip()
    if not category:
        return None
    if confidence == "unknown":
        if not allow_unknown_confidence:
            return None
    elif min_confidence == "strong" and confidence != "strong":
        return None
    elif min_confidence == "weak" and confidence not in ("strong", "weak"):
        return None
    if not allow_generic and is_generic_pain(category):
        return None
    return category


def pain_category_for_display(
    enrichment: dict[str, Any] | None,
) -> tuple[str | None, ConfidenceTier]:
    """Debug/admin path: return (category_or_None, resolved_confidence).

    Includes the 'unknown' tier without filtering. For rendering raw-state
    views without policy. Caller is expected to format the tier explicitly
    (e.g. "Pricing (unknown confidence)" rather than relying on a default).
    """
    if not isinstance(enrichment, dict):
        return None, "unknown"
    confidence = resolve_pain_confidence(enrichment)
    raw = enrichment.get("pain_category")
    if raw is None:
        return None, confidence
    category = str(raw).strip() or None
    return category, confidence


# ---------------------------------------------------------------------------
# Phrase helpers
# ---------------------------------------------------------------------------


def is_about_subject_vendor(phrase_meta: dict[str, Any] | None) -> bool:
    """phrase_subject == 'subject_vendor'. Defaults False on missing."""
    if not isinstance(phrase_meta, dict):
        return False
    return str(phrase_meta.get("subject") or "").strip().lower() == "subject_vendor"


def is_negative_phrase(phrase_meta: dict[str, Any] | None) -> bool:
    """phrase_polarity in {'negative', 'mixed'}. Defaults False on missing."""
    if not isinstance(phrase_meta, dict):
        return False
    return str(phrase_meta.get("polarity") or "").strip().lower() in _NEGATIVE_POLARITIES


def is_primary_driver(phrase_meta: dict[str, Any] | None) -> bool:
    """phrase_role == 'primary_driver'. Defaults False on missing."""
    if not isinstance(phrase_meta, dict):
        return False
    return str(phrase_meta.get("role") or "").strip().lower() == "primary_driver"


def is_verbatim(phrase_meta: dict[str, Any] | None) -> bool:
    """phrase_verbatim is True. Defaults False -- v3 phrases never verbatim."""
    if not isinstance(phrase_meta, dict):
        return False
    return phrase_meta.get("verbatim") is True


def _v4_phrase_passes_gates(
    phrase_meta: dict[str, Any],
    *,
    require_verbatim: bool,
    require_primary: bool,
) -> bool:
    if not is_about_subject_vendor(phrase_meta):
        return False
    if not is_negative_phrase(phrase_meta):
        return False
    if require_primary and not is_primary_driver(phrase_meta):
        return False
    if require_verbatim and not is_verbatim(phrase_meta):
        return False
    return True


def _legacy_phrase_rows(enrichment: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthesize phrase rows from legacy v3 string arrays.

    Each row is stamped with ``legacy=True``, ``verbatim=False``,
    ``subject='unknown'``, ``polarity='unknown'``, and ``role='unknown'``
    so downstream code can identify it as untagged theme material that
    must not be quoted. Empty enrichment or zero legacy phrases returns [].
    """
    rows: list[dict[str, Any]] = []
    for field in _LEGACY_PHRASE_FIELDS:
        raw = enrichment.get(field)
        if not isinstance(raw, list):
            continue
        for idx, value in enumerate(raw):
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            rows.append({
                "field": field,
                "index": idx,
                "text": text,
                "legacy": True,
                "verbatim": False,
                "subject": "unknown",
                "polarity": "unknown",
                "role": "unknown",
            })
    return rows


def quote_grade_phrases(
    enrichment: dict[str, Any] | None,
    *,
    field: str | None = None,
    require_primary: bool = False,
) -> list[dict[str, Any]]:
    """Phrases safe to render as quotes in any consumer surface.

    All gates applied: subject_vendor + negative/mixed + verbatim
    (+optionally primary_driver). v3-era enrichments return [] -- legacy
    phrases are NEVER quote-grade because we have no verbatim guarantee.

    ``field=None`` aggregates across all phrase fields; pass a specific
    field name (e.g. 'pricing_phrases') to scope the read.
    """
    if not isinstance(enrichment, dict):
        return []
    if not is_v2_tagged(enrichment):
        return []
    if field is not None:
        rows = phrase_metadata_by_field(enrichment, field)
    else:
        rows = list((enrichment.get("phrase_metadata") or []))
        rows = [r for r in rows if isinstance(r, dict)]
    return [
        r for r in rows
        if _v4_phrase_passes_gates(
            r,
            require_verbatim=True,
            require_primary=require_primary,
        )
    ]


def theme_grade_phrases(
    enrichment: dict[str, Any] | None,
    *,
    field: str | None = None,
    require_primary: bool = False,
    include_legacy: bool = False,
) -> list[dict[str, Any]]:
    """Phrases usable as theme/topic material but NOT as quotes.

    For v4 rows: subject_vendor + negative/mixed gates applied; verbatim
    NOT required. ``require_primary=True`` adds the primary_driver gate.

    For v3 legacy rows: included only when ``include_legacy=True``. Each
    legacy row is stamped with ``legacy=True, verbatim=False,
    subject='unknown', polarity='unknown', role='unknown'`` so the caller
    can distinguish them from v4 rows and never render them as quotes.
    Subject/polarity gates are skipped for legacy rows because v3 has no
    such tags -- the alternative would be to drop them, which loses theme
    material. Caller opts in explicitly via include_legacy.

    ``field=None`` aggregates across all phrase fields.
    """
    if not isinstance(enrichment, dict):
        return []
    results: list[dict[str, Any]] = []
    if is_v2_tagged(enrichment):
        if field is not None:
            v4_rows = phrase_metadata_by_field(enrichment, field)
        else:
            v4_rows = [
                r for r in (enrichment.get("phrase_metadata") or [])
                if isinstance(r, dict)
            ]
        for row in v4_rows:
            if _v4_phrase_passes_gates(
                row,
                require_verbatim=False,
                require_primary=require_primary,
            ):
                results.append(row)
        return results
    if include_legacy:
        legacy_rows = _legacy_phrase_rows(enrichment)
        if field is not None:
            legacy_rows = [r for r in legacy_rows if r["field"] == field]
        # require_primary cannot be honored for legacy rows (no role tag);
        # they are excluded when require_primary=True so the caller does
        # not get a row that fails its own gate.
        if require_primary:
            return []
        results.extend(legacy_rows)
    return results


__all__ = [
    "ConfidenceTier",
    "GENERIC_PAIN_CATEGORIES",
    "is_generic_pain",
    "resolve_pain_confidence",
    "pain_category_for_headline",
    "pain_category_for_bucket",
    "pain_category_for_display",
    "is_about_subject_vendor",
    "is_negative_phrase",
    "is_primary_driver",
    "is_verbatim",
    "quote_grade_phrases",
    "theme_grade_phrases",
    # Re-exports of low-level primitives so consumers have one import path
    "enrichment_schema_version",
    "is_v2_tagged",
    "phrase_metadata_by_field",
    "phrase_metadata_map",
    "phrase_tag",
]
