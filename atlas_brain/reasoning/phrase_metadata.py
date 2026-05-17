"""Helpers for reading the v2 phrase_metadata parallel field.

The v2 enrichment schema (enrichment_schema_version >= 4) adds a top-level
``phrase_metadata`` array alongside the six legacy string arrays
(specific_complaints, pricing_phrases, feature_gaps, quotable_phrases,
recommendation_language, positive_aspects). The legacy arrays retain their
``list[str]`` semantics unchanged; tags live in ``phrase_metadata`` keyed by
``(field, index)``.

These helpers are the supported way to read phrase metadata. They short-circuit
to empty results on v1 enrichments so callers get a safe default path.
"""

from __future__ import annotations

from typing import Any

_V2_SCHEMA_MIN_VERSION = 4
_LEGACY_PHRASE_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
    "recommendation_language",
    "positive_aspects",
)


def enrichment_schema_version(enrichment: dict[str, Any] | None) -> int:
    """Return the integer schema version for an enrichment dict. 0 if missing."""
    if not isinstance(enrichment, dict):
        return 0
    try:
        return int(enrichment.get("enrichment_schema_version") or 0)
    except (TypeError, ValueError):
        return 0


def is_v2_tagged(enrichment: dict[str, Any] | None) -> bool:
    """True if this enrichment carries the v2 phrase_metadata contract.

    A v4 row with ``phrase_metadata = []`` still counts: it represents a
    legitimate zero-phrase review that the LLM tagged correctly under the v2
    schema. Only v<4 or a missing/non-list ``phrase_metadata`` field returns
    False. This keeps schema-aware readers from misclassifying empty-but-valid
    v4 rows as legacy.
    """
    if enrichment_schema_version(enrichment) < _V2_SCHEMA_MIN_VERSION:
        return False
    if not isinstance(enrichment, dict):
        return False
    return isinstance(enrichment.get("phrase_metadata"), list)


def phrase_metadata_by_field(
    enrichment: dict[str, Any] | None,
    field_name: str,
) -> list[dict[str, Any]]:
    """Return metadata rows for ``field_name`` in index order."""
    if not is_v2_tagged(enrichment):
        return []
    metadata = enrichment.get("phrase_metadata") or []
    matches: list[dict[str, Any]] = []
    for row in metadata:
        if not isinstance(row, dict):
            continue
        if str(row.get("field") or "") != field_name:
            continue
        matches.append(row)
    matches.sort(key=lambda r: _safe_index(r.get("index")))
    return matches


def phrase_metadata_map(
    enrichment: dict[str, Any] | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Return a ``{(field, index): row}`` map for constant-time lookup."""
    if not is_v2_tagged(enrichment):
        return {}
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in enrichment.get("phrase_metadata") or []:
        if not isinstance(row, dict):
            continue
        field = str(row.get("field") or "")
        idx = _safe_index(row.get("index"))
        if not field or idx < 0:
            continue
        result[(field, idx)] = row
    return result


def phrase_tag(
    enrichment: dict[str, Any] | None,
    field: str,
    index: int,
    tag_name: str,
    default: Any = None,
) -> Any:
    """Return a single tag value for a specific phrase, or ``default``."""
    rows = phrase_metadata_map(enrichment)
    row = rows.get((field, index))
    if row is None:
        return default
    value = row.get(tag_name)
    return value if value is not None else default


def _safe_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


__all__ = [
    "enrichment_schema_version",
    "is_v2_tagged",
    "phrase_metadata_by_field",
    "phrase_metadata_map",
    "phrase_tag",
]
