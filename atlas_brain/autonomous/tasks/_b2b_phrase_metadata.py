"""Compatibility wrapper for phrase metadata helpers.

Canonical helpers live in ``atlas_brain.reasoning.phrase_metadata``.
"""

from __future__ import annotations

from atlas_brain.reasoning.phrase_metadata import (
    enrichment_schema_version,
    is_v2_tagged,
    phrase_metadata_by_field,
    phrase_metadata_map,
    phrase_tag,
)

__all__ = [
    "enrichment_schema_version",
    "is_v2_tagged",
    "phrase_metadata_by_field",
    "phrase_metadata_map",
    "phrase_tag",
]
