"""Compatibility wrapper for phrase metadata helpers.

Canonical helpers live in ``atlas_brain.reasoning.phrase_metadata``.

.. deprecated::
    Import from ``atlas_brain.reasoning.phrase_metadata`` directly.
    This module will be removed once all callers have migrated.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "atlas_brain.autonomous.tasks._b2b_phrase_metadata is deprecated; "
    "import from atlas_brain.reasoning.phrase_metadata instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
