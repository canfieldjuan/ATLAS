"""Compatibility wrapper for the shared extracted reasoning archetypes module.

PR-C1h replaced the prior ~590-line local fork with this thin re-export
wrapper. The canonical implementation lives in
`extracted_reasoning_core.archetypes` (consolidated via PR #94 / PR-C1a).

This wrapper preserves the public-import path
(`from extracted_content_pipeline.reasoning.archetypes import ...`) so
existing content_pipeline callers and the
`tests/test_extracted_reasoning_archetypes.py` test suite keep working
unchanged. The public contract type `ArchetypeMatch` is re-exported
from `extracted_reasoning_core.types` (promoted in PR-C1c).
"""

from __future__ import annotations

from extracted_reasoning_core.archetypes import (
    ARCHETYPES,
    MATCH_THRESHOLD,
    ArchetypeProfile,
    SignalRule,
    best_match,
    enrich_evidence_with_archetypes,
    get_archetype,
    get_falsification_conditions,
    score_evidence,
    top_matches,
)
from extracted_reasoning_core.types import ArchetypeMatch


__all__ = [
    "ARCHETYPES",
    "ArchetypeMatch",
    "ArchetypeProfile",
    "MATCH_THRESHOLD",
    "SignalRule",
    "best_match",
    "enrich_evidence_with_archetypes",
    "get_archetype",
    "get_falsification_conditions",
    "score_evidence",
    "top_matches",
]
