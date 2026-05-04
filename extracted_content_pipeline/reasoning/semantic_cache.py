"""Compatibility wrapper for the semantic cache surface in this package.

PR-C2.1 (follow-up to PR-C2 #144). The atlas-side
``atlas_brain/autonomous/tasks/_b2b_cross_vendor_synthesis.py`` does
``from ...reasoning.semantic_cache import compute_evidence_hash``,
which resolves to ``atlas_brain.reasoning.semantic_cache`` in atlas
but to ``extracted_content_pipeline.reasoning.semantic_cache`` in the
mirror. Without this wrapper module the mirror import blew up with
``ModuleNotFoundError`` -- a regression introduced when PR-A5c (#142)
re-pointed that file to the atlas-side helper and PR-C2 (#144) bundled
the mirror sync without adding the missing wrapper file.

Re-exports match the surface of the
``extracted_competitive_intelligence/reasoning/semantic_cache.py``
sibling wrapper so callers of either content_pipeline or
competitive_intelligence get the same names.
"""

from __future__ import annotations

from extracted_llm_infrastructure.reasoning.semantic_cache import (
    SemanticCache,
    SemanticCachePool,
)
from extracted_reasoning_core.semantic_cache_keys import (
    CacheEntry,
    STALE_THRESHOLD,
    apply_decay,
    compute_evidence_hash,
    row_to_cache_entry,
)


__all__ = [
    "CacheEntry",
    "STALE_THRESHOLD",
    "SemanticCache",
    "SemanticCachePool",
    "apply_decay",
    "compute_evidence_hash",
    "row_to_cache_entry",
]
