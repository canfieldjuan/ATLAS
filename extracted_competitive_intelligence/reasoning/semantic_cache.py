"""Semantic cache surface for the competitive-intelligence package.

PR-C2 (PR 4 from the reasoning boundary audit) closed the audit's
acceptance criterion #3: "competitive intelligence no longer bridges
Atlas semantic cache". This module previously did a programmatic
``importlib`` re-export of every name from
``atlas_brain.reasoning.semantic_cache``; it now imports from the
canonical homes directly:

  - pure primitives -> ``extracted_reasoning_core.semantic_cache_keys``
  - Postgres storage -> ``extracted_llm_infrastructure.reasoning.semantic_cache``

The public-import path (``from
extracted_competitive_intelligence.reasoning.semantic_cache import
SemanticCache, CacheEntry, compute_evidence_hash``) is preserved so
existing callers in ``extracted_competitive_intelligence/autonomous/tasks/``
keep working unchanged.
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
