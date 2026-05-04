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

Standalone-mode handling: ``content_pipeline`` ships an
``EXTRACTED_PIPELINE_STANDALONE=1`` mode (see
``extracted_content_pipeline/pipelines/llm.py``,
``services/b2b/anthropic_batch.py``) where peer modules avoid
importing ``extracted_llm_infrastructure``. Following that pattern
here: the LLM-infra import is gated behind the same env var so this
module stays importable even when ``extracted_llm_infrastructure``
isn't installed -- callers that only need the pure helpers
(``compute_evidence_hash``, ``CacheEntry``, etc.) still work.
``SemanticCache`` and ``SemanticCachePool`` are stubbed out in
standalone mode so attribute access at import time doesn't crash; any
runtime use raises ``RuntimeError`` with a clear message.

Re-exports match the surface of the
``extracted_competitive_intelligence/reasoning/semantic_cache.py``
sibling wrapper so callers of either content_pipeline or
competitive_intelligence get the same names.
"""

from __future__ import annotations

import os
from typing import Any

from extracted_reasoning_core.semantic_cache_keys import (
    CacheEntry,
    STALE_THRESHOLD,
    apply_decay,
    compute_evidence_hash,
    row_to_cache_entry,
)


if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    # Standalone install: LLM infrastructure isn't guaranteed to be
    # present. Keep the module importable for the pure helpers above;
    # provide stubs that raise on use rather than at import time.

    class SemanticCachePool:  # type: ignore[no-redef]
        """Stub: standalone content_pipeline ships without LLM-infra storage.

        Construction is allowed (so type annotations / Protocol-like
        usage doesn't crash); any await on the methods raises.
        """

        async def fetchrow(self, query: str, *args: Any) -> Any:
            raise RuntimeError(
                "SemanticCachePool requires extracted_llm_infrastructure; "
                "unset EXTRACTED_PIPELINE_STANDALONE or install the package."
            )

        async def fetch(self, query: str, *args: Any) -> Any:
            raise RuntimeError(
                "SemanticCachePool requires extracted_llm_infrastructure; "
                "unset EXTRACTED_PIPELINE_STANDALONE or install the package."
            )

        async def execute(self, query: str, *args: Any) -> Any:
            raise RuntimeError(
                "SemanticCachePool requires extracted_llm_infrastructure; "
                "unset EXTRACTED_PIPELINE_STANDALONE or install the package."
            )

    class SemanticCache:  # type: ignore[no-redef]
        """Stub: storage requires extracted_llm_infrastructure."""

        STALE_THRESHOLD = STALE_THRESHOLD

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "SemanticCache requires extracted_llm_infrastructure; "
                "unset EXTRACTED_PIPELINE_STANDALONE or install the package."
            )

else:
    from extracted_llm_infrastructure.reasoning.semantic_cache import (  # noqa: F401
        SemanticCache,
        SemanticCachePool,
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
