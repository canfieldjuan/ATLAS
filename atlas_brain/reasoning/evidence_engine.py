"""Declarative evidence evaluation engine (atlas-side wrapper).

PR-D7b3 promoted core's slim conclusions+suppression engine into
:mod:`extracted_reasoning_core.evidence_engine`. Atlas keeps the
import surface ``atlas_brain.reasoning.evidence_engine`` as a
subclass wrapper that mixes in the atlas-owned per-review enrichment
pack from ``atlas_brain.reasoning.review_enrichment``:

  - ``compute_urgency``
  - ``override_pain``
  - ``derive_recommend``
  - ``derive_price_complaint`` (depends on atlas-only
    ``atlas_brain.reasoning.phrase_metadata``)
  - ``derive_budget_authority``

The subclass pattern keeps existing atlas callers
(``b2b_churn_intelligence``, ``_b2b_shared``,
``services/b2b/enrichment_derivation``, the test stubs in
``test_b2b_enrichment.py``) writing ``engine.compute_urgency(...)``
against a single object -- core's slim conclusions/suppression
methods (``evaluate_conclusions`` / ``evaluate_suppression`` /
``get_confidence_tier`` / ``get_confidence_label`` /
``evaluate_conclusion``) come from the core base class; the six
enrichment methods come from this subclass.

Re-exports ``ConclusionResult`` / ``SuppressionResult`` from
:mod:`extracted_reasoning_core.types` -- both shapes are byte-
identical to atlas's pre-PR-D7b3 local dataclasses (verified during
PR-D7b3 drift analysis), so atlas callers reading
``r.conclusion_id`` / ``r.met`` / ``r.confidence`` /
``r.fallback_label`` / ``r.fallback_action`` and ``s.suppress`` /
``s.degrade`` / ``s.disclaimer`` / ``s.fallback_label`` keep
working through the wrapper.

Atlas's factory ``get_evidence_engine`` continues to consult
``settings.b2b_churn.evidence_map_path`` before falling back to the
default YAML beside this module. Core's factory stays config-free
because external products don't ship atlas's settings module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from extracted_reasoning_core.evidence_engine import (
    EvidenceEngine as _CoreEvidenceEngine,
)
from extracted_reasoning_core.types import ConclusionResult, SuppressionResult

from .review_enrichment import ReviewEnrichmentMixin

logger = logging.getLogger("atlas.reasoning.evidence_engine")

_DEFAULT_MAP_PATH = Path(__file__).parent / "evidence_map.yaml"


class EvidenceEngine(ReviewEnrichmentMixin, _CoreEvidenceEngine):
    # Mixin-first MRO keeps enrichment methods on the atlas product pack and
    # out of the extracted core slim engine.
    """Core slim engine extended with atlas-side per-review enrichment."""

    def _init_from_rules(
        self,
        rules: dict[str, Any],
        map_path_str: str,
        raw_bytes: bytes,
    ) -> None:
        super()._init_from_rules(rules, map_path_str, raw_bytes)
        self._init_review_enrichment()


_engine: EvidenceEngine | None = None
_engine_path: str | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    """Return a cached EvidenceEngine, respecting config path.

    If no map_path is provided, checks ``settings.b2b_churn.evidence_map_path``
    before falling back to the default YAML beside this module.

    The singleton is invalidated if the resolved path changes (e.g. config
    reload with a different path).  In-process YAML edits still require a
    ``reload_evidence_engine()`` call or process restart.
    """
    global _engine, _engine_path

    if map_path is None:
        try:
            from ..config import settings
            configured = (settings.b2b_churn.evidence_map_path or "").strip()
            if configured:
                map_path = configured
        except Exception:
            pass

    resolved = str(map_path) if map_path else str(_DEFAULT_MAP_PATH)

    if _engine is not None and _engine_path == resolved:
        return _engine

    try:
        _engine = EvidenceEngine(resolved)
    except FileNotFoundError:
        if resolved != str(_DEFAULT_MAP_PATH):
            logger.error(
                "Evidence map not found at configured path %s, falling back to default",
                resolved,
            )
            resolved = str(_DEFAULT_MAP_PATH)
            _engine = EvidenceEngine(resolved)
        else:
            raise
    _engine_path = resolved
    logger.info(
        "Evidence engine loaded: %s (hash=%s, %d enrichment rules, %d conclusions)",
        resolved,
        _engine.map_hash,
        len(_engine._enrichment),
        len(_engine._conclusions),
    )
    return _engine


def reload_evidence_engine() -> EvidenceEngine:
    """Force-reload the Evidence Map from disk (e.g. after YAML edits)."""
    global _engine, _engine_path
    _engine = None
    _engine_path = None
    return get_evidence_engine()


__all__ = [
    "ConclusionResult",
    "EvidenceEngine",
    "SuppressionResult",
    "get_evidence_engine",
    "reload_evidence_engine",
]
