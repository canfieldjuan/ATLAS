"""Churn Archetype Definitions and Signal Matching (WS2).

Defines the canonical churn archetypes with signal signatures. The
scorer evaluates vendor evidence (including temporal data from WS1)
against each archetype's expected signal profile and returns ranked
matches. Pure data, no LLM.

PR-D7b5 promoted this module's body into
:mod:`extracted_reasoning_core.archetypes` (per the PR-C1a
consolidation). Atlas keeps the import surface
``atlas_brain.reasoning.archetypes`` as a thin re-export wrapper so
internal callers (``b2b_churn_intelligence``, ``_b2b_shared``,
``test_reasoning_live``, ``test_archetype_propagation``) keep their
existing import sites working.

Drift handling: core deliberately split ``ArchetypeMatch`` into two
shapes per PR-C1a's design.

- ``_ArchetypeMatchInternal`` (in ``core.archetypes``) -- the rich
  internal result with atlas's original field names (``archetype`` /
  ``score`` / ``matched_signals`` / ``missing_signals`` /
  ``risk_level``). Core's ``score_evidence`` / ``best_match`` /
  ``top_matches`` / ``enrich_evidence_with_archetypes`` all return
  this type. Atlas-side callers continue to consume it directly.

- ``ArchetypeMatch`` (in ``core.types``) -- the canonical public
  contract with renamed fields (``archetype_id`` / ``label`` /
  ``evidence_hits`` / ``missing_evidence`` / ``risk_label``). Only
  reachable through ``core.api.score_archetypes`` for external
  products that need the stable surface.

This wrapper aliases atlas's ``ArchetypeMatch`` to the rich internal
type so existing atlas code reading ``.archetype`` / ``.matched_signals``
/ ``.risk_level`` keeps working without translation. The canonical
``core.types.ArchetypeMatch`` is intentionally NOT exposed here -- if
an atlas caller ever needs the public shape, it should reach through
``core.api`` like any other product.
"""

from __future__ import annotations

from extracted_reasoning_core.archetypes import (
    ARCHETYPES,
    MATCH_THRESHOLD,
    ArchetypeProfile,
    SignalRule,
    _ArchetypeMatchInternal as ArchetypeMatch,
    best_match,
    enrich_evidence_with_archetypes,
    get_archetype,
    get_falsification_conditions,
    score_evidence,
    top_matches,
)

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
