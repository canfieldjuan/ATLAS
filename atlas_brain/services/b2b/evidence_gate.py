"""Evidence-claim coverage gate for campaign generation (Gap 3, shadow mode).

The deterministic implementation lives in
``extracted_quality_gate.evidence_pack`` (PR-B5a). This module is a
thin atlas-side re-export so the existing caller path
(``from atlas_brain.services.b2b.evidence_gate import
audit_witness_evidence_coverage``) keeps working without changes.

See ``extracted_quality_gate.evidence_pack`` for the contract and
the parametric coverage thresholds the new
``evaluate_evidence_coverage`` pack entry point exposes.
"""

from __future__ import annotations

from extracted_quality_gate.evidence_pack import (
    _PAIN_CONFIDENCE_RANK_FLOOR,
    _rank_floor,
    audit_witness_evidence_coverage,
    evaluate_evidence_coverage,
)


__all__ = [
    "audit_witness_evidence_coverage",
    "evaluate_evidence_coverage",
]
