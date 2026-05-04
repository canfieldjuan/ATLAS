"""Shared witness-scope UI/report render gates.

The deterministic render-gate logic lives in
``extracted_quality_gate.source_quality_pack`` (PR-B5c). This module
is a thin atlas-side re-export so the existing import path
(``from atlas_brain.services.b2b.witness_render_gate import
apply_witness_render_gate``) keeps working without changes.

See ``extracted_quality_gate.source_quality_pack`` for the contract
and the parametric thresholds the new ``evaluate_source_quality``
pack entry point exposes.
"""

from __future__ import annotations

from extracted_quality_gate.source_quality_pack import (
    apply_witness_render_gate,
    evaluate_source_quality,
)


__all__ = [
    "apply_witness_render_gate",
    "evaluate_source_quality",
]
