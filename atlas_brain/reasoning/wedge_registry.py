"""Sales-oriented wedge types for B2B reasoning synthesis.

A 'wedge' is the opening angle a sales rep uses to engage a churning
vendor's customers. Wedges map 1:1 from the base churn archetypes used
by synthesis and deterministic evidence scoring, plus two compound
patterns and a stable fallback.

The wedge registry is the single source of truth for:
- Valid wedge enum values (prompt injection + post-LLM validation)
- Archetype-to-wedge mapping
- Sales motion guidance per wedge
- Required pool layers per wedge

PR-D7b2 promoted this module's body into
:mod:`extracted_reasoning_core.wedge_registry`. Atlas keeps the import
surface ``atlas_brain.reasoning.wedge_registry`` as a thin re-export
so internal callers (B2B synthesis tasks, blog post generation, the
synthesis_v2 test suite) don't need to change import sites -- the
audit's "atlas adapts to shared core without behavior drift"
criterion is satisfied. Drift was cosmetic only: core gained an
``__all__`` declaration and trimmed two inline comments + one
docstring sentence; the eight public symbols (Wedge, WedgeMeta,
WEDGE_ENUM_VALUES, wedge_from_archetype, validate_wedge,
get_wedge_meta, get_sales_motion, get_required_pools) are
implementation-identical.
"""

from __future__ import annotations

from extracted_reasoning_core.wedge_registry import (
    WEDGE_ENUM_VALUES,
    Wedge,
    WedgeMeta,
    get_required_pools,
    get_sales_motion,
    get_wedge_meta,
    validate_wedge,
    wedge_from_archetype,
)

__all__ = [
    "WEDGE_ENUM_VALUES",
    "Wedge",
    "WedgeMeta",
    "get_required_pools",
    "get_sales_motion",
    "get_wedge_meta",
    "validate_wedge",
    "wedge_from_archetype",
]
