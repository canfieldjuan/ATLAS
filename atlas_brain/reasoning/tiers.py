"""Hierarchical Abstraction Tiers (WS0G).

Defines the 4-tier reasoning hierarchy and inheritance rules:

    T4: Market Dynamics    (quarterly re-reason)
    T3: Category Patterns  (monthly re-reason)
    T2: Vendor Archetypes  (weekly re-reason)
    T1: Vendor State       (daily, deterministic, no LLM)

Lower tiers inherit higher-tier conclusions as priors (context tags) and
never re-derive them. When T2 analyzes "Vendor X (CRM)", it receives T4's
"ai_disruption_pressure: high" tag as input context.

PR-D7b1 promoted this module's body into
:mod:`extracted_reasoning_core.tiers`. Atlas keeps the import surface
``atlas_brain.reasoning.tiers`` as a thin re-export so internal
callers (tests, autonomous tasks) don't need to change import sites --
the audit's "atlas adapts to shared core without behavior drift"
criterion is satisfied. Two minor namespace changes flow from the
move: the debug logger name migrates from ``atlas.reasoning.tiers``
to ``extracted_reasoning_core.tiers`` (correct -- the code is core's),
and ``TierConfig`` is now ``frozen=True`` with ``inherits_from`` as
a ``tuple`` instead of a ``list`` (stricter; no atlas caller mutated
either, verified pre-conversion).
"""

from __future__ import annotations

from extracted_reasoning_core.tiers import (
    TIER_CONFIGS,
    Tier,
    TierConfig,
    build_tiered_pattern_sig,
    gather_tier_context,
    get_tier_config,
    needs_refresh,
)

__all__ = [
    "TIER_CONFIGS",
    "Tier",
    "TierConfig",
    "build_tiered_pattern_sig",
    "gather_tier_context",
    "get_tier_config",
    "needs_refresh",
]
