"""Pin atlas's re-exports from ``atlas_brain.reasoning.tiers``.

PR-D7b1 promoted the tiers module into
``extracted_reasoning_core.tiers`` and replaced atlas's 190-line fork
with a thin re-export wrapper. Internal callers (autonomous tasks +
tests/test_reasoning_live.py) keep their existing import sites
``from atlas_brain.reasoning.tiers import Tier, gather_tier_context``
working through the wrapper.

These tests use ``is`` identity checks rather than just ``hasattr``:
each alias must point to the exact same object as the core export, so
a future refactor that accidentally redefines a symbol inside
``atlas_brain.reasoning.tiers`` (instead of importing) would surface
here. Mirrors the alias-identity pattern PR-C4e1 established for
``atlas_brain.reasoning.graph``.
"""

from __future__ import annotations

from atlas_brain.reasoning import tiers as atlas_tiers
from extracted_reasoning_core import tiers as core_tiers


def test_tier_alias_identity() -> None:
    assert atlas_tiers.Tier is core_tiers.Tier


def test_tier_config_alias_identity() -> None:
    assert atlas_tiers.TierConfig is core_tiers.TierConfig


def test_tier_configs_alias_identity() -> None:
    # The TIER_CONFIGS dict itself must be the same object so a caller
    # that holds a reference doesn't end up reading from a stale copy.
    assert atlas_tiers.TIER_CONFIGS is core_tiers.TIER_CONFIGS


def test_get_tier_config_alias_identity() -> None:
    assert atlas_tiers.get_tier_config is core_tiers.get_tier_config


def test_build_tiered_pattern_sig_alias_identity() -> None:
    assert atlas_tiers.build_tiered_pattern_sig is core_tiers.build_tiered_pattern_sig


def test_needs_refresh_alias_identity() -> None:
    assert atlas_tiers.needs_refresh is core_tiers.needs_refresh


def test_gather_tier_context_alias_identity() -> None:
    assert atlas_tiers.gather_tier_context is core_tiers.gather_tier_context


def test_atlas_tiers_all_matches_re_export_set() -> None:
    # __all__ is the public-surface contract; pin it so a regression
    # that drops a symbol from the wrapper surfaces a test failure
    # rather than a silent ImportError on an existing caller.
    assert set(atlas_tiers.__all__) == {
        "TIER_CONFIGS",
        "Tier",
        "TierConfig",
        "build_tiered_pattern_sig",
        "gather_tier_context",
        "get_tier_config",
        "needs_refresh",
    }


def test_atlas_tiers_does_not_redefine_symbols() -> None:
    # Defensive: the wrapper module body should not contain any
    # ``def`` / ``class`` re-implementations -- only the re-export
    # imports + __all__ literal. Catches a refactor that accidentally
    # forks the implementation back into atlas-side.
    import ast
    import inspect

    source = inspect.getsource(atlas_tiers)
    tree = ast.parse(source)

    redefinitions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            redefinitions.append(node.name)

    assert redefinitions == [], (
        f"atlas_brain.reasoning.tiers should be a pure re-export wrapper, "
        f"but it defines: {redefinitions}"
    )
