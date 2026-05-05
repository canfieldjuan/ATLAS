"""Pin atlas's re-exports from ``atlas_brain.reasoning.temporal``.

PR-D7b4 promoted the temporal module into
``extracted_reasoning_core.temporal`` (engine + constants) and
``extracted_reasoning_core.types`` (the five dataclasses, frozen+slots
since PR-C1c). Atlas keeps the import surface
``atlas_brain.reasoning.temporal`` as a thin re-export wrapper so
internal callers (b2b_churn_intelligence, _b2b_shared, market_pulse,
the reasoning_temporal / reasoning_market_pulse / reasoning_live test
suites) keep their existing import sites working.

Mirrors the alias-identity pattern PR-D7b1/b2 established for the
tiers and wedge_registry wrappers.
"""

from __future__ import annotations

from atlas_brain.reasoning import temporal as atlas_temporal
from extracted_reasoning_core import temporal as core_temporal
from extracted_reasoning_core import types as core_types


def test_temporal_engine_alias_identity() -> None:
    assert atlas_temporal.TemporalEngine is core_temporal.TemporalEngine


def test_temporal_evidence_alias_identity() -> None:
    # The five dataclasses live in core.types (per PR-C1c). Atlas's
    # wrapper re-exports them from there, so the identity check must
    # target core.types -- not core.temporal.
    assert atlas_temporal.TemporalEvidence is core_types.TemporalEvidence


def test_vendor_velocity_alias_identity() -> None:
    assert atlas_temporal.VendorVelocity is core_types.VendorVelocity


def test_long_term_trend_alias_identity() -> None:
    assert atlas_temporal.LongTermTrend is core_types.LongTermTrend


def test_category_percentile_alias_identity() -> None:
    assert atlas_temporal.CategoryPercentile is core_types.CategoryPercentile


def test_anomaly_score_alias_identity() -> None:
    assert atlas_temporal.AnomalyScore is core_types.AnomalyScore


def test_constants_alias_identity() -> None:
    # Constants are immutable scalars so ``is`` may or may not hold
    # depending on Python's int/string interning. Use ``==`` for
    # values, but pin exact values too so a regression in core's
    # canonical defaults surfaces here.
    assert atlas_temporal.MIN_DAYS_FOR_VELOCITY == core_temporal.MIN_DAYS_FOR_VELOCITY == 2
    assert (
        atlas_temporal.MIN_DAYS_FOR_ACCELERATION
        == core_temporal.MIN_DAYS_FOR_ACCELERATION
        == 3
    )
    # MIN_DAYS_FOR_PERCENTILES was canonicalized to 3 in PR-C1b
    # (atlas had constant=7 alongside hardcoded gate=3; core picked
    # the actual runtime value).
    assert (
        atlas_temporal.MIN_DAYS_FOR_PERCENTILES
        == core_temporal.MIN_DAYS_FOR_PERCENTILES
        == 3
    )
    assert atlas_temporal.MIN_DAYS_FOR_TREND == core_temporal.MIN_DAYS_FOR_TREND == 14
    assert (
        atlas_temporal.RECENCY_HALF_LIFE_DAYS
        == core_temporal.RECENCY_HALF_LIFE_DAYS
        == 14
    )


def test_atlas_temporal_all_matches_re_export_set() -> None:
    assert set(atlas_temporal.__all__) == {
        "AnomalyScore",
        "CategoryPercentile",
        "LongTermTrend",
        "MIN_DAYS_FOR_ACCELERATION",
        "MIN_DAYS_FOR_PERCENTILES",
        "MIN_DAYS_FOR_TREND",
        "MIN_DAYS_FOR_VELOCITY",
        "RECENCY_HALF_LIFE_DAYS",
        "TemporalEngine",
        "TemporalEvidence",
        "VendorVelocity",
    }


def test_atlas_temporal_does_not_redefine_symbols() -> None:
    # AST-level guard: the wrapper body should contain only re-export
    # imports + ``__all__`` literal -- no def/class statements.
    import ast
    import inspect

    source = inspect.getsource(atlas_temporal)
    tree = ast.parse(source)

    redefinitions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            redefinitions.append(node.name)

    assert redefinitions == [], (
        f"atlas_brain.reasoning.temporal should be a pure re-export wrapper, "
        f"but it defines: {redefinitions}"
    )


def test_dataclasses_are_frozen() -> None:
    # Pin the immutability invariant so a future revert that drops
    # ``frozen=True`` in core.types surfaces here. Atlas callers
    # rebuild dataclasses on update rather than mutating fields, so
    # the frozen contract is load-bearing.
    from dataclasses import FrozenInstanceError

    velocity = atlas_temporal.VendorVelocity(
        vendor_name="acme",
        metric="urgency",
        current_value=1.0,
        previous_value=0.0,
        velocity=1.0,
        days_between=1,
    )
    with pytest.raises(FrozenInstanceError):
        velocity.metric = "different"  # type: ignore[misc]


import pytest  # noqa: E402  (used by the frozen-invariant test above)
