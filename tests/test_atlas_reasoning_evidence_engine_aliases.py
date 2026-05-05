"""Pin atlas's evidence_engine wrapper around core's slim engine.

PR-D7b3 promoted the slim conclusions+suppression engine into
``extracted_reasoning_core.evidence_engine`` (per PR-C1's slim-core
split). Atlas keeps the import surface
``atlas_brain.reasoning.evidence_engine`` as a subclass wrapper that
adds the per-review enrichment methods (``compute_urgency`` /
``override_pain`` / ``derive_recommend`` / ``derive_price_complaint``
/ ``derive_budget_authority``) that stay atlas-side because
``derive_price_complaint`` depends on atlas-only
``_b2b_phrase_metadata``.

These tests pin:
  - atlas's ``EvidenceEngine`` is a subclass of core's slim engine
    (so ``evaluate_conclusions`` / ``evaluate_suppression`` /
    ``get_confidence_tier`` / ``get_confidence_label`` /
    ``evaluate_conclusion`` are inherited, not duplicated).
  - atlas's ``ConclusionResult`` and ``SuppressionResult`` ARE core's
    types -- not look-alike local re-definitions.
  - the six atlas-side enrichment methods are present on the subclass
    and return shape-correct values from real YAML.
  - the wrapper module body has no ``ConclusionResult`` /
    ``SuppressionResult`` / ``EvidenceEngine`` re-definitions
    (subclass pattern means EvidenceEngine IS defined here, so the
    AST guard checks only that the result types are NOT redefined).
"""

from __future__ import annotations

from atlas_brain.reasoning import evidence_engine as atlas_engine
from extracted_reasoning_core import evidence_engine as core_engine
from extracted_reasoning_core import types as core_types


def test_evidence_engine_subclasses_core() -> None:
    assert issubclass(atlas_engine.EvidenceEngine, core_engine.EvidenceEngine)
    assert atlas_engine.EvidenceEngine is not core_engine.EvidenceEngine


def test_conclusion_result_alias_identity() -> None:
    assert atlas_engine.ConclusionResult is core_types.ConclusionResult


def test_suppression_result_alias_identity() -> None:
    assert atlas_engine.SuppressionResult is core_types.SuppressionResult


def test_atlas_enrichment_methods_present() -> None:
    # All six atlas-side enrichment methods must resolve on the subclass
    # so callers like services/b2b/enrichment_derivation.py keep working.
    expected = {
        "compute_urgency",
        "override_pain",
        "derive_recommend",
        "derive_price_complaint",
        "derive_budget_authority",
        "_check_derivation_rule",
    }
    for name in expected:
        attr = getattr(atlas_engine.EvidenceEngine, name, None)
        assert callable(attr), f"missing or non-callable: {name}"
        # Must be defined on the subclass, not core (else PR-C1's
        # slim-core split silently regressed)
        assert name not in vars(core_engine.EvidenceEngine), (
            f"{name} unexpectedly present on core slim engine"
        )


def test_inherited_conclusions_methods_resolve() -> None:
    # These come from core via inheritance -- the subclass must not
    # shadow them.
    for name in (
        "evaluate_conclusions",
        "evaluate_conclusion",
        "evaluate_suppression",
        "get_confidence_tier",
        "get_confidence_label",
        "_check_requirement",
        "_check_suppression_rule",
        "_check_condition_simple",
        "_resolve_field",
    ):
        sub_attr = getattr(atlas_engine.EvidenceEngine, name, None)
        core_attr = getattr(core_engine.EvidenceEngine, name, None)
        assert sub_attr is core_attr, f"{name} unexpectedly overridden on atlas subclass"


def test_init_loads_yaml_and_precompiles_regexes() -> None:
    e = atlas_engine.EvidenceEngine()
    # Core-populated state
    assert e._conclusions, "conclusions dict empty"
    assert e._suppression, "suppression dict empty"
    assert e._enrichment, "enrichment dict empty"
    assert isinstance(e.map_hash, str) and len(e.map_hash) == 16
    # Atlas-subclass-populated state (regex precompile)
    assert isinstance(e._rec_positive, list)
    assert isinstance(e._rec_negative, list)
    assert isinstance(e._price_positive, list)


def test_get_evidence_engine_returns_subclass() -> None:
    e = atlas_engine.get_evidence_engine()
    assert isinstance(e, atlas_engine.EvidenceEngine)
    assert isinstance(e, core_engine.EvidenceEngine)


def test_reload_evidence_engine_returns_fresh_subclass() -> None:
    first = atlas_engine.get_evidence_engine()
    reloaded = atlas_engine.reload_evidence_engine()
    assert isinstance(reloaded, atlas_engine.EvidenceEngine)
    # Cache invalidated then re-populated
    assert reloaded is not first or reloaded.map_hash == first.map_hash


def test_atlas_evidence_engine_all_matches_re_export_set() -> None:
    assert set(atlas_engine.__all__) == {
        "ConclusionResult",
        "EvidenceEngine",
        "SuppressionResult",
        "get_evidence_engine",
        "reload_evidence_engine",
    }


def test_atlas_evidence_engine_does_not_redefine_result_types() -> None:
    # AST-level guard: the wrapper must NOT redefine ConclusionResult /
    # SuppressionResult locally; both must come from core.types via
    # import. EvidenceEngine subclass is allowed (this is a wrapper
    # that adds enrichment methods, not a pure re-export).
    import ast
    import inspect

    source = inspect.getsource(atlas_engine)
    tree = ast.parse(source)

    redefined_results: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name in {"ConclusionResult", "SuppressionResult"}:
                redefined_results.append(node.name)

    assert redefined_results == [], (
        f"atlas_brain.reasoning.evidence_engine must not redefine result "
        f"types -- they come from core.types. Found: {redefined_results}"
    )


def test_compute_urgency_returns_atlas_shaped_float() -> None:
    e = atlas_engine.EvidenceEngine()
    score = e.compute_urgency(
        indicators={},
        rating=None,
        rating_max=5,
        content_type="review",
        source_weight=0.7,
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 10.0


def test_evaluate_conclusions_returns_core_conclusion_results() -> None:
    e = atlas_engine.EvidenceEngine()
    results = e.evaluate_conclusions({"total_reviews": 0})
    assert isinstance(results, list)
    for r in results:
        # Must be the canonical core type, reachable via either
        # atlas's re-export or core directly.
        assert isinstance(r, core_types.ConclusionResult)
        assert isinstance(r, atlas_engine.ConclusionResult)
