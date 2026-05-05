"""Pin atlas's re-exports from ``atlas_brain.reasoning.archetypes``.

PR-D7b5 promoted the archetypes module into
``extracted_reasoning_core.archetypes`` (PR-C1a's 10-archetype
consolidation). Atlas keeps the import surface
``atlas_brain.reasoning.archetypes`` as a thin re-export wrapper.

Notable design decision (mirrors core's docstring): atlas's
``ArchetypeMatch`` aliases core's ``_ArchetypeMatchInternal`` -- the
rich internal type that preserves atlas's original field names
(``archetype`` / ``score`` / ``matched_signals`` / ``missing_signals``
/ ``risk_level``). Core's canonical public ``ArchetypeMatch`` (in
``core.types``, with renamed fields like ``archetype_id`` / ``label`` /
``evidence_hits``) is only reachable via ``core.api.score_archetypes``
for external products. The wrapper keeps atlas's existing
``score_evidence`` / ``best_match`` / ``top_matches`` /
``enrich_evidence_with_archetypes`` callers working without field
renames -- ``test_reasoning_live.py`` reads ``m.archetype`` /
``m.matched_signals`` / ``m.risk_level`` and those still resolve.
"""

from __future__ import annotations

from atlas_brain.reasoning import archetypes as atlas_archetypes
from extracted_reasoning_core import archetypes as core_archetypes


def test_archetype_match_aliases_internal_not_public() -> None:
    # atlas's ArchetypeMatch is the rich internal type from core,
    # NOT the canonical public ArchetypeMatch in core.types. Pin
    # this so a future refactor that "fixes" the import to point at
    # core.types.ArchetypeMatch doesn't silently break atlas callers
    # reading m.archetype / m.matched_signals / m.risk_level.
    assert atlas_archetypes.ArchetypeMatch is core_archetypes._ArchetypeMatchInternal


def test_archetype_match_preserves_atlas_field_names() -> None:
    # atlas-style field names must be present on the aliased type so
    # test_reasoning_live's m.archetype / m.matched_signals reads
    # keep working through the wrapper.
    fields = {
        f.name for f in atlas_archetypes.ArchetypeMatch.__dataclass_fields__.values()
    }
    assert fields == {
        "archetype",
        "score",
        "matched_signals",
        "missing_signals",
        "risk_level",
    }


def test_archetype_profile_alias_identity() -> None:
    assert atlas_archetypes.ArchetypeProfile is core_archetypes.ArchetypeProfile


def test_signal_rule_alias_identity() -> None:
    assert atlas_archetypes.SignalRule is core_archetypes.SignalRule


def test_archetypes_dict_alias_identity() -> None:
    # The ARCHETYPES dict itself must be the same object so a caller
    # that holds a reference always reads the canonical 10-entry set.
    assert atlas_archetypes.ARCHETYPES is core_archetypes.ARCHETYPES


def test_archetypes_dict_has_ten_canonical_entries() -> None:
    # Pin the count + names so a future revert that drops one of the
    # PR-C1a archetypes surfaces here.
    assert set(atlas_archetypes.ARCHETYPES.keys()) == {
        "pricing_shock",
        "feature_gap",
        "acquisition_decay",
        "leadership_redesign",
        "integration_break",
        "support_collapse",
        "category_disruption",
        "compliance_gap",
        "scale_up_stumble",
        "pivot_abandonment",
    }


def test_match_threshold_alias_identity() -> None:
    assert atlas_archetypes.MATCH_THRESHOLD == core_archetypes.MATCH_THRESHOLD == 0.25


def test_score_evidence_alias_identity() -> None:
    assert atlas_archetypes.score_evidence is core_archetypes.score_evidence


def test_best_match_alias_identity() -> None:
    assert atlas_archetypes.best_match is core_archetypes.best_match


def test_top_matches_alias_identity() -> None:
    assert atlas_archetypes.top_matches is core_archetypes.top_matches


def test_get_archetype_alias_identity() -> None:
    assert atlas_archetypes.get_archetype is core_archetypes.get_archetype


def test_get_falsification_conditions_alias_identity() -> None:
    assert (
        atlas_archetypes.get_falsification_conditions
        is core_archetypes.get_falsification_conditions
    )


def test_enrich_evidence_with_archetypes_alias_identity() -> None:
    assert (
        atlas_archetypes.enrich_evidence_with_archetypes
        is core_archetypes.enrich_evidence_with_archetypes
    )


def test_atlas_archetypes_all_matches_re_export_set() -> None:
    assert set(atlas_archetypes.__all__) == {
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
    }


def test_atlas_archetypes_does_not_redefine_symbols() -> None:
    # AST-level guard: the wrapper body should contain only re-export
    # imports + ``__all__`` literal -- no def/class statements.
    import ast
    import inspect

    source = inspect.getsource(atlas_archetypes)
    tree = ast.parse(source)

    redefinitions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            redefinitions.append(node.name)

    assert redefinitions == [], (
        f"atlas_brain.reasoning.archetypes should be a pure re-export "
        f"wrapper, but it defines: {redefinitions}"
    )


def test_score_evidence_returns_atlas_style_match() -> None:
    # End-to-end smoke: scoring real evidence yields matches whose
    # field reads work via atlas's ArchetypeMatch alias. Catches a
    # regression where score_evidence's return type drifts off the
    # internal type.
    evidence = {
        "vendor_name": "TestCo",
        "avg_urgency": 7.0,
        "top_pain": "pricing surprise",
        "competitor_count": 4,
    }
    matches = atlas_archetypes.score_evidence(evidence)
    assert len(matches) > 0
    top = matches[0]
    # Atlas-style field reads -- these would fail if ArchetypeMatch
    # were rebound to the canonical core.types shape.
    assert isinstance(top.archetype, str)
    assert isinstance(top.score, float)
    assert isinstance(top.matched_signals, list)
    assert isinstance(top.missing_signals, list)
    assert isinstance(top.risk_level, str)
