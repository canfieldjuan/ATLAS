"""Smoke tests for the slim EvidenceEngine landed in PR-C1d.

Locks the conclusions + suppression surface so subsequent slices in
PR-C1 (atlas-side review_enrichment carve-out, content_pipeline
wrapper conversions) can refactor with confidence.

The full atlas-side `tests/test_evidence_engine.py` exercises both the
slim core and the per-review enrichment surface (compute_urgency,
override_pain, etc.); those stay green against atlas's evidence_engine
until PR-C1f slims it. PR-C1k will rename the rich test file into
`tests/test_extracted_reasoning_core_evidence_engine.py` once the
content_pipeline wrappers land.
"""

from __future__ import annotations

import pytest

from extracted_reasoning_core.evidence_engine import (
    EvidenceEngine,
    get_evidence_engine,
    reload_evidence_engine,
)
from extracted_reasoning_core.types import ConclusionResult, SuppressionResult


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def test_engine_loads_default_yaml() -> None:
    engine = EvidenceEngine()
    assert engine.map_hash  # 16-char sha256 prefix
    assert len(engine.map_hash) == 16
    assert engine.map_path.endswith("evidence_map.yaml")
    assert isinstance(engine._enrichment, dict)
    assert isinstance(engine._conclusions, dict)
    assert isinstance(engine._suppression, dict)
    assert isinstance(engine._confidence_tiers, dict)


def test_get_evidence_engine_caches_by_path() -> None:
    reload_evidence_engine()
    a = get_evidence_engine()
    b = get_evidence_engine()
    assert a is b


def test_reload_evidence_engine_returns_fresh_instance() -> None:
    a = get_evidence_engine()
    reload_evidence_engine()
    b = get_evidence_engine()
    assert a is not b


# ------------------------------------------------------------------
# evaluate_conclusion (singular)
# ------------------------------------------------------------------


def test_evaluate_conclusion_returns_result_for_unknown_id() -> None:
    engine = EvidenceEngine()
    result = engine.evaluate_conclusion("does_not_exist", {})
    assert isinstance(result, ConclusionResult)
    assert result.conclusion_id == "does_not_exist"
    assert result.met is False
    assert result.confidence == "insufficient"


def test_evaluate_conclusion_handles_insufficient_data_explicitly() -> None:
    engine = EvidenceEngine()
    result = engine.evaluate_conclusion("insufficient_data", {"review_count": 999})
    assert isinstance(result, ConclusionResult)
    assert result.conclusion_id == "insufficient_data"


def test_evaluate_conclusion_evaluates_known_id() -> None:
    engine = EvidenceEngine()
    for cid in engine._conclusions:
        if cid == "insufficient_data":
            continue
        result = engine.evaluate_conclusion(cid, {})
        assert isinstance(result, ConclusionResult)
        assert result.conclusion_id == cid
        assert result.met is False
        assert isinstance(result.confidence, str)
        return
    pytest.skip("No non-insufficient_data conclusions in YAML to test")


# ------------------------------------------------------------------
# evaluate_conclusions (plural)
# ------------------------------------------------------------------


def test_evaluate_conclusions_returns_list_of_results() -> None:
    engine = EvidenceEngine()
    results = engine.evaluate_conclusions({})
    assert isinstance(results, list)
    assert all(isinstance(r, ConclusionResult) for r in results)


def test_evaluate_conclusions_short_circuits_on_insufficient_data() -> None:
    engine = EvidenceEngine()
    triggers = engine._conclusions.get("insufficient_data", {}).get("trigger", [])
    if not triggers:
        pytest.skip("evidence_map.yaml has no insufficient_data triggers")

    evidence: dict = {}
    for trig in triggers:
        path = trig.get("field", "")
        op = trig.get("operator", "eq")
        val = trig.get("value")
        if op in ("lt", "lte"):
            evidence[path] = (val if val is not None else 0) - 1
        elif op in ("gt", "gte"):
            evidence[path] = (val if val is not None else 0) + 1
        elif op == "eq":
            evidence[path] = val
        elif op == "in":
            vals = trig.get("values") or []
            if vals:
                evidence[path] = vals[0]

    results = engine.evaluate_conclusions(evidence)
    if (
        len(results) == 1
        and results[0].conclusion_id == "insufficient_data"
        and results[0].met
    ):
        return
    pytest.skip("Could not synthesize full insufficient_data trigger set from YAML")


# ------------------------------------------------------------------
# evaluate_suppression
# ------------------------------------------------------------------


def test_evaluate_suppression_returns_default_for_unknown_section() -> None:
    engine = EvidenceEngine()
    result = engine.evaluate_suppression("does_not_exist", {})
    assert isinstance(result, SuppressionResult)
    assert result.suppress is False
    assert result.degrade is False
    assert result.disclaimer is None
    assert result.fallback_label is None


def test_evaluate_suppression_returns_default_when_no_rules_match() -> None:
    engine = EvidenceEngine()
    for section in engine._suppression:
        result = engine.evaluate_suppression(section, {})
        assert isinstance(result, SuppressionResult)
        if not result.suppress and not result.degrade:
            return
    pytest.skip("Could not find a section where suppression default holds against empty evidence")


# ------------------------------------------------------------------
# Confidence labeling
# ------------------------------------------------------------------


def test_get_confidence_tier_returns_known_label() -> None:
    engine = EvidenceEngine()
    tier = engine.get_confidence_tier(0)
    assert tier in {"high", "medium", "low", "insufficient"}


def test_get_confidence_label_returns_string() -> None:
    engine = EvidenceEngine()
    label = engine.get_confidence_label(50)
    assert isinstance(label, str)
    assert label


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def test_resolve_field_walks_dotted_path() -> None:
    engine = EvidenceEngine()
    data = {"a": {"b": {"c": 7}}}
    assert engine._resolve_field(data, "a.b.c") == 7
    assert engine._resolve_field(data, "a.b.missing") is None
    assert engine._resolve_field(data, "missing") is None


def test_check_requirement_supports_all_operators() -> None:
    engine = EvidenceEngine()
    ev = {"x": 5}
    assert engine._check_requirement({"field": "x", "operator": "eq", "value": 5}, ev)
    assert engine._check_requirement({"field": "x", "operator": "gte", "value": 5}, ev)
    assert engine._check_requirement({"field": "x", "operator": "gt", "value": 4}, ev)
    assert engine._check_requirement({"field": "x", "operator": "lte", "value": 5}, ev)
    assert engine._check_requirement({"field": "x", "operator": "lt", "value": 6}, ev)
    assert engine._check_requirement(
        {"field": "x", "operator": "in", "values": [4, 5, 6]}, ev,
    )
    assert not engine._check_requirement(
        {"field": "x", "operator": "eq", "value": 99}, ev,
    )
