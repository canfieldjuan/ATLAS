from __future__ import annotations

import pytest

from extracted_reasoning_core.api import (
    build_semantic_cache_key,
    compute_evidence_hash,
)
from extracted_reasoning_core.types import EvidenceItem, ReasoningInput


def _input(*, entity_id: str = "acme", goal: str = "synthesize", pack_name: str | None = None, evidence_text: str = "Pricing concern raised.") -> ReasoningInput:
    return ReasoningInput(
        entity_id=entity_id,
        entity_type="vendor",
        goal=goal,
        evidence=(
            EvidenceItem(source_type="review", source_id="r1", text=evidence_text),
            EvidenceItem(source_type="ticket", source_id="t9", text="Renewal pending."),
        ),
        pack_name=pack_name,
    )


def test_compute_evidence_hash_is_stable_and_order_independent() -> None:
    a = compute_evidence_hash({"k1": "v1", "k2": "v2", "nested": {"a": 1, "b": 2}})
    b = compute_evidence_hash({"k2": "v2", "nested": {"b": 2, "a": 1}, "k1": "v1"})
    assert a == b
    # 16-char hex prefix of sha256 (matches semantic_cache_keys storage convention)
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_compute_evidence_hash_distinguishes_different_content() -> None:
    a = compute_evidence_hash({"k": "v1"})
    b = compute_evidence_hash({"k": "v2"})
    assert a != b


def test_build_semantic_cache_key_is_deterministic_for_identical_inputs() -> None:
    k1 = build_semantic_cache_key(_input(), tier="L2", pack_name="reasoning_synthesis")
    k2 = build_semantic_cache_key(_input(), tier="L2", pack_name="reasoning_synthesis")
    assert k1 == k2
    assert k1.startswith("reasoning/L2/reasoning_synthesis/")
    digest = k1.split("/")[-1]
    assert len(digest) == 16


def test_build_semantic_cache_key_is_sensitive_to_tier_and_pack() -> None:
    base = build_semantic_cache_key(_input(), tier="L2", pack_name="reasoning_synthesis")
    different_tier = build_semantic_cache_key(_input(), tier="L3", pack_name="reasoning_synthesis")
    different_pack = build_semantic_cache_key(_input(), tier="L2", pack_name="other_pack")
    pack_from_input = build_semantic_cache_key(_input(pack_name="from_input"), tier="L2")
    pack_default = build_semantic_cache_key(_input(), tier="L2")

    assert base != different_tier
    assert base != different_pack
    assert pack_from_input != pack_default
    # Default pack name matches run_reasoning / continue_reasoning fallback so
    # cache lookups for inputs without an explicit pack align with what
    # synthesis computed against.
    assert pack_default.startswith("reasoning/L2/reasoning_synthesis/")


def test_build_semantic_cache_key_is_sensitive_to_evidence_and_goal() -> None:
    base = build_semantic_cache_key(_input(), tier="L2")
    different_goal = build_semantic_cache_key(_input(goal="different goal"), tier="L2")
    different_entity = build_semantic_cache_key(_input(entity_id="other_vendor"), tier="L2")
    different_evidence = build_semantic_cache_key(_input(evidence_text="entirely different review text"), tier="L2")

    assert base != different_goal
    assert base != different_entity
    assert base != different_evidence


def test_build_semantic_cache_key_rejects_slashes_in_tier_or_pack() -> None:
    with pytest.raises(ValueError, match="tier"):
        build_semantic_cache_key(_input(), tier="L2/x")
    with pytest.raises(ValueError, match="pack_name"):
        build_semantic_cache_key(_input(), tier="L2", pack_name="bad/pack")
