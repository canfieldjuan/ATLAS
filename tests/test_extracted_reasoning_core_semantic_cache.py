from __future__ import annotations

from extracted_reasoning_core.api import (
    build_semantic_cache_key,
    compute_evidence_hash,
)
from extracted_reasoning_core.types import EvidenceItem, ReasoningInput


def _input(*, entity_id: str = "acme", goal: str = "synthesize", pack_name: str | None = None) -> ReasoningInput:
    return ReasoningInput(
        entity_id=entity_id,
        entity_type="vendor",
        goal=goal,
        evidence=(
            EvidenceItem(source_type="review", source_id="r1", text="Pricing concern raised."),
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
    # Default pack name materializes as "default" in the key prefix.
    assert pack_default.startswith("reasoning/L2/default/")


def test_build_semantic_cache_key_is_sensitive_to_evidence_and_goal() -> None:
    base = build_semantic_cache_key(_input(), tier="L2")
    different_goal = build_semantic_cache_key(_input(goal="different goal"), tier="L2")
    different_entity = build_semantic_cache_key(_input(entity_id="other_vendor"), tier="L2")

    assert base != different_goal
    assert base != different_entity
