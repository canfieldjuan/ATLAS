"""Tests for B2B reasoning synthesis v2: wedge registry, pool compression,
validation, and reader contract."""

import pytest
from datetime import datetime, timezone


# ---- Module 1: Wedge Registry ----

class TestWedgeRegistry:
    def test_all_archetypes_mapped(self):
        from atlas_brain.reasoning.wedge_registry import (
            Wedge, wedge_from_archetype,
        )
        archetypes = [
            "pricing_shock", "feature_gap", "acquisition_decay",
            "leadership_redesign", "integration_break", "support_collapse",
            "category_disruption", "compliance_gap", "mixed", "stable",
        ]
        for arch in archetypes:
            result = wedge_from_archetype(arch)
            assert isinstance(result, Wedge), f"No mapping for {arch}"

    def test_validate_wedge_accepts_valid(self):
        from atlas_brain.reasoning.wedge_registry import (
            WEDGE_ENUM_VALUES, validate_wedge,
        )
        for val in WEDGE_ENUM_VALUES:
            assert validate_wedge(val) is not None, f"Should accept {val}"

    def test_validate_wedge_rejects_bad(self):
        from atlas_brain.reasoning.wedge_registry import validate_wedge
        assert validate_wedge("pricing_support_compound") is None
        assert validate_wedge("feature_disruption_compound") is None
        assert validate_wedge("totally_made_up") is None
        assert validate_wedge("") is None

    def test_wedge_enum_values_frozenset(self):
        from atlas_brain.reasoning.wedge_registry import WEDGE_ENUM_VALUES
        assert isinstance(WEDGE_ENUM_VALUES, frozenset)
        assert len(WEDGE_ENUM_VALUES) == 10

    def test_archetype_to_wedge_defaults(self):
        from atlas_brain.reasoning.wedge_registry import (
            Wedge, wedge_from_archetype,
        )
        # Unknown archetype defaults to SEGMENT_MISMATCH
        assert wedge_from_archetype("nonexistent") == Wedge.SEGMENT_MISMATCH

    def test_sales_motion_populated(self):
        from atlas_brain.reasoning.wedge_registry import (
            Wedge, get_sales_motion,
        )
        for w in Wedge:
            motion = get_sales_motion(w)
            assert isinstance(motion, str)
            assert len(motion) > 10


# ---- Module 2: Pool Compression ----

def _make_evidence_vault():
    return {
        "weakness_evidence": [
            {"category": "pricing", "mention_count": 25, "review_ids": ["r1", "r2"]},
            {"category": "support", "mention_count": 15, "review_ids": ["r3"]},
            {"category": "performance", "mention_count": 5},
        ],
        "strength_evidence": [
            {"category": "ease_of_use", "mention_count": 30, "review_ids": ["r4"]},
            {"category": "integrations", "mention_count": 10},
        ],
        "metric_snapshot": {
            "total_reviews": 120,
            "churn_density": 0.35,
            "displacement_mention_count": 18,
            "avg_rating": 3.2,
        },
        "provenance": {
            "enrichment_window_start": "2025-12-01",
            "enrichment_window_end": "2026-03-15",
        },
    }


def _make_layers():
    return {
        "evidence_vault": _make_evidence_vault(),
        "segment": {
            "affected_departments": [
                {"department": "engineering", "churn_rate": 0.25, "review_count": 40},
                {"department": "marketing", "churn_rate": 0.15, "review_count": 20},
            ],
            "affected_roles": [
                {"role_type": "admin", "churn_rate": 0.30, "review_count": 15},
            ],
            "contract_segments": [
                {"segment": "enterprise", "count": 50, "churn_rate": 0.20},
            ],
        },
        "temporal": {
            "timeline_signal_summary": {
                "evaluation_deadline_signals": 7,
                "contract_end_signals": 3,
            },
            "keyword_spikes": {
                "spike_count": 1,
                "spike_keywords": ["alternative"],
                "keyword_details": [
                    {"keyword": "alternative", "magnitude": 3.5},
                ],
            },
            "evaluation_deadlines": [
                {"label": "Q2 renewal", "urgency": 8.0, "trigger_type": "timeline_signal"},
            ],
        },
        "displacement": [
            {
                "to_vendor": "CompetitorA",
                "flow_summary": {
                    "mention_count": 12,
                    "explicit_switch_count": 4,
                    "active_evaluation_count": 8,
                },
            },
            {
                "to_vendor": "CompetitorB",
                "flow_summary": {
                    "mention_count": 5,
                    "explicit_switch_count": 1,
                    "active_evaluation_count": 3,
                },
            },
        ],
        "category": {
            "vendor_count": 12,
            "displacement_flow_count": 5,
            "market_regime": {
                "regime_type": "fragmented",
                "confidence": 0.7,
                "narrative": "Market is fragmenting",
            },
        },
        "accounts": {
            "summary": {
                "total_accounts": 42,
                "decision_maker_count": 15,
                "high_intent_count": 8,
            },
            "accounts": [
                {"company_name": "Acme Corp", "urgency_score": 0.9, "decision_maker": True},
                {"company_name": "Globex", "urgency_score": 0.5},
            ],
        },
    }


class TestPoolCompression:
    def test_compress_produces_packet(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        assert packet.vendor_name == "TestVendor"
        assert "evidence_vault" in packet.pools
        assert "displacement" in packet.pools
        assert len(packet.aggregates) > 0

    def test_scoring_order(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        # Evidence vault items should be sorted by score descending
        ev_items = packet.pools.get("evidence_vault", [])
        scores = [i.score for i in ev_items]
        assert scores == sorted(scores, reverse=True)

    def test_source_ids_populated(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        sids = packet.source_ids()
        assert len(sids) > 0
        # All source IDs follow pool:kind:key pattern
        for sid in sids:
            parts = sid.split(":")
            assert len(parts) >= 3, f"Bad source_id format: {sid}"

    def test_aggregate_computation(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        agg_map = {a.label: a for a in packet.aggregates}
        assert agg_map["total_reviews"].value == 120
        assert agg_map["total_explicit_switches"].value == 5  # 4 + 1
        assert agg_map["total_active_evaluations"].value == 11  # 8 + 3
        assert agg_map["total_accounts"].value == 42

    def test_to_llm_payload(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        payload = packet.to_llm_payload()
        # Every pool item has _sid
        for pool_name, items in payload.items():
            if pool_name == "precomputed_aggregates":
                continue
            assert isinstance(items, list)
            for item in items:
                assert "_sid" in item, f"Missing _sid in {pool_name}"
        # Aggregates have {value, _sid} wrappers
        aggs = payload["precomputed_aggregates"]
        for label, wrapper in aggs.items():
            assert "value" in wrapper
            assert "_sid" in wrapper

    def test_max_items_per_pool(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers, max_items_per_pool=2)
        # Displacement should be capped at 2
        assert len(packet.pools.get("displacement", [])) <= 2

    def test_empty_layers(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        packet = compress_vendor_pools("EmptyVendor", {})
        assert packet.vendor_name == "EmptyVendor"
        assert len(packet.pools) == 0
        assert len(packet.aggregates) == 0

    def test_review_ids_carried_through(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        ev_items = packet.pools.get("evidence_vault", [])
        # At least one item should have review_ids
        has_rids = any(len(i.source_ref.review_ids) > 0 for i in ev_items)
        assert has_rids


# ---- Module 3: Prompt ----

class TestPrompt:
    def test_prompt_contains_wedge_values(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        from atlas_brain.reasoning.wedge_registry import WEDGE_ENUM_VALUES
        for val in WEDGE_ENUM_VALUES:
            assert val in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_requires_sid(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "_sid" in BATTLE_CARD_REASONING_PROMPT
        assert "source_id" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_has_citations_arrays(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert '"citations": ["<_sid>"]' in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_has_audience_grounding(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "Sales reps" in BATTLE_CARD_REASONING_PROMPT
        assert "Precision matters more than coverage" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_has_section_priority(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "causal_narrative > migration_proof" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_has_conflict_handling(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "conflicts across pools" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_has_signal_trigger_type(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "deadline|spike|announcement|seasonal|signal" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_has_structured_falsification(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert '"monitorable"' in BATTLE_CARD_REASONING_PROMPT
        assert '"signal_source"' in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_schema_uses_placeholders_not_defaults(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        # Schema section should use placeholder markers, not literal defaults
        schema = BATTLE_CARD_REASONING_PROMPT[BATTLE_CARD_REASONING_PROMPT.find("Output ONLY"):]
        idx = schema.find('"switching_is_real"')
        schema_line = schema[idx:idx + 80]
        assert "<true|false" in schema_line, f"Schema should use placeholder: {schema_line}"

    def test_prompt_meta_exempt_from_citation(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "exempt" in BATTLE_CARD_REASONING_PROMPT.lower()

    def test_prompt_version_changes(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT_VERSION,
        )
        assert isinstance(BATTLE_CARD_REASONING_PROMPT_VERSION, str)
        assert len(BATTLE_CARD_REASONING_PROMPT_VERSION) == 8

    def test_backward_compat_alias(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            VALID_WEDGE_TYPES,
        )
        assert isinstance(VALID_WEDGE_TYPES, tuple)
        assert len(VALID_WEDGE_TYPES) == 10


# ---- Module 4: Validation ----

def _make_valid_synthesis(packet=None):
    """Produce a synthesis dict that should pass validation."""
    from atlas_brain.autonomous.tasks._b2b_pool_compression import (
        compress_vendor_pools,
    )
    if packet is None:
        packet = compress_vendor_pools("TestVendor", _make_layers())

    # Build aggregate lookup for valid source_ids
    agg_sids = {a.source_id: a.value for a in packet.aggregates}
    # Pick a valid accounts summary sid
    total_accounts_sid = "accounts:summary:total_accounts"
    total_accounts_val = agg_sids.get(total_accounts_sid, 42)
    total_switches_sid = "displacement:aggregate:total_explicit_switches"
    total_switches_val = agg_sids.get(total_switches_sid, 5)
    eval_signals_sid = "temporal:signal:evaluation_deadline_signals"
    eval_signals_val = agg_sids.get(eval_signals_sid, 7)

    # Pick a valid pool item sid for citations
    pool_sid = "vault:weakness:pricing"

    return {
        "schema_version": "2.0",
        "causal_narrative": {
            "primary_wedge": "price_squeeze",
            "trigger": "Price increase",
            "who_most_affected": "Enterprise Ops teams",
            "why_now": "Recent pricing change",
            "what_would_weaken_thesis": [
                {
                    "condition": "Prices stabilize",
                    "signal_source": "temporal",
                    "monitorable": True,
                },
            ],
            "causal_chain": "Price hike -> budget pressure -> evaluation",
            "confidence": "high",
            "data_gaps": [],
            "citations": [pool_sid],
        },
        "segment_playbook": {
            "priority_segments": [
                {
                    "segment": "Enterprise Ops",
                    "why_vulnerable": "Budget pressure",
                    "best_opening_angle": "TCO comparison",
                    "disqualifier": "Long-term contract",
                    "estimated_reach": {
                        "value": total_accounts_val,
                        "source_id": total_accounts_sid,
                    },
                    "citations": [pool_sid],
                },
            ],
            "confidence": "medium",
            "data_gaps": [],
        },
        "timing_intelligence": {
            "best_timing_window": "Q2 renewal cycle",
            "immediate_triggers": [
                {
                    "trigger": "Contract renewals",
                    "type": "deadline",
                    "urgency": "high",
                    "action": "Outreach",
                    "source": {
                        "source_id": "temporal:deadline:q2_renewal",
                        "source_type": "temporal",
                    },
                },
            ],
            "active_eval_signals": {
                "value": eval_signals_val,
                "source_id": eval_signals_sid,
            },
            "seasonal_pattern": "Q2/Q4 renewal spikes",
            "confidence": "medium",
            "data_gaps": [],
            "citations": ["temporal:deadline:q2_renewal"],
        },
        "competitive_reframes": {
            "reframes": [
                {
                    "incumbent_claim": "Ease of use",
                    "why_buyers_believe_it": "Legacy familiarity",
                    "reframe": "Ease masks complexity",
                    "proof_point": {
                        "field": "weakness_pricing",
                        "value": 25,
                        "source_id": pool_sid,
                        "interpretation": "High pricing complaints",
                    },
                    "best_segment": "Enterprise Ops",
                    "citations": [pool_sid],
                },
            ],
            "confidence": "medium",
            "data_gaps": [],
        },
        "migration_proof": {
            "switching_is_real": True,
            "evidence_type": "explicit_switch",
            "switch_volume": {
                "value": total_switches_val,
                "source_id": total_switches_sid,
            },
            "top_destination": {
                "value": "CompetitorA",
                "source_id": "displacement:flow:competitora",
            },
            "primary_switch_driver": {
                "value": "Pricing",
                "source_id": pool_sid,
            },
            "named_examples": [
                {
                    "company": "Acme Corp",
                    "evidence": "Switched due to pricing",
                    "source_type": "review_site",
                    "quotable": True,
                    "source_id": pool_sid,
                },
            ],
            "evaluation_vs_switching": "More evaluating than switching",
            "confidence": "high",
            "data_gaps": [],
            "citations": [pool_sid, total_switches_sid],
        },
        "meta": {
            "evidence_window_start": "2025-12-01",
            "evidence_window_end": "2026-03-15",
            "total_evidence_items": 20,
            "synthesized_at": "2026-03-19T00:00:00Z",
        },
    }, packet


class TestValidation:
    def test_valid_passes(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, f"Expected valid, got: {result.summary()}"

    def test_bad_wedge_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["causal_narrative"]["primary_wedge"] = "totally_made_up"
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "invalid_wedge" in codes

    def test_missing_section_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        del synthesis["migration_proof"]
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "missing_section" in codes

    def test_hallucinated_source_id_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 99,
            "source_id": "fake:nonexistent:source",
        }
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "hallucinated_source_id" in codes

    def test_value_mismatch_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        # Correct source_id but wrong value
        synthesis["timing_intelligence"]["active_eval_signals"] = {
            "value": 999,
            "source_id": "temporal:signal:evaluation_deadline_signals",
        }
        result = validate_synthesis(synthesis, packet)
        # Value mismatch is a warning, not an error
        assert result.is_valid
        codes = [w.code for w in result.warnings]
        assert "value_mismatch" in codes

    def test_insufficient_without_gaps_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["competitive_reframes"]["confidence"] = "insufficient"
        synthesis["competitive_reframes"]["data_gaps"] = []
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # Warning, not error
        codes = [w.code for w in result.warnings]
        assert "missing_data_gaps" in codes

    def test_invalid_evidence_type_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["evidence_type"] = "maybe_switched"
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # Warning
        codes = [w.code for w in result.warnings]
        assert "invalid_evidence_type" in codes

    def test_missing_citations_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        del synthesis["causal_narrative"]["citations"]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # Warning, not error
        codes = [w.code for w in result.warnings]
        assert "missing_citations" in codes

    def test_unstructured_falsification_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["causal_narrative"]["what_would_weaken_thesis"] = [
            "Prices stabilize",  # bare string instead of structured
        ]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # Warning
        codes = [w.code for w in result.warnings]
        assert "unstructured_falsification" in codes

    def test_invalid_trigger_type_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["timing_intelligence"]["immediate_triggers"] = [
            {"trigger": "test", "type": "magic", "urgency": "high", "action": "do"},
        ]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # Warning
        codes = [w.code for w in result.warnings]
        assert "invalid_trigger_type" in codes

    def test_signal_trigger_type_accepted(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["timing_intelligence"]["immediate_triggers"] = [
            {"trigger": "eval post", "type": "signal", "urgency": "high", "action": "reach out"},
        ]
        result = validate_synthesis(synthesis, packet)
        trigger_warns = [w for w in result.warnings if w.code == "invalid_trigger_type"]
        assert len(trigger_warns) == 0

    def test_no_packet_still_validates(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, _ = _make_valid_synthesis()
        # Without packet, source_id and value checks are skipped
        result = validate_synthesis(synthesis, packet=None)
        assert result.is_valid


# ---- Module 6: Reader Contract ----

class TestSynthesisReader:
    def test_v1_backward_compat(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        v1_raw = {
            "causal_narrative": {
                "primary_wedge": "price_squeeze",
                "confidence": "high",
            },
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "low"},
            "competitive_reframes": {"confidence": "insufficient"},
            "migration_proof": {"confidence": "high"},
            "evidence_window": {
                "earliest": "2025-12-01",
                "latest": "2026-03-15",
            },
        }
        view = load_synthesis_view(v1_raw, "TestVendor")
        assert view.schema_version == "v1"
        assert not view.is_v2
        assert view.primary_wedge is not None
        assert view.wedge_label == "Price Squeeze"

    def test_v2_detection(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        v2_raw = {
            "schema_version": "2.0",
            "causal_narrative": {"primary_wedge": "feature_parity", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "meta": {"evidence_window_start": "2025-12-01"},
        }
        view = load_synthesis_view(v2_raw, "TestVendor")
        assert view.is_v2
        assert view.schema_version == "v2"

    def test_should_suppress(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "insufficient"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "low"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        assert not view.should_suppress("causal_narrative")
        assert view.should_suppress("segment_playbook")
        assert not view.should_suppress("timing_intelligence")

    def test_trace_number_v2(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "schema_version": "2.0",
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {
                "active_eval_signals": {"value": 7, "source_id": "temporal:signal:evaluation_deadline_signals"},
                "confidence": "medium",
            },
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "meta": {},
        }
        view = load_synthesis_view(raw, "V")
        tn = view.trace_number("timing_intelligence", "active_eval_signals")
        assert tn.value == 7
        assert tn.has_provenance
        assert "temporal" in tn.source_id

    def test_trace_number_v1_bare(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {
                "active_eval_signals": 7,
                "confidence": "medium",
            },
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        tn = view.trace_number("timing_intelligence", "active_eval_signals")
        assert tn.value == 7
        assert not tn.has_provenance

    def test_is_quotable(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "low"},
            "competitive_reframes": {"confidence": "insufficient"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        assert view.is_quotable("causal_narrative")
        assert view.is_quotable("segment_playbook")
        assert not view.is_quotable("timing_intelligence")  # low
        assert not view.is_quotable("competitive_reframes")  # insufficient

    def test_inject_synthesis_into_card(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view, inject_synthesis_into_card,
        )
        raw = {
            "schema_version": "2.0",
            "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high", "trigger": "Price hike"},
            "segment_playbook": {"confidence": "insufficient", "data_gaps": ["No segment data"]},
            "timing_intelligence": {"confidence": "medium", "best_timing_window": "Q2"},
            "competitive_reframes": {"confidence": "medium", "reframes": []},
            "migration_proof": {"confidence": "high", "switching_is_real": True},
            "meta": {"evidence_window_start": "2025-12-01"},
        }
        view = load_synthesis_view(raw, "TestVendor")
        card: dict = {}
        inject_synthesis_into_card(card, view)

        # Sections with sufficient confidence are injected
        assert "causal_narrative" in card
        assert card["causal_narrative"]["trigger"] == "Price hike"
        # Insufficient section is suppressed
        assert "segment_playbook" not in card
        # Wedge info injected
        assert card["synthesis_wedge"] == "price_squeeze"
        assert card["synthesis_wedge_label"] == "Price Squeeze"
        assert card["synthesis_schema_version"] == "v2"

    def test_data_gaps(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "insufficient", "data_gaps": ["No buyers"]},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        assert view.data_gaps("segment_playbook") == ["No buyers"]
        assert view.data_gaps("causal_narrative") == []

    def test_validation_warnings_accessible(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "_validation_warnings": [
                {"path": "timing_intelligence.active_eval_signals", "code": "value_mismatch", "message": "off by 1"},
            ],
        }
        view = load_synthesis_view(raw, "V")
        assert len(view.validation_warnings) == 1
        # Warning on timing_intelligence makes it not quotable
        assert not view.is_quotable("timing_intelligence")

    def test_citations_accessor(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {
                "primary_wedge": "stable",
                "confidence": "high",
                "citations": ["vault:weakness:pricing", "temporal:spike:alternative"],
            },
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        assert view.citations("causal_narrative") == [
            "vault:weakness:pricing", "temporal:spike:alternative",
        ]
        assert view.citations("segment_playbook") == []

    def test_falsification_conditions_structured(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {
                "primary_wedge": "price_squeeze",
                "confidence": "high",
                "what_would_weaken_thesis": [
                    {
                        "condition": "Prices drop",
                        "signal_source": "temporal",
                        "monitorable": True,
                    },
                ],
            },
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        fc = view.falsification_conditions()
        assert len(fc) == 1
        assert fc[0]["condition"] == "Prices drop"
        assert fc[0]["monitorable"] is True

    def test_falsification_conditions_v1_compat(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {
                "primary_wedge": "stable",
                "confidence": "high",
                "what_would_weaken_thesis": ["Prices drop"],
            },
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        fc = view.falsification_conditions()
        assert len(fc) == 1
        assert fc[0]["condition"] == "Prices drop"
        assert fc[0]["monitorable"] is False


# ---- Integration: Compression -> Validation -> Reader ----

class TestIntegration:
    def test_compress_validate_read_pipeline(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view, inject_synthesis_into_card,
        )

        layers = _make_layers()
        packet = compress_vendor_pools("IntegrationVendor", layers)

        # Build a known-good synthesis
        synthesis, _ = _make_valid_synthesis(packet)

        # Validate
        vresult = validate_synthesis(synthesis, packet)
        assert vresult.is_valid, f"Validation failed: {vresult.summary()}"

        # Read
        view = load_synthesis_view(synthesis, "IntegrationVendor")
        assert view.is_v2
        assert view.primary_wedge is not None

        # Inject
        card: dict = {}
        inject_synthesis_into_card(card, view)
        assert "causal_narrative" in card
        assert "synthesis_wedge" in card

    def test_source_id_round_trip(self):
        """Source IDs from compression survive through validation."""
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        layers = _make_layers()
        packet = compress_vendor_pools("RoundTrip", layers)
        synthesis, _ = _make_valid_synthesis(packet)

        # All source_ids in synthesis should be in packet
        vresult = validate_synthesis(synthesis, packet)
        hallucinated = [
            e for e in vresult.errors if e.code == "hallucinated_source_id"
        ]
        assert len(hallucinated) == 0, f"Hallucinated: {hallucinated}"


class TestEdgeCases:
    def test_nested_proof_point_source_id_validated(self):
        """Hallucinated source_id inside nested proof_point is caught."""
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        packet = compress_vendor_pools("Edge", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["competitive_reframes"]["reframes"] = [{
            "incumbent_claim": "test",
            "reframe": "test",
            "proof_point": {
                "field": "f",
                "value": 1,
                "source_id": "FAKE:BAD:SOURCE",
                "interpretation": "test",
            },
            "best_segment": "test",
        }]
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "hallucinated_source_id" in codes

    def test_v1_wedge_not_in_v2_registry(self):
        """Old v1 wedge types ('pricing_shock') don't validate in v2."""
        from atlas_brain.reasoning.wedge_registry import validate_wedge
        # These are v1 wedge types that should NOT be in v2 registry
        for old_wedge in ("pricing_shock", "feature_gap", "acquisition_decay",
                          "leadership_redesign", "integration_break",
                          "support_collapse", "category_disruption",
                          "compliance_gap"):
            assert validate_wedge(old_wedge) is None, f"{old_wedge} should not validate"

    def test_v1_reader_no_wedge_injection(self):
        """V1 synthesis (old wedge types) does not inject wedge fields."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view, inject_synthesis_into_card,
        )
        v1_raw = {
            "causal_narrative": {"primary_wedge": "pricing_shock", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "evidence_window": {"earliest": "2025-12-01", "latest": "2026-03-15"},
        }
        view = load_synthesis_view(v1_raw, "V1Vendor")
        assert view.primary_wedge is None  # old wedge not in v2 registry
        card: dict = {}
        inject_synthesis_into_card(card, view)
        assert "synthesis_wedge" not in card  # not injected
        assert "causal_narrative" in card  # section still injected

    def test_missing_confidence_defaults_to_insufficient(self):
        """Section without confidence field defaults to insufficient."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "causal_narrative": {"primary_wedge": "stable"},  # no confidence
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }
        view = load_synthesis_view(raw, "V")
        assert view.should_suppress("causal_narrative")
        assert not view.is_quotable("causal_narrative")

    def test_empty_slug_defaults_to_unknown(self):
        """Pool items with empty category default to 'unknown' in source_id."""
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = {
            "evidence_vault": {
                "weakness_evidence": [{"category": "", "mention_count": 5}],
                "strength_evidence": [],
                "metric_snapshot": {},
                "provenance": {},
            },
        }
        packet = compress_vendor_pools("SlugTest", layers)
        ev_items = packet.pools.get("evidence_vault", [])
        assert len(ev_items) == 1
        assert ev_items[0].source_ref.source_id == "vault:weakness:unknown"

    def test_value_mismatch_in_proof_point_detected(self):
        """Value mismatch inside a proof_point with extra keys is caught."""
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        packet = compress_vendor_pools("Mismatch", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        # Set active_eval_signals to wrong value but correct source_id
        synthesis["timing_intelligence"]["active_eval_signals"] = {
            "value": 999,
            "source_id": "temporal:signal:evaluation_deadline_signals",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # warning, not error
        codes = [w.code for w in result.warnings]
        assert "value_mismatch" in codes

    def test_switching_real_zero_volume_warns(self):
        """switching_is_real=true with switch_volume=0 and wrong evidence_type."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["switching_is_real"] = True
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        synthesis["migration_proof"]["evidence_type"] = "explicit_switch"
        result = validate_synthesis(synthesis, packet)
        codes = [w.code for w in result.warnings]
        assert "switching_real_but_zero_volume" in codes

    def test_switching_real_zero_volume_active_eval_no_warn(self):
        """switching_is_real=true + volume=0 + evidence_type=active_evaluation is OK."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["switching_is_real"] = True
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        synthesis["migration_proof"]["evidence_type"] = "active_evaluation"
        result = validate_synthesis(synthesis, packet)
        coherence_warns = [w for w in result.warnings if w.code == "switching_real_but_zero_volume"]
        assert len(coherence_warns) == 0

    def test_inject_overrides_archetype_with_wedge(self):
        """Synthesis wedge overrides the old archetype label."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view, inject_synthesis_into_card,
        )
        raw = {
            "schema_version": "2.0",
            "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "meta": {"evidence_window_start": "2026-03-01", "evidence_window_end": "2026-03-18"},
        }
        view = load_synthesis_view(raw, "TestVendor")
        card = {"archetype": "support_collapse"}  # old archetype from stratified reasoner
        inject_synthesis_into_card(card, view)
        # Wedge should override archetype
        assert card["archetype"] == "price_squeeze"
        assert card["archetype_label"] == "Price Squeeze"
        assert card["synthesis_wedge"] == "price_squeeze"

    def test_inject_adds_evidence_depth_warning(self):
        """Thin evidence window adds warning to card."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view, inject_synthesis_into_card,
        )
        raw = {
            "schema_version": "2.0",
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "meta": {"evidence_window_start": "2026-03-15", "evidence_window_end": "2026-03-20"},
        }
        view = load_synthesis_view(raw, "V")
        card: dict = {}
        inject_synthesis_into_card(card, view)
        assert card["evidence_window_days"] == 5
        assert "evidence_depth_warning" in card
        assert "5 days" in card["evidence_depth_warning"]

    def test_inject_no_warning_for_adequate_window(self):
        """Normal evidence window does not add warning."""
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view, inject_synthesis_into_card,
        )
        raw = {
            "schema_version": "2.0",
            "causal_narrative": {"primary_wedge": "stable", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
            "meta": {"evidence_window_start": "2026-02-15", "evidence_window_end": "2026-03-20"},
        }
        view = load_synthesis_view(raw, "V")
        card: dict = {}
        inject_synthesis_into_card(card, view)
        assert card["evidence_window_days"] == 33
        assert "evidence_depth_warning" not in card
