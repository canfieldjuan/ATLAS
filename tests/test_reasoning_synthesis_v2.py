"""Tests for B2B reasoning synthesis v2: wedge registry, pool compression,
validation, and reader contract."""

import json
import time
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


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
            "reviews_in_analysis_window": 120,
            "reviews_in_recent_window": 42,
            "churn_density": 0.35,
            "avg_urgency": 7.1,
            "recommend_yes": 24,
            "recommend_no": 8,
            "recommend_ratio": 0.75,
            "price_complaint_rate": 0.18,
            "dm_churn_rate": 0.21,
            "positive_review_pct": 0.44,
            "displacement_mention_count": 18,
            "keyword_spike_count": 3,
            "avg_rating": 3.2,
        },
        "company_signals": [
            {
                "company_name": "Acme",
                "urgency_score": 9.1,
                "buying_stage": "evaluation",
                "decision_maker": True,
            },
            {
                "company_name": "Bravo",
                "urgency_score": 7.4,
                "buying_stage": "active_purchase",
                "decision_maker": False,
            },
            {
                "company_name": "Charlie",
                "urgency_score": 8.0,
                "buying_stage": "monitoring",
                "decision_maker": True,
            },
        ],
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
            "affected_company_sizes": {
                "avg_seat_count": 120,
                "median_seat_count": 75,
                "max_seat_count": 800,
                "size_distribution": [
                    {"segment": "Mid-Market", "review_count": 22, "churn_rate": 0.31},
                    {"segment": "SMB", "review_count": 4, "churn_rate": 0.2},
                ],
            },
            "budget_pressure": {
                "dm_churn_rate": 0.21,
                "price_increase_rate": 0.12,
                "price_increase_count": 9,
                "annual_spend_signals": ["120k annual spend", "renewal cost pressure"],
                "price_per_seat_signals": ["45 per seat"],
            },
            "contract_segments": [
                {"segment": "enterprise", "count": 50, "churn_rate": 0.20},
            ],
            "usage_duration_segments": [
                {"duration": "1_to_3_years", "count": 18, "churn_rate": 0.42},
            ],
            "top_use_cases_under_pressure": [
                {"use_case": "CRM", "mention_count": 11, "confidence_score": 0.7},
                {"use_case": "automation", "mention_count": 4, "confidence_score": 0.6},
            ],
            "buying_stage_distribution": [
                {"stage": "evaluation", "count": 126},
                {"stage": "active_purchase", "count": 7},
                {"stage": "post_purchase", "count": 9},
            ],
        },
        "temporal": {
            "timeline_signal_summary": {
                "evaluation_deadline_signals": 7,
                "contract_end_signals": 3,
            },
            "immediate_triggers": [
                {
                    "trigger": "Q2 renewal",
                    "type": "deadline",
                    "date": "2026-06-30",
                    "urgency": 8.0,
                },
            ],
            "keyword_spikes": {
                "spike_count": 1,
                "spike_keywords": ["alternative"],
                "keyword_details": [
                    {"keyword": "alternative", "change_pct": 250, "is_spike": True, "volume": 12},
                ],
            },
            "sentiment_trajectory": {
                "declining": 6,
                "stable": 2,
                "improving": 1,
                "total": 9,
                "declining_pct": 0.67,
                "improving_pct": 0.11,
            },
            "evaluation_deadlines": [
                {"label": "Q2 renewal", "urgency": 8.0, "trigger_type": "timeline_signal"},
            ],
            "turning_points": [
                {"trigger": "pricing change", "mentions": 5},
            ],
            "sentiment_tenure": [
                {"tenure": "1_to_3_years", "count": 4},
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
                "active_eval_signal_count": 9,
            },
            "accounts": [
                {
                    "company_name": "Acme Corp",
                    "urgency_score": 0.9,
                    "decision_maker": True,
                    "buying_stage": "evaluation",
                },
                {"company_name": "Globex", "urgency_score": 0.5, "buying_stage": "monitoring"},
            ],
        },
        "reviews": [
            {
                "id": "r1",
                "source": "g2",
                "rating": 2.0,
                "rating_max": 5,
                "summary": "Renewal pricing shock",
                "review_text": "We were quoted $50k at renewal and started evaluating CompetitorA.",
                "pros": "",
                "cons": "",
                "reviewer_title": "VP Finance",
                "reviewer_company": "Acme Corp",
                "reviewed_at": "2026-03-20T00:00:00+00:00",
                "raw_metadata": {"source_weight": 1.0},
                "enrichment": {
                    "churn_signals": {
                        "intent_to_leave": True,
                        "actively_evaluating": True,
                        "contract_renewal_mentioned": True,
                        "migration_in_progress": False,
                    },
                    "reviewer_context": {"decision_maker": True},
                    "would_recommend": False,
                    "pain_categories": [{"category": "pricing", "severity": "primary"}],
                    "evidence_spans": [
                        {
                            "span_id": "review:r1:span:0-20",
                            "_sid": "review:r1:span:0-20",
                            "text": "quoted $50k at renewal",
                            "signal_type": "pricing_backlash",
                            "pain_category": "pricing",
                            "competitor": "CompetitorA",
                            "company_name": "Acme Corp",
                            "reviewer_title": "VP Finance",
                            "time_anchor": "renewal",
                            "numeric_literals": {"currency_mentions": ["$50k"]},
                            "flags": ["explicit_dollar", "named_org", "deadline", "named_competitor"],
                            "replacement_mode": "competitor_switch",
                            "operating_model_shift": "none",
                            "productivity_delta_claim": "unknown",
                        },
                    ],
                },
            },
            {
                "id": "r2",
                "source": "reddit",
                "rating": 4.0,
                "rating_max": 5,
                "summary": "Docs workflow works better",
                "review_text": "We are more productive with docs and async updates than we were in chat.",
                "pros": "",
                "cons": "",
                "reviewer_title": "Ops Lead",
                "reviewer_company": "Globex",
                "reviewed_at": "2026-03-18T00:00:00+00:00",
                "raw_metadata": {"source_weight": 0.7},
                "enrichment": {
                    "churn_signals": {
                        "intent_to_leave": False,
                        "actively_evaluating": False,
                        "contract_renewal_mentioned": False,
                        "migration_in_progress": False,
                    },
                    "reviewer_context": {"decision_maker": False},
                    "would_recommend": True,
                    "sentiment_trajectory": {"direction": "stable_positive"},
                    "pain_categories": [{"category": "pricing", "severity": "primary"}],
                    "evidence_spans": [
                        {
                            "span_id": "review:r2:span:0-20",
                            "_sid": "review:r2:span:0-20",
                            "text": "more productive with docs and async updates",
                            "signal_type": "positive_anchor",
                            "pain_category": "pricing",
                            "competitor": None,
                            "company_name": "Globex",
                            "reviewer_title": "Ops Lead",
                            "time_anchor": None,
                            "numeric_literals": {},
                            "flags": ["named_org", "workflow_substitution", "sync_to_async"],
                            "replacement_mode": "workflow_substitution",
                            "operating_model_shift": "sync_to_async",
                            "productivity_delta_claim": "more_productive",
                        },
                    ],
                },
            },
        ],
    }


class TestPoolHash:
    def test_compute_pool_hash_ignores_non_semantic_metadata(self):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash,
        )

        base = _make_layers()
        changed = deepcopy(base)
        changed["evidence_vault"]["as_of_date"] = "2026-04-01"
        changed["segment"]["as_of_date"] = "2026-04-01"
        changed["temporal"]["as_of_date"] = "2026-04-01"
        changed["displacement"][0]["as_of_date"] = "2026-04-01"
        changed["category"]["as_of_date"] = "2026-04-01"
        changed["accounts"]["as_of_date"] = "2026-04-01"
        changed["evidence_vault"]["metric_snapshot"]["snapshot_date"] = "2026-04-01"
        changed["evidence_vault"]["provenance"]["enrichment_window_start"] = "2026-01-01"
        changed["evidence_vault"]["provenance"]["enrichment_window_end"] = "2026-04-01"

        assert _compute_pool_hash(base) == _compute_pool_hash(changed)

    def test_compute_pool_hash_changes_on_semantic_pool_delta(self):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash,
        )

        base = _make_layers()
        changed = deepcopy(base)
        changed["segment"]["budget_pressure"]["annual_spend_signals"].append(
            "250k annual spend",
        )

        assert _compute_pool_hash(base) != _compute_pool_hash(changed)


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
            if pool_name in {"precomputed_aggregates", "section_packets"}:
                continue
            assert isinstance(items, list)
            for item in items:
                assert "_sid" in item, f"Missing _sid in {pool_name}"
        # Aggregates have {value, _sid} wrappers
        aggs = payload["precomputed_aggregates"]
        for label, wrapper in aggs.items():
            assert "value" in wrapper
            assert "_sid" in wrapper

    def test_packet_includes_witness_pack(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        packet = compress_vendor_pools("TestVendor", _make_layers())

        assert packet.witness_pack
        assert packet.section_packets["causal_packet"]["witness_ids"]
        assert any(witness["witness_id"].startswith("witness:r1") for witness in packet.witness_pack)

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

    def test_missing_displacement_layer_emits_zero_aggregates(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        layers.pop("displacement")
        packet = compress_vendor_pools("TestVendor", layers)
        agg_map = {a.label: a for a in packet.aggregates}
        assert "displacement" not in packet.pools
        assert agg_map["total_explicit_switches"].value == 0
        assert agg_map["total_active_evaluations"].value == 0

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

    def test_canonical_evidence_vault_key_uses_specific_source_id(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = {
            "evidence_vault": {
                "weakness_evidence": [
                    {"key": "pricing", "label": "Pricing opacity", "mention_count_total": 6},
                ],
                "strength_evidence": [],
                "metric_snapshot": {},
                "provenance": {},
            },
        }
        packet = compress_vendor_pools("CanonicalVendor", layers)
        ev_items = packet.pools.get("evidence_vault", [])
        assert len(ev_items) == 1
        assert ev_items[0].source_ref.source_id == "vault:weakness:pricing"

    def test_account_active_eval_aggregate_present(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        packet = compress_vendor_pools("TestVendor", _make_layers())
        agg_map = {a.label: a.value for a in packet.aggregates}
        assert agg_map["active_eval_signal_count"] == 9
        assert agg_map["segment_active_eval_signal_count"] == 133

    def test_category_regime_metrics_are_emitted_and_carried_into_contracts(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["category"]["council_summary"] = {
            "market_regime": "feature_competition",
            "winner": "CompetitorA",
            "loser": "TestVendor",
            "conclusion": "The category is moving toward feature-led evaluation.",
            "confidence": 0.62,
            "key_insights": ["Feature depth is outweighing price in top evaluations."],
            "durability_assessment": "Likely durable through the next 2 quarters",
            "segment_dynamics": {"mid_market": "shifting faster"},
            "category_default": "feature_depth",
        }
        layers["category"]["market_regime"]["avg_churn_velocity"] = 0.7
        layers["category"]["market_regime"]["avg_price_pressure"] = 0.0
        packet = compress_vendor_pools("TestVendor", layers)
        agg_map = {a.source_id: a.value for a in packet.aggregates}
        assert agg_map["category:regime:confidence"] == 0.7
        assert agg_map["category:regime:avg_churn_velocity"] == 0.7
        assert agg_map["category:regime:avg_price_pressure"] == 0.0

        contracts = build_reasoning_contracts({"category_reasoning": {}}, packet)
        category = contracts["category_reasoning"]
        assert category["market_regime"] == "feature_competition"
        assert category["avg_churn_velocity"] == 0.7
        assert category["avg_price_pressure"] == 0.0
        assert category["outlier_vendors"] == []
        assert category["segment_dynamics"] == {"mid_market": "shifting faster"}
        assert category["category_default"] == "feature_depth"

    def test_temporal_deadline_source_id_uses_timeline_when_label_missing(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        layers["temporal"]["evaluation_deadlines"] = [
            {
                "decision_timeline": "within_quarter",
                "trigger_type": "timeline_signal",
                "urgency": 8.0,
            },
        ]
        packet = compress_vendor_pools("TestVendor", layers)
        sids = packet.source_ids()
        assert "temporal:timeline_signal:within_quarter" in sids

    def test_temporal_sentiment_aggregates_and_items_are_present(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        agg_map = {a.source_id: a.value for a in packet.aggregates}
        assert agg_map["temporal:sentiment:declining_count"] == 6
        assert agg_map["temporal:sentiment:improving_count"] == 1
        assert agg_map["temporal:sentiment:declining_pct"] == 0.67
        sids = packet.source_ids()
        assert "temporal:sentiment:declining" in sids

    def test_temporal_turning_points_and_tenure_become_items(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        sids = packet.source_ids()
        assert "temporal:turning_point:pricing_change" in sids
        assert "temporal:tenure:1_to_3_years" in sids

    def test_account_summary_fallbacks_fill_missing_counts(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        layers["accounts"] = {
            "summary": {
                "total_accounts": 2,
                "decision_maker_count": 1,
            },
            "accounts": [
                {
                    "company_name": "Acme Corp",
                    "urgency_score": 8.0,
                    "decision_maker": True,
                    "buying_stage": "evaluation",
                },
                {
                    "company_name": "Globex",
                    "urgency_score": 4.0,
                    "decision_maker": False,
                    "buying_stage": "monitoring",
                },
            ],
        }
        packet = compress_vendor_pools("TestVendor", layers)
        agg_map = {a.label: a.value for a in packet.aggregates}
        assert agg_map["high_intent_count"] == 1
        assert agg_map["active_eval_signal_count"] == 1

    def test_segment_active_eval_aggregate_tracks_buying_stage_distribution(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        packet = compress_vendor_pools("TestVendor", layers)
        agg_map = {a.source_id: a.value for a in packet.aggregates}
        assert agg_map["segment:aggregate:active_eval_signal_count"] == 133

    def test_low_sample_segments_are_suppressed(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        layers = _make_layers()
        layers["segment"] = {
            "affected_departments": [
                {"department": "support", "churn_rate": 1.0, "review_count": 2},
                {"department": "engineering", "churn_rate": 0.2, "review_count": 8},
            ],
            "affected_roles": [
                {"role_type": "admin", "churn_rate": 0.8, "review_count": 3},
            ],
            "contract_segments": [
                {"segment": "enterprise", "count": 4, "churn_rate": 0.9},
            ],
        }
        packet = compress_vendor_pools("TestVendor", layers)
        segment_items = packet.pools.get("segment", [])
        keys = {item.source_ref.source_id for item in segment_items}
        assert "segment:department:support" not in keys
        assert "segment:role:admin" not in keys
        assert "segment:contract:enterprise" not in keys
        assert "segment:department:engineering" in keys

    def test_segment_compression_includes_duration_and_use_case_items(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        keys = {item.source_ref.source_id for item in packet.pools.get("segment", [])}
        assert "segment:duration:1_to_3_years" in keys
        assert "segment:use_case:crm" in keys
        assert "segment:use_case:automation" not in keys

    def test_segment_role_compression_preserves_strategic_priority_order(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        layers["segment"]["affected_roles"] = [
            {"role_type": "end_user", "churn_rate": 0.0, "review_count": 40},
            {"role_type": "economic_buyer", "churn_rate": 0.0, "review_count": 15},
            {"role_type": "evaluator", "churn_rate": 0.0, "review_count": 10},
            {"role_type": "champion", "churn_rate": 0.0, "review_count": 4},
        ]

        packet = compress_vendor_pools("TestVendor", layers)
        role_keys = [
            item.source_ref.source_id
            for item in packet.pools.get("segment", [])
            if item.source_ref.kind == "role"
        ]
        assert role_keys[:3] == [
            "segment:role:economic_buyer",
            "segment:role:end_user",
            "segment:role:evaluator",
        ]

    def test_segment_compression_includes_size_and_budget_aggregates(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        agg_map = {a.source_id: a.value for a in packet.aggregates}
        assert agg_map["segment:budget:price_increase_count"] == 9
        assert agg_map["segment:budget:annual_spend_signal_count"] == 2
        assert agg_map["segment:budget:price_per_seat_signal_count"] == 1
        assert agg_map["segment:size:avg_seat_count"] == 120
        assert agg_map["segment:size:median_seat_count"] == 75
        assert agg_map["segment:size:max_seat_count"] == 800
        assert agg_map["vault:weakness:pricing:mention_count_total"] == 25
        assert agg_map["vault:strength:ease_of_use:mention_count_total"] == 30
        assert agg_map["vault:metric:reviews_in_analysis_window"] == 120
        assert agg_map["vault:metric:reviews_in_recent_window"] == 42
        assert agg_map["vault:metric:avg_urgency"] == 7.1
        assert agg_map["vault:metric:recommend_yes"] == 24
        assert agg_map["vault:metric:recommend_no"] == 8
        assert agg_map["vault:metric:recommend_ratio"] == 0.75
        assert agg_map["vault:metric:price_complaint_rate"] == 0.18
        assert agg_map["vault:metric:dm_churn_rate"] == 0.21
        assert agg_map["vault:metric:positive_review_pct"] == 0.44
        assert agg_map["vault:metric:keyword_spike_count"] == 3
        assert agg_map["vault:company_signals:count"] == 3
        assert agg_map["vault:company_signals:high_urgency_count"] == 2
        assert agg_map["vault:company_signals:evaluation_count"] == 1
        assert agg_map["vault:company_signals:active_purchase_count"] == 1
        assert agg_map["vault:company_signals:decision_maker_count"] == 2
        assert agg_map["segment:reach:department:engineering"] == 40
        assert agg_map["segment:reach:contract:enterprise"] == 50
        assert agg_map["segment:reach:duration:1_to_3_years"] == 18
        assert agg_map["segment:reach:use_case:crm"] == 11
        assert agg_map["segment:reach:size:mid_market"] == 22

    def test_segment_compression_preserves_size_items_when_pool_is_crowded(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        layers["segment"]["affected_departments"] = [
            {"department": "engineering", "churn_rate": 0.82, "review_count": 40},
            {"department": "operations", "churn_rate": 0.78, "review_count": 28},
            {"department": "finance", "churn_rate": 0.74, "review_count": 22},
        ]
        layers["segment"]["affected_roles"] = [
            {"role_type": "economic_buyer", "churn_rate": 0.30, "review_count": 15},
            {"role_type": "evaluator", "churn_rate": 0.26, "review_count": 10},
            {"role_type": "end_user", "churn_rate": 0.18, "review_count": 40},
        ]
        layers["segment"]["contract_segments"] = [
            {"segment": "enterprise_high", "count": 50, "churn_rate": 0.65},
            {"segment": "mid_market", "count": 24, "churn_rate": 0.58},
        ]
        layers["segment"]["usage_duration_segments"] = [
            {"duration": "1 year", "count": 18, "churn_rate": 0.61},
            {"duration": "2 years", "count": 14, "churn_rate": 0.55},
        ]
        layers["segment"]["top_use_cases_under_pressure"] = [
            {"use_case": "CRM", "mention_count": 30, "confidence_score": 0.8},
            {"use_case": "ticketing", "mention_count": 20, "confidence_score": 0.7},
            {"use_case": "analytics", "mention_count": 15, "confidence_score": 0.6},
        ]
        layers["segment"]["affected_company_sizes"]["size_distribution"] = [
            {"segment": "Mid-Market", "review_count": 22, "churn_rate": 0.01},
        ]

        packet = compress_vendor_pools("TestVendor", layers)
        keys = {item.source_ref.source_id for item in packet.pools.get("segment", [])}

        assert "segment:size:mid_market" in keys
        assert "segment:role:economic_buyer" in keys
        assert "segment:department:engineering" in keys
        assert "segment:contract:enterprise_high" in keys
        assert "segment:duration:1_year" in keys
        assert "segment:use_case:crm" in keys

    def test_evidence_vault_ranking_prefers_recent_accelerating_weakness(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        layers["evidence_vault"]["weakness_evidence"] = [
            {
                "key": "pricing",
                "mention_count_total": 10,
                "mention_count_recent": 1,
                "trend": {"direction": "stable"},
            },
            {
                "key": "support",
                "mention_count_total": 10,
                "mention_count_recent": 8,
                "trend": {"direction": "accelerating"},
            },
        ]

        packet = compress_vendor_pools("TestVendor", layers)
        ev_items = packet.pools.get("evidence_vault", [])
        assert ev_items[0].source_ref.source_id == "vault:weakness:support"

    def test_build_segment_intelligence_prefers_known_roles_over_unknown(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            build_segment_intelligence,
        )

        seg = build_segment_intelligence(
            "TestVendor",
            buyer_auth={
                "role_types": {"unknown": 12, "economic_buyer": 3, "evaluator": 2},
                "buying_stages": {"evaluation": 5},
            },
        )
        roles = seg["affected_roles"]
        assert roles[0]["role_type"] == "economic_buyer"
        assert {item["role_type"] for item in roles} == {"economic_buyer", "evaluator"}
        assert roles[0]["priority_score"] > roles[1]["priority_score"]

    def test_build_segment_intelligence_prioritizes_strategic_roles_over_raw_volume(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            build_segment_intelligence,
        )

        seg = build_segment_intelligence(
            "TestVendor",
            buyer_auth={
                "role_types": {
                    "end_user": 40,
                    "economic_buyer": 15,
                    "evaluator": 10,
                    "champion": 4,
                },
                "buying_stages": {"evaluation": 9},
            },
        )
        roles = seg["affected_roles"]
        assert [item["role_type"] for item in roles[:3]] == [
            "economic_buyer",
            "end_user",
            "evaluator",
        ]
        assert roles[0]["review_count"] == 15
        assert roles[0]["priority_score"] > roles[1]["priority_score"]
        assert roles[1]["priority_score"] > roles[2]["priority_score"]

    def test_build_segment_intelligence_preserves_role_stage_linkage_and_size_distribution(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            build_segment_intelligence,
        )

        seg = build_segment_intelligence(
            "TestVendor",
            buyer_auth={
                "role_types": {"economic_buyer": 3, "evaluator": 2},
                "buying_stages": {"evaluation": 4, "active_purchase": 1},
                "role_buying_stages": {
                    "economic_buyer": {"active_purchase": 2, "evaluation": 1},
                    "evaluator": {"evaluation": 2},
                },
            },
            company_size_entries=[
                {"segment": "Mid-Market", "review_count": 8, "churn_rate": 0.25},
                {"segment": "SMB", "review_count": 3, "churn_rate": 0.1},
            ],
        )
        roles = {item["role_type"]: item for item in seg["affected_roles"]}
        assert roles["economic_buyer"]["top_buying_stage"] == "active_purchase"
        assert roles["evaluator"]["top_buying_stage"] == "evaluation"
        assert seg["affected_company_sizes"]["size_distribution"] == [
            {"segment": "Mid-Market", "review_count": 8, "churn_rate": 0.25},
            {"segment": "SMB", "review_count": 3, "churn_rate": 0.1},
        ]

    def test_build_segment_intelligence_prefers_known_stage_over_unknown(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            build_segment_intelligence,
        )

        seg = build_segment_intelligence(
            "TestVendor",
            buyer_auth={
                "role_types": {"end_user": 5},
                "buying_stages": {"unknown": 10, "evaluation": 2},
                "role_buying_stages": {
                    "end_user": {"unknown": 4, "evaluation": 2},
                },
            },
        )
        assert seg["affected_roles"][0]["top_buying_stage"] == "evaluation"


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

    def test_prompt_distinguishes_switch_and_displacement_volume(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "active_evaluation_volume" in BATTLE_CARD_REASONING_PROMPT
        assert "displacement_mention_volume" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_allows_empty_segments_and_reframes(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "priority_segments: []" in BATTLE_CARD_REASONING_PROMPT
        assert "reframes: []" in BATTLE_CARD_REASONING_PROMPT
        assert "Do not emit labels like" in BATTLE_CARD_REASONING_PROMPT
        assert "pipeline jargon like" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_requires_metric_backed_proof_points(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "proof_point.source_id" in BATTLE_CARD_REASONING_PROMPT
        assert "metric_ledger" in BATTLE_CARD_REASONING_PROMPT
        assert "witness_highlights" in BATTLE_CARD_REASONING_PROMPT
        assert "confirmed explicit switches only" in BATTLE_CARD_REASONING_PROMPT

    def test_prompt_rejects_unknown_vault_citations(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert "vault:weakness:unknown" in BATTLE_CARD_REASONING_PROMPT
        assert "move it to ``data_gaps``" in BATTLE_CARD_REASONING_PROMPT

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

    def test_prompt_schema_is_contracts_first(self):
        from atlas_brain.reasoning.single_pass_prompts.battle_card_reasoning import (
            BATTLE_CARD_REASONING_PROMPT,
        )
        assert '"reasoning_shape": "contracts_first_v1"' in BATTLE_CARD_REASONING_PROMPT
        assert '"reasoning_contracts"' in BATTLE_CARD_REASONING_PROMPT
        assert '"vendor_core_reasoning"' in BATTLE_CARD_REASONING_PROMPT
        assert '"displacement_reasoning"' in BATTLE_CARD_REASONING_PROMPT

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


# ---- Module 3b: Neutral Reasoning Synthesis Prompt ----

class TestNeutralPrompt:
    def test_prompt_imports(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
            REASONING_SYNTHESIS_PROMPT_VERSION,
        )
        assert isinstance(REASONING_SYNTHESIS_PROMPT, str)
        assert isinstance(REASONING_SYNTHESIS_PROMPT_VERSION, str)
        assert len(REASONING_SYNTHESIS_PROMPT_VERSION) == 8

    def test_prompt_contains_wedge_values(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        from atlas_brain.reasoning.wedge_registry import WEDGE_ENUM_VALUES
        for val in WEDGE_ENUM_VALUES:
            assert val in REASONING_SYNTHESIS_PROMPT

    def test_prompt_requires_sid_and_source_id(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        assert "_sid" in REASONING_SYNTHESIS_PROMPT
        assert "source_id" in REASONING_SYNTHESIS_PROMPT

    def test_prompt_no_sales_language(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        assert "Sales reps" not in REASONING_SYNTHESIS_PROMPT
        assert "AEs" not in REASONING_SYNTHESIS_PROMPT
        assert "revenue leaders" not in REASONING_SYNTHESIS_PROMPT
        assert "sales battle cards" not in REASONING_SYNTHESIS_PROMPT

    def test_prompt_has_governance_awareness(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        assert "contradiction_rows" in REASONING_SYNTHESIS_PROMPT
        assert "coverage_gaps" in REASONING_SYNTHESIS_PROMPT
        assert "retention_proof" in REASONING_SYNTHESIS_PROMPT
        assert "minority_signals" in REASONING_SYNTHESIS_PROMPT
        assert "metric_ledger" not in REASONING_SYNTHESIS_PROMPT

    def test_prompt_mentions_witness_packets(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        assert "witness_pack" in REASONING_SYNTHESIS_PROMPT
        assert "section_packets" in REASONING_SYNTHESIS_PROMPT

    def test_prompt_has_phase3_contract_sections(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        assert "why_they_stay" in REASONING_SYNTHESIS_PROMPT
        assert "confidence_posture" in REASONING_SYNTHESIS_PROMPT
        assert "switch_triggers" in REASONING_SYNTHESIS_PROMPT
        assert "neutralization" in REASONING_SYNTHESIS_PROMPT

    def test_prompt_is_contracts_first(self):
        from atlas_brain.reasoning.single_pass_prompts.reasoning_synthesis import (
            REASONING_SYNTHESIS_PROMPT,
        )
        assert '"reasoning_contracts"' in REASONING_SYNTHESIS_PROMPT
        assert '"vendor_core_reasoning"' in REASONING_SYNTHESIS_PROMPT
        assert '"displacement_reasoning"' in REASONING_SYNTHESIS_PROMPT
        assert '"reasoning_shape": "contracts_first_v1"' in REASONING_SYNTHESIS_PROMPT

    def test_synthesis_task_uses_neutral_prompt(self):
        """The synthesis task should import the neutral prompt, not battle card."""
        import ast
        with open("atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py") as f:
            source = f.read()
        assert "reasoning_synthesis" in source
        assert "REASONING_SYNTHESIS_PROMPT" in source
        assert "BATTLE_CARD_REASONING_PROMPT" not in source

    def test_battle_card_prompt_not_imported_by_synthesis_task(self):
        """battle_card_reasoning should only be used for battle card rendering."""
        import ast
        with open("atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py") as f:
            source = f.read()
        assert "battle_card_reasoning" not in source


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
    eval_signals_sid = "accounts:summary:active_eval_signal_count"
    eval_signals_val = agg_sids.get(eval_signals_sid, 9)
    active_eval_volume_sid = "displacement:aggregate:total_active_evaluations"
    active_eval_volume_val = agg_sids.get(active_eval_volume_sid, 11)
    mention_volume_sid = "vault:metric:displacement_mention_count"
    mention_volume_val = agg_sids.get(mention_volume_sid, 18)

    # Pick a valid pool item sid for citations
    pool_sid = "vault:weakness:pricing"
    proof_sid = "vault:weakness:pricing:mention_count_total"
    witness_sid = next(
        (
            str(witness.get("witness_id") or witness.get("_sid"))
            for witness in (packet.witness_pack or [])
            if str(witness.get("witness_id") or witness.get("_sid") or "").strip()
        ),
        pool_sid,
    )

    return {
        "schema_version": "2.1",
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
            "citations": [pool_sid, witness_sid],
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
                    "citations": [pool_sid, witness_sid],
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
                        "source_id": proof_sid,
                        "interpretation": "High pricing complaints",
                    },
                    "best_segment": "Enterprise Ops",
                    "citations": [pool_sid, witness_sid],
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
            "active_evaluation_volume": {
                "value": active_eval_volume_val,
                "source_id": active_eval_volume_sid,
            },
            "displacement_mention_volume": {
                "value": mention_volume_val,
                "source_id": mention_volume_sid,
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
                    "source_id": witness_sid,
                },
            ],
            "evaluation_vs_switching": "More evaluating than switching",
            "confidence": "high",
            "data_gaps": [],
            "citations": [pool_sid, witness_sid, total_switches_sid],
        },
        "account_reasoning": {
            "confidence": "medium",
            "data_gaps": [],
            "market_summary": "Account activity is concentrated in active evaluation stages.",
            "total_accounts": {
                "value": total_accounts_val,
                "source_id": total_accounts_sid,
            },
            "high_intent_count": {
                "value": agg_sids.get("accounts:summary:high_intent_count", 9),
                "source_id": "accounts:summary:high_intent_count",
            },
            "active_eval_count": {
                "value": eval_signals_val,
                "source_id": eval_signals_sid,
            },
            "top_accounts": [],
            "citations": [],
        },
        "category_reasoning": {
            "market_regime": "fragmented",
            "narrative": "Category demand is fragmented with mixed buyer urgency.",
            "confidence": "medium",
            "data_gaps": [],
            "citations": [],
        },
        "vendor_core_reasoning": {
            "why_they_stay": {
                "strengths": [
                    {"area": "Ecosystem breadth", "evidence": "Multiple integrations"},
                ],
            },
            "confidence_posture": {
                "limits": ["Limited insider signals"],
            },
        },
        "meta": {
            "evidence_window_start": "2025-12-01",
            "evidence_window_end": "2026-03-15",
            "total_evidence_items": 20,
            "synthesized_at": "2026-03-19T00:00:00Z",
        },
    }, packet


class TestValidation:
    def test_validate_contracts_first_payload_with_account_reasoning(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        agg_sids = {a.source_id: a.value for a in packet.aggregates}
        contracts_first = {
            "schema_version": "2.2",
            "reasoning_shape": "contracts_first_v1",
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": synthesis["causal_narrative"],
                    "segment_playbook": synthesis["segment_playbook"],
                    "timing_intelligence": synthesis["timing_intelligence"],
                },
                "displacement_reasoning": {
                    "schema_version": "v1",
                    "competitive_reframes": synthesis["competitive_reframes"],
                    "migration_proof": synthesis["migration_proof"],
                },
                "account_reasoning": {
                    "confidence": "medium",
                    "data_gaps": [],
                    "market_summary": "Account activity is concentrated in active evaluation stages.",
                    "total_accounts": {
                        "value": agg_sids.get("accounts:summary:total_accounts", 42),
                        "source_id": "accounts:summary:total_accounts",
                    },
                    "high_intent_count": {
                        "value": agg_sids.get("accounts:summary:high_intent_count", 9),
                        "source_id": "accounts:summary:high_intent_count",
                    },
                    "active_eval_count": {
                        "value": agg_sids.get("accounts:summary:active_eval_signal_count", 9),
                        "source_id": "accounts:summary:active_eval_signal_count",
                    },
                    "top_accounts": [],
                    "citations": [],
                },
                "category_reasoning": synthesis["category_reasoning"],
            },
            "meta": synthesis["meta"],
        }

        result = validate_synthesis(contracts_first, packet)
        assert result.is_valid, result.summary()

    def test_normalize_shorthand_aggregate_source_ids(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["switch_volume"]["source_id"] = "total_explicit_switches"
        synthesis["migration_proof"]["active_evaluation_volume"]["source_id"] = "total_active_evaluations"
        synthesis["migration_proof"]["displacement_mention_volume"]["source_id"] = "displacement_mention_count"
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["migration_proof"]["switch_volume"]["source_id"] == "displacement:aggregate:total_explicit_switches"
        assert synthesis["migration_proof"]["active_evaluation_volume"]["source_id"] == "displacement:aggregate:total_active_evaluations"
        assert synthesis["migration_proof"]["displacement_mention_volume"]["source_id"] == "vault:metric:displacement_mention_count"
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_signal_count_alias_to_count_variant(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["account_reasoning"]["active_eval_count"]["source_id"] = (
            "accounts:summary:active_eval_count"
        )
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["account_reasoning"]["active_eval_count"]["source_id"] == (
            "accounts:summary:active_eval_signal_count"
        )
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_switch_volume_to_explicit_switch_aggregate(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 18,
            "source_id": "vault:metric:displacement_mention_count",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["migration_proof"]["switch_volume"] == {
            "value": 5,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_displacement_mention_volume_to_matching_allowed_aggregate(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": 5,
            "source_id": "vault:metric:displacement_mention_count",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["migration_proof"]["displacement_mention_volume"] == {
            "value": 5,
            "source_id": "category:aggregate:displacement_flow_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_active_evaluation_volume_to_matching_allowed_aggregate(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["active_evaluation_volume"] = {
            "value": 133,
            "source_id": "displacement:aggregate:total_active_evaluations",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["migration_proof"]["active_evaluation_volume"] == {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_segment_estimated_reach_allows_segment_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 50,
            "source_id": "segment:reach:contract:enterprise",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_legacy_segment_item_source_id_to_segment_reach_aggregate(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 50,
            "source_id": "segment:contract:enterprise",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] == {
            "value": 50,
            "source_id": "segment:reach:contract:enterprise",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_valid_segment_item_source_id_when_field_requires_aggregate(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 50,
            "source_id": "segment:contract:enterprise",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"]["source_id"] == (
            "segment:reach:contract:enterprise"
        )

    def test_normalize_segment_reach_from_matching_disallowed_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 50,
            "source_id": "vault:weakness:pricing:mention_count_recent",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] == {
            "value": 50,
            "source_id": "segment:reach:contract:enterprise",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_segment_reach_value_to_aggregate_when_source_is_already_canonical(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 5,
            "source_id": "segment:reach:contract:enterprise",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] == {
            "value": 50,
            "source_id": "segment:reach:contract:enterprise",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_valid_vault_item_source_id_when_proof_point_requires_aggregate(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"] = (
            "vault:weakness:pricing"
        )
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"] == (
            "vault:weakness:pricing:mention_count_total"
        )

    def test_normalize_vault_underscore_metric_aliases(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"] = (
            "vault:weakness:pricing_mention_count_total"
        )
        synthesis["migration_proof"]["primary_switch_driver"]["source_id"] = (
            "vault:weakness:pricing_mention_count_total"
        )
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"] == (
            "vault:weakness:pricing:mention_count_total"
        )
        assert synthesis["migration_proof"]["primary_switch_driver"]["source_id"] == (
            "vault:weakness:pricing:mention_count_total"
        )

    def test_normalize_aggregate_required_wrapper_value_to_packet_value(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
        )
        synthesis, packet = _make_valid_synthesis()
        expected = next(
            agg.value
            for agg in packet.aggregates
            if agg.source_id == "vault:weakness:pricing:mention_count_total"
        )
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"] = {
            "value": 3.68,
            "source_id": "vault:weakness:pricing:mention_count_total",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["competitive_reframes"]["reframes"][0]["proof_point"] == {
            "value": expected,
            "source_id": "vault:weakness:pricing:mention_count_total",
        }

    def test_normalize_clears_priority_segments_without_segment_reach_support(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import TrackedAggregate
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        packet.aggregates = [
            agg for agg in packet.aggregates
            if not agg.source_id.startswith("segment:reach:")
        ] + [
            TrackedAggregate(
                label="avg_seat_count",
                value=120,
                source_id="segment:size:avg_seat_count",
            ),
        ]
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 120,
            "source_id": "segment:size:avg_seat_count",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["segment_playbook"]["priority_segments"] == []
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_drops_priority_segment_with_too_small_unmappable_reach(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 4,
            "source_id": "accounts:summary:active_eval_signal_count",
        }
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["segment_playbook"]["priority_segments"] == []
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_accounts_alias_maps_to_account_summary_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["account_reasoning"]["top_accounts"] = [
            {
                "name": "Acme Corp",
                "intent_score": 0.9,
                "source_id": "accounts",
            },
        ]
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["account_reasoning"]["top_accounts"][0]["source_id"] == (
            "accounts:summary:total_accounts"
        )
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_validate_missing_account_and_category_sections_are_errors(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis.pop("account_reasoning", None)
        synthesis.pop("category_reasoning", None)
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "missing_section" in codes

    def test_segment_estimated_reach_rejects_generic_segment_aggregate_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 50,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "invalid_field_source" in codes

    def test_valid_passes(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, f"Expected valid, got: {result.summary()}"

    def test_timing_active_eval_allows_segment_aggregate_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["timing_intelligence"]["active_eval_signals"] = {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_timing_active_eval_allows_displacement_aggregate_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["timing_intelligence"]["active_eval_signals"] = {
            "value": 8,
            "source_id": "displacement:aggregate:total_active_evaluations",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_build_persistable_synthesis_normalizes_migration_evidence_type(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        synthesis["migration_proof"]["active_evaluation_volume"] = {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }
        synthesis["migration_proof"]["evidence_type"] = "insufficient_data"
        synthesis["migration_proof"]["switching_is_real"] = False

        persisted = build_persistable_synthesis(synthesis, packet)
        migration = (
            ((persisted.get("reasoning_contracts") or {}).get("displacement_reasoning") or {})
            .get("migration_proof")
            or {}
        )
        assert migration["evidence_type"] == "active_evaluation"
        assert migration["switching_is_real"] is False

    def test_build_persistable_synthesis_backfills_migration_citations(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["citations"] = []
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        synthesis["migration_proof"]["active_evaluation_volume"] = {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": 18,
            "source_id": "vault:metric:displacement_mention_count",
        }

        persisted = build_persistable_synthesis(synthesis, packet)
        migration = (
            ((persisted.get("reasoning_contracts") or {}).get("displacement_reasoning") or {})
            .get("migration_proof")
            or {}
        )
        assert "displacement:aggregate:total_explicit_switches" in migration["citations"]
        assert "segment:aggregate:active_eval_signal_count" in migration["citations"]
        assert "vault:metric:displacement_mention_count" in migration["citations"]

    def test_build_persistable_synthesis_adds_scope_manifest_and_atoms(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)

        persisted = build_persistable_synthesis(synthesis, packet)

        scope_manifest = persisted.get("scope_manifest") or {}
        assert scope_manifest["schema_version"] == "v1"
        assert scope_manifest["selection_strategy"] == "vendor_facet_packet_v1"
        assert scope_manifest["witnesses_in_scope"] >= 1
        atoms = persisted.get("reasoning_atoms") or {}
        assert atoms["schema_version"] == "v1"
        assert atoms["theses"]
        assert atoms["timing_windows"]
        assert atoms["proof_points"]
        assert atoms["coverage_limits"]
        first_thesis = atoms["theses"][0]
        assert "metric_ids" in first_thesis
        assert "witness_ids" in first_thesis
        assert "source_ids" in first_thesis
        assert "evidence_count" in first_thesis
        assert "last_supported_at" in first_thesis

    def test_build_reasoning_delta_detects_changes(self):
        from atlas_brain.autonomous.tasks._b2b_reasoning_atoms import (
            build_reasoning_delta,
        )
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )

        packet = compress_vendor_pools("TestVendor", _make_layers())
        current_synthesis, _ = _make_valid_synthesis(packet)
        previous_synthesis, _ = _make_valid_synthesis(packet)
        current = build_persistable_synthesis(current_synthesis, packet)
        previous = build_persistable_synthesis(previous_synthesis, packet)
        previous["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] = (
            "support_collapse"
        )
        previous["reasoning_contracts"]["displacement_reasoning"]["migration_proof"]["top_destination"] = {
            "value": "CompetitorB",
            "source_id": "displacement:flow:competitorb",
        }
        previous["reasoning_atoms"]["account_signals"] = []
        delta = build_reasoning_delta(
            current,
            previous,
            current_as_of_date="2026-04-06",
            previous_as_of_date="2026-04-05",
        )

        assert delta["changed"] is True
        assert delta["wedge_changed"] is True
        assert delta["top_destination_changed"] is True
        assert delta["new_account_signals"]
        assert delta["previous_as_of_date"] == "2026-04-05"

    def test_migration_active_eval_allows_segment_aggregate_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["active_evaluation_volume"] = {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_placeholder_named_example_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "Unknown SMB customer",
                "evidence": "Quoted expensive seats",
                "source_type": "review_site",
                "quotable": True,
                "source_id": "displacement:flow:competitora",
            },
        ]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid
        assert any(f.code == "placeholder_named_example" for f in result.warnings)

    def test_generic_reviewer_named_example_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "G2 reviewer",
                "evidence": "Quoted pricing pain",
                "source_type": "review_site",
                "quotable": True,
                "source_id": "vault:weakness:pricing",
            },
        ]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid
        assert any(f.code == "placeholder_named_example" for f in result.warnings)

    def test_segment_style_named_example_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "Cost-conscious support team",
                "evidence": "Considering competitor due to price",
                "source_type": "review_site",
                "quotable": True,
                "source_id": "displacement:flow:competitora",
            },
        ]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid
        assert any(f.code == "placeholder_named_example" for f in result.warnings)

    def test_tool_style_named_example_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "Custom ChatGPT Integration",
                "evidence": "Seeking more control over AI features",
                "source_type": "inferred",
                "quotable": False,
                "source_id": "displacement:flow:competitora",
            },
        ]
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid
        assert any(f.code == "placeholder_named_example" for f in result.warnings)

    def test_contracts_first_shape_validates(self):
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        persisted = build_persistable_synthesis(synthesis, packet)

        result = validate_synthesis(persisted, packet)
        assert result.is_valid, f"Expected valid, got: {result.summary()}"

    def test_contracts_first_validation_ignores_stale_flat_sections(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["reasoning_contracts"] = {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "high",
                    "citations": ["vault:weakness:pricing"],
                },
                "segment_playbook": synthesis["segment_playbook"],
                "timing_intelligence": synthesis["timing_intelligence"],
            },
            "displacement_reasoning": {
                "schema_version": "v1",
                "competitive_reframes": synthesis["competitive_reframes"],
                "migration_proof": synthesis["migration_proof"],
            },
        }
        synthesis["causal_narrative"] = {
            "primary_wedge": "totally_made_up",
            "confidence": "high",
        }

        result = validate_synthesis(synthesis, packet)

        assert result.is_valid, f"Expected valid, got: {result.summary()}"
        assert "invalid_wedge" not in [e.code for e in result.errors]

    def test_contracts_first_validation_does_not_backfill_missing_section_from_flat(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["reasoning_contracts"] = {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": synthesis["causal_narrative"],
                "segment_playbook": synthesis["segment_playbook"],
                "timing_intelligence": synthesis["timing_intelligence"],
            },
        }
        synthesis["migration_proof"] = {
            "confidence": "high",
            "evidence_type": "explicit_switch",
        }
        synthesis.pop("competitive_reframes", None)

        result = validate_synthesis(synthesis, packet)

        assert not result.is_valid
        assert "missing_section" in [e.code for e in result.errors]

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

    def test_numeric_equivalent_value_does_not_warn(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["value"] = 25.0
        result = validate_synthesis(synthesis, packet)
        proof_point_warnings = [
            w for w in result.warnings
            if w.path == "competitive_reframes.reframes[0].proof_point"
        ]
        assert not any(w.code == "value_mismatch" for w in proof_point_warnings)

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
        synthesis["migration_proof"]["confidence"] = "medium"
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid  # Warning
        codes = [w.code for w in result.warnings]
        assert "invalid_evidence_type" in codes

    def test_invalid_field_source_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 5,
            "source_id": "category:aggregate:displacement_flow_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "invalid_field_source" in codes

    def test_unclassified_vault_source_id_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["causal_narrative"]["citations"] = ["vault:weakness:unknown"]
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "hallucinated_source_id" in codes or "unclassified_source_id" in codes

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

    def test_unknown_witness_citation_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["account_reasoning"]["citations"] = ["witness:missing:0"]
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "unknown_packet_citation" in codes

    def test_unknown_aggregate_citation_fails(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["category_reasoning"]["citations"] = [
            "displacement:aggregate:not_real",
        ]
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "unknown_packet_citation" in codes

    def test_thin_specific_witness_pool_warns(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        packet.section_packets["_witness_governance"] = {
            "thin_specific_witness_pool": True,
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid
        warning_codes = [w.code for w in result.warnings]
        assert "thin_specific_witness_pool" in warning_codes

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

    def test_priority_segment_requires_reach_of_at_least_five(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 3,
            "source_id": "accounts:summary:high_intent_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "segment_sample_too_small" in codes

    def test_percentage_segment_claim_requires_supported_reach(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["why_vulnerable"] = "100% churn rate and highest urgency"
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 4,
            "source_id": "accounts:summary:high_intent_count",
        }
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "unsupported_segment_percentage" in codes

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

    def test_build_reasoning_contracts_normalizes_timeline_trigger_type(self):
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["timing_intelligence"]["immediate_triggers"] = [
            {"trigger": "Q2 renewal", "type": "timeline", "trigger_type": "timeline_signal", "urgency": "high"},
            {"trigger": "renewal date", "type": "contract_end", "urgency": "high"},
            {"trigger": "ticket sort limitation", "type": "turning_point", "urgency": "high"},
        ]

        contracts = build_reasoning_contracts(synthesis, packet)
        triggers = contracts["vendor_core_reasoning"]["timing_intelligence"]["immediate_triggers"]

        assert triggers[0]["type"] == "signal"
        assert triggers[0]["trigger_type"] == "signal"
        assert triggers[1]["type"] == "deadline"
        assert triggers[2]["type"] == "signal"

    def test_build_reasoning_contracts_normalizes_segment_and_timing_text(self):
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )
        synthesis, packet = _make_valid_synthesis()
        synthesis["segment_playbook"]["priority_segments"][0]["segment"] = "end_user role"
        synthesis["segment_playbook"]["priority_segments"][0]["best_opening_angle"] = (
            "Highlight lower total cost"
        )
        synthesis["timing_intelligence"]["best_timing_window"] = (
            "immediate - active evaluation signals are present across multiple flows "
            "(timeline_signal: immediate)"
        )

        contracts = build_reasoning_contracts(synthesis, packet)
        segment = contracts["vendor_core_reasoning"]["segment_playbook"]["priority_segments"][0]
        timing = contracts["vendor_core_reasoning"]["timing_intelligence"]

        assert segment["segment"] == "end users"
        assert segment["best_opening_angle"] == "highlight lower total cost"
        assert timing["best_timing_window"].startswith(
            "Immediate - buyers are actively evaluating alternatives across multiple flows"
        )
        assert "timeline_signal" not in timing["best_timing_window"]

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
    def test_section_falls_back_to_reasoning_contracts(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "schema_version": "2.1",
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "confidence": "high",
                        "trigger": "Price hike",
                    },
                    "segment_playbook": {"confidence": "medium"},
                    "timing_intelligence": {"confidence": "medium"},
                },
                "displacement_reasoning": {
                    "schema_version": "v1",
                    "competitive_reframes": {"confidence": "medium"},
                    "migration_proof": {"confidence": "high"},
                },
            },
            "meta": {"evidence_window_start": "2025-12-01"},
        }
        view = load_synthesis_view(raw, "TestVendor")
        assert view.section("causal_narrative")["trigger"] == "Price hike"
        assert view.section("migration_proof")["confidence"] == "high"
        assert view.primary_wedge is not None

    def test_section_prefers_reasoning_contracts_over_flat_mirrors(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "schema_version": "2.2",
            "causal_narrative": {
                "primary_wedge": "support_erosion",
                "confidence": "high",
                "trigger": "Old flat trigger",
            },
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "confidence": "high",
                        "trigger": "Contract trigger",
                    },
                },
            },
        }
        view = load_synthesis_view(raw, "TestVendor")
        assert view.section("causal_narrative")["trigger"] == "Contract trigger"
        assert view.section("causal_narrative")["primary_wedge"] == "price_squeeze"

    def test_section_does_not_backfill_missing_contract_section_from_flat_mirror(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "schema_version": "2.2",
            "timing_intelligence": {
                "confidence": "medium",
                "best_timing_window": "Legacy mirror",
            },
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "confidence": "high",
                    },
                    "segment_playbook": {"confidence": "medium"},
                },
            },
        }
        view = load_synthesis_view(raw, "TestVendor")
        assert view.section("timing_intelligence") == {}

    def test_contract_accessor_returns_blocks(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )
        raw = {
            "schema_version": "2.1",
            "reasoning_contracts": {
                "category_reasoning": {
                    "schema_version": "v1",
                    "market_regime": "fragmented",
                },
            },
        }
        view = load_synthesis_view(raw, "TestVendor")
        assert view.contract("category_reasoning")["market_regime"] == "fragmented"
        assert view.contract("missing") == {}

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
            "schema_version": "2.1",
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
            "schema_version": "2.1",
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
            "schema_version": "2.1",
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

        # Flat sections are no longer persisted by default
        assert "causal_narrative" not in card
        assert "segment_playbook" not in card
        # Wedge info injected
        assert card["synthesis_wedge"] == "price_squeeze"
        assert card["synthesis_wedge_label"] == "Price Squeeze"
        assert card["synthesis_schema_version"] == "v2"
        assert card["reasoning_source"] == "b2b_reasoning_synthesis"
        assert card["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
        assert card["reasoning_contracts"]["displacement_reasoning"]["migration_proof"]["switching_is_real"] is True
        assert "vendor_core_reasoning" not in card
        assert "displacement_reasoning" not in card

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
        assert "causal_narrative" not in card
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


class TestReasoningContracts:
    def test_normalize_reasoning_payload_rewrites_legacy_artifact(self):
        from atlas_brain.services.b2b_reasoning_backfill import (
            normalize_reasoning_payload,
        )

        payload = {
            "vendor": "Zendesk",
            "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high"},
            "migration_proof": {"confidence": "high", "switching_is_real": True},
            "vendor_core_reasoning": {"causal_narrative": {"primary_wedge": "support_erosion"}},
        }

        normalized = normalize_reasoning_payload(payload)

        assert "reasoning_contracts" in normalized
        assert normalized["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] == "price_squeeze"
        assert "causal_narrative" not in normalized
        assert "migration_proof" not in normalized
        assert "vendor_core_reasoning" not in normalized

    def test_normalize_reasoning_payload_marks_synthesis_shape(self):
        from atlas_brain.services.b2b_reasoning_backfill import (
            normalize_reasoning_payload,
        )

        payload = {
            "vendor": "Zendesk",
            "schema_version": "2.2",
            "meta": {"evidence_window_start": "2026-03-01", "evidence_window_end": "2026-03-18"},
            "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high"},
            "migration_proof": {"confidence": "high", "switching_is_real": True},
        }

        normalized = normalize_reasoning_payload(payload, synthesis_mode=True)

        assert normalized["reasoning_shape"] == "contracts_first_v1"
        assert "reasoning_contracts" in normalized
        assert normalized["synthesis_wedge"] == "price_squeeze"

    def test_build_reasoning_contracts_decomposes_sections(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["schema_version"] == "v1"
        assert contracts["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] == "price_squeeze"
        assert contracts["displacement_reasoning"]["migration_proof"]["evidence_type"] == "explicit_switch"
        assert contracts["category_reasoning"]["market_regime"] == "fragmented"

    def test_build_reasoning_contracts_prefers_stronger_segment_eval_signal(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["timing_intelligence"]["active_eval_signals"] = {
            "value": 9,
            "source_id": "accounts:summary:active_eval_signal_count",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["vendor_core_reasoning"]["timing_intelligence"]["active_eval_signals"] == {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }

    def test_build_reasoning_contracts_attaches_segment_supporting_evidence(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)

        contracts = build_reasoning_contracts(synthesis, packet)

        supporting = (
            contracts["vendor_core_reasoning"]["segment_playbook"]["supporting_evidence"]
        )
        assert supporting["top_departments"][0]["department"] == "engineering"
        assert supporting["top_roles"][0]["role_type"] == "admin"
        assert supporting["top_contract_segments"][0]["segment"] == "enterprise"
        assert supporting["top_usage_durations"][0]["duration"] == "1_to_3_years"
        assert supporting["top_use_cases"][0]["use_case"] == "CRM"
        assert supporting["top_company_sizes"][0]["segment"] == "Mid-Market"
        assert supporting["company_size_context"]["avg_seat_count"]["source_id"] == "segment:size:avg_seat_count"
        assert supporting["budget_context"]["price_increase_count"]["value"] == 9
        assert supporting["active_eval_signals"]["source_id"] == "segment:aggregate:active_eval_signal_count"

    def test_build_reasoning_contracts_rebuilds_stale_segment_supporting_evidence(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["segment_playbook"]["supporting_evidence"] = {
            "top_contract_segments": [{"segment": "smb", "source_id": "segment:contract:smb"}],
            "top_usage_durations": [{"duration": "3_years", "source_id": "segment:duration:3_years"}],
            "top_use_cases": [{"use_case": "Bogus", "source_id": "segment:use_case:bogus"}],
            "custom_note": "keep",
        }

        contracts = build_reasoning_contracts(synthesis, packet)
        supporting = (
            contracts["vendor_core_reasoning"]["segment_playbook"]["supporting_evidence"]
        )

        assert supporting["top_contract_segments"][0]["source_id"] == "segment:contract:enterprise"
        assert supporting["top_usage_durations"][0]["source_id"] == "segment:duration:1_to_3_years"
        assert supporting["top_use_cases"][0]["source_id"] == "segment:use_case:crm"
        assert supporting["custom_note"] == "keep"

    def test_build_reasoning_contracts_attaches_timing_supporting_evidence(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)

        contracts = build_reasoning_contracts(synthesis, packet)
        timing = contracts["vendor_core_reasoning"]["timing_intelligence"]
        supporting = timing["supporting_evidence"]

        assert timing["sentiment_direction"] == "declining"
        assert supporting["timeline_signal_summary"]["evaluation_deadline_signals"]["source_id"] == (
            "temporal:signal:evaluation_deadline_signals"
        )
        assert supporting["sentiment_snapshot"]["declining_count"]["value"] == 6
        assert supporting["top_keyword_spikes"][0]["source_id"] == "temporal:spike:alternative"
        assert supporting["top_turning_points"][0]["source_id"] == (
            "temporal:turning_point:pricing_change"
        )
        assert supporting["top_sentiment_tenure"][0]["source_id"] == (
            "temporal:tenure:1_to_3_years"
        )
        assert any(
            item.get("source_id") == "temporal:deadline:q2_renewal"
            and item.get("type") == "deadline"
            for item in supporting["top_timing_signals"]
        )

    def test_build_reasoning_contracts_rebuilds_stale_timing_supporting_evidence(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["timing_intelligence"]["supporting_evidence"] = {
            "top_keyword_spikes": [
                {"keyword": "bogus", "source_id": "temporal:spike:bogus"},
            ],
            "custom_note": "keep",
        }
        synthesis["timing_intelligence"]["sentiment_direction"] = "improving"

        contracts = build_reasoning_contracts(synthesis, packet)
        timing = contracts["vendor_core_reasoning"]["timing_intelligence"]
        supporting = timing["supporting_evidence"]

        assert timing["sentiment_direction"] == "declining"
        assert supporting["top_keyword_spikes"][0]["source_id"] == "temporal:spike:alternative"
        assert supporting["custom_note"] == "keep"

    def test_build_reasoning_contracts_backfills_top_timing_signals_from_triggers(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        packet.pools["temporal"] = []
        synthesis, _ = _make_valid_synthesis(packet)

        contracts = build_reasoning_contracts(synthesis, packet)
        supporting = contracts["vendor_core_reasoning"]["timing_intelligence"]["supporting_evidence"]

        assert supporting["top_timing_signals"][0]["source_id"] == (
            "temporal:deadline:q2_renewal"
        )
        assert supporting["top_timing_signals"][0]["type"] == "deadline"

    def test_build_reasoning_contracts_filters_invalid_causal_citations(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["causal_narrative"]["citations"] = [
            "vault:weakness:pricing",
            "category:regime:disruption",
        ]

        contracts = build_reasoning_contracts(synthesis, packet)
        citations = contracts["vendor_core_reasoning"]["causal_narrative"]["citations"]

        assert citations == ["vault:weakness:pricing"]

    def test_build_reasoning_contracts_attaches_top_strategic_roles(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["segment"]["affected_roles"] = [
            {"role_type": "end_user", "churn_rate": 0.0, "review_count": 40},
            {"role_type": "economic_buyer", "churn_rate": 0.0, "review_count": 15},
            {"role_type": "evaluator", "churn_rate": 0.0, "review_count": 10},
            {"role_type": "champion", "churn_rate": 0.0, "review_count": 4},
        ]
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)

        contracts = build_reasoning_contracts(synthesis, packet)

        strategic = (
            contracts["vendor_core_reasoning"]["segment_playbook"]["supporting_evidence"]["top_strategic_roles"]
        )
        assert [item["role_type"] for item in strategic] == [
            "economic_buyer",
            "evaluator",
        ]
        assert strategic[0]["source_id"] == "segment:role:economic_buyer"

    def test_account_reasoning_normalizes_urgency_scale_to_intent_score(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["accounts"]["accounts"] = [
            {
                "company_name": "Acme Corp",
                "urgency_score": 9.0,
                "decision_maker": True,
                "buying_stage": "evaluation",
            },
            {
                "company_name": "Globex",
                "urgency_score": 5.0,
                "buying_stage": "monitoring",
            },
        ]

        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)

        contracts = build_reasoning_contracts(synthesis, packet)

        top_accounts = contracts["account_reasoning"]["top_accounts"]
        assert top_accounts[0]["name"] == "Acme Corp"
        assert top_accounts[0]["intent_score"] == 0.9
        assert top_accounts[1]["intent_score"] == 0.5

    def test_build_reasoning_contracts_prefers_deterministic_account_top_accounts(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["accounts"]["accounts"] = [
            {
                "company_name": "Acme Corp",
                "urgency_score": 9.0,
                "decision_maker": True,
                "buying_stage": "evaluation",
            },
        ]
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["account_reasoning"]["top_accounts"] = [
            {
                "name": "Legacy Corp",
                "intent_score": 0.21,
                "source_id": "accounts:company:legacy_corp",
            },
        ]

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["account_reasoning"]["top_accounts"] == [
            {
                "name": "Acme Corp",
                "intent_score": 0.9,
                "source_id": "accounts:company:acme_corp",
            },
        ]

    def test_displacement_compression_merges_duplicate_labels_and_filters_tool_targets(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        layers["displacement"] = [
            {
                "from_vendor": "Zendesk",
                "to_vendor": "Custom ChatGPT integration",
                "flow_summary": {
                    "total_flow_mentions": 8,
                    "explicit_switch_count": 0,
                    "active_evaluation_count": 0,
                },
                "edge_metrics": {"mention_count": 8, "primary_driver": "features"},
            },
            {
                "from_vendor": "Zendesk",
                "to_vendor": "Custom ChatGPT Integration",
                "flow_summary": {
                    "total_flow_mentions": 5,
                    "explicit_switch_count": 0,
                    "active_evaluation_count": 0,
                },
                "edge_metrics": {"mention_count": 5, "primary_driver": "features"},
            },
            {
                "from_vendor": "Zendesk",
                "to_vendor": "Freshdesk",
                "flow_summary": {
                    "total_flow_mentions": 20,
                    "explicit_switch_count": 0,
                    "active_evaluation_count": 3,
                },
                "edge_metrics": {"mention_count": 20, "primary_driver": "pricing"},
            },
        ]

        packet = compress_vendor_pools("Zendesk", layers)
        flows = packet.pools.get("displacement", [])
        sids = [item.source_ref.source_id for item in flows]
        agg_map = {agg.source_id: agg.value for agg in packet.aggregates}

        assert sids == ["displacement:flow:freshdesk"]
        assert agg_map["displacement:aggregate:total_active_evaluations"] == 3
        assert agg_map["displacement:aggregate:total_flow_mentions"] == 20

    def test_displacement_compression_prefers_switch_and_eval_signal_over_mentions(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )

        layers = _make_layers()
        layers["displacement"] = [
            {
                "from_vendor": "Zendesk",
                "to_vendor": "CompetitorA",
                "flow_summary": {
                    "total_flow_mentions": 25,
                    "explicit_switch_count": 0,
                    "active_evaluation_count": 0,
                },
                "edge_metrics": {"mention_count": 25, "primary_driver": "pricing"},
            },
            {
                "from_vendor": "Zendesk",
                "to_vendor": "CompetitorB",
                "flow_summary": {
                    "total_flow_mentions": 12,
                    "explicit_switch_count": 2,
                    "active_evaluation_count": 4,
                },
                "edge_metrics": {"mention_count": 12, "primary_driver": "features"},
            },
        ]

        packet = compress_vendor_pools("Zendesk", layers)
        flows = packet.pools.get("displacement", [])

        assert flows[0].data["to_vendor"] == "CompetitorB"
        assert flows[1].data["to_vendor"] == "CompetitorA"

    def test_account_top_accounts_rank_active_eval_ahead_of_monitoring(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["accounts"]["accounts"] = [
            {
                "company_name": "MonitorCo",
                "urgency_score": 8.0,
                "buying_stage": "monitoring",
            },
            {
                "company_name": "EvalCo",
                "urgency_score": 8.0,
                "buying_stage": "evaluation",
            },
        ]

        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        contracts = build_reasoning_contracts(synthesis, packet)

        top_accounts = contracts["account_reasoning"]["top_accounts"]
        assert top_accounts[0]["name"] == "EvalCo"
        assert top_accounts[1]["name"] == "MonitorCo"

    def test_account_reasoning_falls_back_to_vault_company_items_when_accounts_sparse(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["accounts"]["accounts"] = []
        layers["evidence_vault"]["company_signals"] = [
            {
                "company_name": "VaultCo",
                "urgency_score": 9.0,
                "buying_stage": "evaluation",
                "decision_maker": True,
                "review_id": "r-vault-1",
            },
        ]

        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["account_reasoning"]["top_accounts"] == [
            {
                "name": "VaultCo",
                "intent_score": 0.9,
                "source_id": "vault:company:vaultco",
            },
        ]

    def test_build_reasoning_contracts_derives_segment_sample_size_from_reach(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["segment_playbook"]["priority_segments"][0]["estimated_reach"] = {
            "value": 22,
            "source_id": "segment:reach:size:mid_market",
        }
        synthesis["segment_playbook"]["priority_segments"][0]["sample_size"] = None

        contracts = build_reasoning_contracts(synthesis, packet)

        segment = contracts["vendor_core_reasoning"]["segment_playbook"]["priority_segments"][0]
        assert segment["estimated_reach"]["source_id"] == "segment:reach:size:mid_market"
        assert segment["sample_size"] == 22

    def test_build_reasoning_contracts_upgrades_migration_eval_volume(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["active_evaluation_volume"] = {
            "value": 11,
            "source_id": "displacement:aggregate:total_active_evaluations",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["displacement_reasoning"]["migration_proof"]["active_evaluation_volume"] == {
            "value": 133,
            "source_id": "segment:aggregate:active_eval_signal_count",
        }

    def test_build_persistable_synthesis_downgrades_stale_explicit_switch_claims(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        layers = _make_layers()
        for flow in layers["displacement"]:
            flow["flow_summary"]["explicit_switch_count"] = 0
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["switching_is_real"] = True
        synthesis["migration_proof"]["evidence_type"] = "explicit_switch"
        synthesis["migration_proof"]["switch_volume"] = {
            "value": 5,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }

        persisted = build_persistable_synthesis(synthesis, packet)
        migration = persisted["reasoning_contracts"]["displacement_reasoning"]["migration_proof"]

        assert migration["switch_volume"] == {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        assert migration["switching_is_real"] is False
        assert migration["evidence_type"] == "active_evaluation"
        assert migration["confidence"] == "medium"

        result = validate_synthesis(persisted, packet)
        assert result.is_valid, result.summary()

    def test_build_persistable_synthesis_backfills_contradiction_hedging(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        packet.contradiction_rows = [
            {"dimension": "support"},
            {"dimension": "ux"},
        ]
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["causal_narrative"]["confidence"] = "high"
        synthesis["causal_narrative"]["data_gaps"] = []

        persisted = build_persistable_synthesis(synthesis, packet)
        vendor_core = persisted["reasoning_contracts"]["vendor_core_reasoning"]
        causal = vendor_core["causal_narrative"]
        limits = vendor_core["confidence_posture"]["limits"]

        assert causal["confidence"] == "medium"
        assert any(
            "support" in gap.lower() and "ux" in gap.lower()
            for gap in causal["data_gaps"]
        )
        assert any(
            "contradict" in limit.lower() and "support" in limit.lower()
            for limit in limits
        )

        result = validate_synthesis(persisted, packet, governance_blocking=True)
        assert result.is_valid, result.summary()

    def test_build_reasoning_contracts_uses_broader_displacement_mentions(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["evidence_vault"]["metric_snapshot"]["displacement_mention_count"] = 0
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": 0,
            "source_id": "vault:metric:displacement_mention_count",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["displacement_reasoning"]["migration_proof"]["displacement_mention_volume"] == {
            "value": 5,
            "source_id": "category:aggregate:displacement_flow_count",
        }

    def test_build_reasoning_contracts_coerces_string_numeric_wrapper_values(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["evidence_vault"]["metric_snapshot"]["displacement_mention_count"] = 0
        layers["category"]["displacement_flow_count"] = 0
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": "0",
            "source_id": "vault:metric:displacement_mention_count",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["displacement_reasoning"]["migration_proof"]["displacement_mention_volume"] == {
            "value": 0,
            "source_id": "vault:metric:displacement_mention_count",
        }

    def test_normalize_total_flow_mentions_alias_to_allowed_displacement_volume_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": 5,
            "source_id": "displacement:aggregate:total_flow_mentions",
        }

        normalize_synthesis_source_ids(synthesis, packet)

        assert synthesis["migration_proof"]["displacement_mention_volume"] == {
            "value": 5,
            "source_id": "category:aggregate:displacement_flow_count",
        }

        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_displacement_flow_source_allowed_for_displacement_mention_volume(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": 12,
            "source_id": "displacement:flow:competitora",
        }

        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_total_flow_mentions_allowed_for_displacement_mention_volume(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["migration_proof"]["displacement_mention_volume"] = {
            "value": 5,
            "source_id": "displacement:aggregate:total_flow_mentions",
        }

        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_build_reasoning_contracts_nulls_invalid_destination_fields(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        for flow in layers["displacement"]:
            flow["flow_summary"]["explicit_switch_count"] = 0
            flow["flow_summary"]["active_evaluation_count"] = 0
            flow["flow_summary"]["mention_count"] = 0
            flow["flow_summary"]["total_flow_mentions"] = 0
            flow["edge_metrics"] = {"mention_count": 0, "primary_driver": None}
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["top_destination"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_active_evaluations",
        }
        synthesis["migration_proof"]["primary_switch_driver"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["displacement_reasoning"]["migration_proof"]["top_destination"] == {
            "value": None,
            "source_id": None,
        }
        assert contracts["displacement_reasoning"]["migration_proof"]["primary_switch_driver"] == {
            "value": None,
            "source_id": None,
        }

    def test_build_reasoning_contracts_nulls_na_destination_fields(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        for flow in layers["displacement"]:
            flow["flow_summary"]["explicit_switch_count"] = 0
            flow["flow_summary"]["active_evaluation_count"] = 0
            flow["flow_summary"]["mention_count"] = 0
            flow["flow_summary"]["total_flow_mentions"] = 0
            flow["edge_metrics"] = {"mention_count": 0, "primary_driver": None}
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["top_destination"] = {
            "value": "N/A",
            "source_id": "displacement:aggregate:total_flow_mentions",
        }
        synthesis["migration_proof"]["primary_switch_driver"] = {
            "value": "N/A",
            "source_id": "displacement:aggregate:total_flow_mentions",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["displacement_reasoning"]["migration_proof"]["top_destination"] == {
            "value": None,
            "source_id": None,
        }
        assert contracts["displacement_reasoning"]["migration_proof"]["primary_switch_driver"] == {
            "value": None,
            "source_id": None,
        }

    def test_build_reasoning_contracts_derives_destination_and_driver_from_best_flow(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        layers = _make_layers()
        layers["displacement"][0]["edge_metrics"] = {
            "mention_count": 12,
            "primary_driver": "pricing",
        }
        layers["displacement"][0]["flow_summary"]["total_flow_mentions"] = 12
        layers["displacement"][1]["edge_metrics"] = {
            "mention_count": 5,
            "primary_driver": "support",
        }
        layers["displacement"][1]["flow_summary"]["total_flow_mentions"] = 5
        packet = compress_vendor_pools("ContractVendor", layers)
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["top_destination"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_active_evaluations",
        }
        synthesis["migration_proof"]["primary_switch_driver"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["displacement_reasoning"]["migration_proof"]["top_destination"] == {
            "value": "CompetitorA",
            "source_id": "displacement:flow:competitora",
        }
        assert contracts["displacement_reasoning"]["migration_proof"]["primary_switch_driver"] == {
            "value": "pricing",
            "source_id": "displacement:flow:competitora",
        }

    def test_build_reasoning_contracts_keeps_displacement_counts_distinct(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        contracts = build_reasoning_contracts(synthesis, packet)

        displacement = contracts["displacement_reasoning"]
        assert displacement["confirmed_switch_count"]["source_id"] == "displacement:aggregate:total_explicit_switches"
        assert displacement["active_evaluation_count"]["source_id"] == "displacement:aggregate:total_active_evaluations"
        assert displacement["displacement_mention_volume"]["source_id"] == "vault:metric:displacement_mention_count"
        assert displacement["top_flows"][0]["to_vendor"] == "CompetitorA"

    def test_build_reasoning_contracts_strips_placeholder_named_examples(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "Unknown SMB customer",
                "evidence": "Quoted expensive seats",
                "source_type": "review_site",
                "quotable": True,
                "source_id": "displacement:flow:competitora",
            },
        ]

        contracts = build_reasoning_contracts(synthesis, packet)

        migration = contracts["displacement_reasoning"]["migration_proof"]
        assert migration["named_examples"] == []
        assert "No credible named migration examples" in migration["data_gaps"]

    def test_build_reasoning_contracts_strips_generic_reviewer_named_examples(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "G2 reviewer",
                "evidence": "Quoted expensive seats",
                "source_type": "review_site",
                "quotable": True,
                "source_id": "vault:weakness:pricing",
            },
        ]

        contracts = build_reasoning_contracts(synthesis, packet)
        migration = contracts["displacement_reasoning"]["migration_proof"]
        assert migration["named_examples"] == []
        assert "No credible named migration examples" in migration["data_gaps"]

    def test_build_reasoning_contracts_strips_segment_style_named_examples(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "AI-averse organization",
                "evidence": "Avoiding AI pricing bundles",
                "source_type": "review_site",
                "quotable": True,
                "source_id": "vault:weakness:pricing",
            },
        ]

        contracts = build_reasoning_contracts(synthesis, packet)
        migration = contracts["displacement_reasoning"]["migration_proof"]
        assert migration["named_examples"] == []
        assert "No credible named migration examples" in migration["data_gaps"]

    def test_build_reasoning_contracts_strips_tool_style_named_examples(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["migration_proof"]["named_examples"] = [
            {
                "company": "Custom ChatGPT Integration",
                "evidence": "Seeking more control over AI features",
                "source_type": "inferred",
                "quotable": False,
                "source_id": "displacement:flow:custom_chatgpt_integration",
            },
        ]

        contracts = build_reasoning_contracts(synthesis, packet)
        migration = contracts["displacement_reasoning"]["migration_proof"]
        assert migration["named_examples"] == []
        assert "No credible named migration examples" in migration["data_gaps"]

    def test_build_reasoning_contracts_does_not_backfill_missing_explicit_sections(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_reasoning_contracts,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis = {
            "schema_version": "2.2",
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {"primary_wedge": "price_squeeze"},
                },
            },
            "timing_intelligence": {"best_timing_window": "Legacy fallback"},
            "migration_proof": {"confidence": "high"},
        }

        contracts = build_reasoning_contracts(synthesis, packet)

        assert contracts["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] == "price_squeeze"
        assert "timing_intelligence" not in contracts["vendor_core_reasoning"]
        assert "displacement_reasoning" not in contracts
        assert contracts["category_reasoning"]["market_regime"] == "fragmented"

    def test_materialized_contracts_fall_back_to_flat_sections(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        raw = {
            "schema_version": "2.1",
            "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high"},
            "segment_playbook": {"confidence": "medium"},
            "timing_intelligence": {"confidence": "medium"},
            "competitive_reframes": {"confidence": "medium"},
            "migration_proof": {"confidence": "high"},
        }

        view = load_synthesis_view(raw, "FallbackVendor")
        contracts = view.materialized_contracts()

        assert contracts["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] == "price_squeeze"
        assert contracts["displacement_reasoning"]["migration_proof"]["confidence"] == "high"
        assert "category_reasoning" not in contracts

    def test_materialized_contracts_preserve_explicit_category_contract(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        raw = {
            "schema_version": "2.1",
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {"schema_version": "v1"},
                "displacement_reasoning": {"schema_version": "v1"},
                "category_reasoning": {"schema_version": "v1", "market_regime": "fragmented"},
            },
        }

        view = load_synthesis_view(raw, "CategoryVendor")
        contracts = view.materialized_contracts()

        assert contracts["category_reasoning"]["market_regime"] == "fragmented"

    def test_materialized_contracts_do_not_backfill_missing_sections_when_contracts_exist(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        raw = {
            "schema_version": "2.2",
            "migration_proof": {"confidence": "high"},
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high"},
                },
            },
        }

        view = load_synthesis_view(raw, "ContractOnlyVendor")
        contracts = view.materialized_contracts()

        assert "vendor_core_reasoning" in contracts
        assert "displacement_reasoning" not in contracts

    def test_materialized_contracts_use_raw_category_reasoning_when_contract_missing(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        raw = {
            "schema_version": "2.2",
            "category_reasoning": {"schema_version": "v1", "market_regime": "fragmented"},
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {"schema_version": "v1"},
            },
        }

        view = load_synthesis_view(raw, "CompatVendor")
        contracts = view.materialized_contracts()

        assert contracts["category_reasoning"]["market_regime"] == "fragmented"

    def test_materialized_contracts_repair_stale_timing_support(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        raw = {
            "schema_version": "2.2",
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "timing_intelligence": {
                        "confidence": "medium",
                        "immediate_triggers": [
                            {
                                "trigger": "Contract renewals",
                                "type": "deadline",
                                "source": {
                                    "source_id": "temporal:deadline:q2_renewal",
                                    "source_type": "temporal",
                                },
                            },
                        ],
                        "supporting_evidence": {
                            "timeline_signal_summary": {
                                "contract_end_signals": {
                                    "value": 1,
                                    "source_id": "temporal:signal:contract_end_signals",
                                },
                            },
                        },
                    },
                },
            },
        }

        view = load_synthesis_view(raw, "TimingRepairVendor")
        timing = view.materialized_contracts()["vendor_core_reasoning"]["timing_intelligence"]

        assert timing["supporting_evidence"]["top_timing_signals"][0]["source_id"] == (
            "temporal:deadline:q2_renewal"
        )
        assert any(
            item.get("source_id") == "temporal:signal:contract_end_signals"
            and item.get("type") == "deadline"
            for item in timing["supporting_evidence"]["top_timing_signals"]
        )

    def test_materialized_contracts_repair_stale_segment_and_timing_labels(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        raw = {
            "schema_version": "2.2",
            "reasoning_contracts": {
                "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "segment_playbook": {
                        "priority_segments": [
                            {
                                "segment": "role:end_user",
                                "best_opening_angle": "Offer lower total cost",
                            },
                        ],
                    },
                    "timing_intelligence": {
                        "best_timing_window": (
                            "immediate - active evaluation signals are present across "
                            "multiple flows (timeline_signal: immediate)"
                        ),
                    },
                },
            },
        }

        view = load_synthesis_view(raw, "RepairVendor")
        vendor_core = view.materialized_contracts()["vendor_core_reasoning"]

        assert vendor_core["segment_playbook"]["priority_segments"][0]["segment"] == "end users"
        assert (
            vendor_core["segment_playbook"]["priority_segments"][0]["best_opening_angle"]
            == "offer lower total cost"
        )
        assert vendor_core["timing_intelligence"]["best_timing_window"].startswith(
            "Immediate - buyers are actively evaluating alternatives across multiple flows"
        )


    def test_build_persistable_synthesis_is_contracts_first(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        persisted = build_persistable_synthesis(synthesis, packet)

        assert persisted["schema_version"] == "2.1"
        assert persisted["reasoning_shape"] == "contracts_first_v1"
        assert persisted["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] == "price_squeeze"
        assert persisted["synthesis_wedge"] == "price_squeeze"
        assert persisted["synthesis_wedge_label"] == "Price Squeeze"
        assert "meta" in persisted
        assert "causal_narrative" not in persisted
        assert "segment_playbook" not in persisted
        assert "timing_intelligence" not in persisted
        assert "competitive_reframes" not in persisted
        assert "migration_proof" not in persisted

    def test_build_persistable_synthesis_tracks_metric_and_witness_references(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        persisted = build_persistable_synthesis(synthesis, packet)

        assert persisted["reference_ids"]["metric_ids"]
        assert persisted["reference_ids"]["witness_ids"]
        assert persisted["packet_artifacts"]["witness_pack"]
        assert persisted["packet_artifacts"]["section_packets"]["causal_packet"]["witness_ids"]

    def test_load_synthesis_view_reads_contracts_first_persisted_shape(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_reasoning_contracts import (
            build_persistable_synthesis,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            load_synthesis_view,
        )

        packet = compress_vendor_pools("ContractVendor", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        persisted = build_persistable_synthesis(synthesis, packet)

        view = load_synthesis_view(persisted, "ContractVendor")

        assert view.section("causal_narrative")["primary_wedge"] == "price_squeeze"
        assert view.section("migration_proof")["evidence_type"] == "explicit_switch"
        assert view.primary_wedge is not None


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

    def test_proof_point_requires_numeric_support_source(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        packet = compress_vendor_pools("Edge", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"] = (
            "vault:weakness:pricing"
        )
        result = validate_synthesis(synthesis, packet)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "proof_point_requires_numeric_support_source" in codes

    def test_proof_point_allows_displacement_flow_shortlist_source(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            validate_synthesis,
        )

        packet = compress_vendor_pools("Edge", _make_layers())
        synthesis, _ = _make_valid_synthesis(packet)
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"] = {
            "field": "displacement_flow_mentions",
            "value": 12,
            "source_id": "displacement:flow:competitora",
            "interpretation": "Buyers are actively naming CompetitorA.",
        }
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_witness_backed_proof_point_to_numeric_support_source(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"] = (
            "witness:ee295534-c0cb-423e-af89-173a4e090374:0"
        )
        normalize_synthesis_source_ids(synthesis, packet)
        assert (
            synthesis["competitive_reframes"]["reframes"][0]["proof_point"]["source_id"]
            == "vault:weakness:pricing:mention_count_total"
        )
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

    def test_normalize_drops_reframe_when_numeric_support_is_unavailable(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            normalize_synthesis_source_ids,
            validate_synthesis,
        )

        synthesis, packet = _make_valid_synthesis()
        synthesis["competitive_reframes"]["reframes"][0]["proof_point"] = {
            "field": "unsupported",
            "value": 999,
            "source_id": "witness:ee295534-c0cb-423e-af89-173a4e090374:0",
            "interpretation": "unsupported",
        }
        synthesis["competitive_reframes"]["reframes"][0]["citations"] = [
            "witness:ee295534-c0cb-423e-af89-173a4e090374:0",
        ]
        normalize_synthesis_source_ids(synthesis, packet)
        assert synthesis["competitive_reframes"]["reframes"] == []
        assert (
            "Numeric support too thin for competitive reframes."
            in synthesis["competitive_reframes"]["data_gaps"]
        )
        result = validate_synthesis(synthesis, packet)
        assert result.is_valid, result.summary()

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
        assert "causal_narrative" not in card

    def test_inject_synthesis_into_card_can_materialize_flat_sections(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
            inject_synthesis_into_card,
            load_synthesis_view,
        )
        raw = {
            "schema_version": "2.1",
            "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "high", "trigger": "Price hike"},
            "segment_playbook": {"confidence": "insufficient", "data_gaps": ["No segment data"]},
            "timing_intelligence": {"confidence": "medium", "best_timing_window": "Q2"},
            "competitive_reframes": {"confidence": "medium", "reframes": []},
            "migration_proof": {"confidence": "high", "switching_is_real": True},
        }
        view = load_synthesis_view(raw, "TestVendor")
        card: dict = {}
        inject_synthesis_into_card(card, view, materialize_flat_sections=True)
        assert card["causal_narrative"]["trigger"] == "Price hike"
        assert card["timing_intelligence"]["best_timing_window"] == "Q2"
        assert "segment_playbook" not in card

    def test_missing_confidence_defaults_to_insufficient(self):
        """Backward-compatible minimal sections stay usable without confidence."""
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
        assert view.confidence("causal_narrative") == "medium"
        assert not view.should_suppress("causal_narrative")
        assert view.is_quotable("causal_narrative")

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

    def test_switching_real_zero_volume_errors(self):
        """switching_is_real=true with zero confirmed switches is invalid."""
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
        codes = [e.code for e in result.errors]
        assert "switching_requires_confirmed_switches" in codes
        assert "explicit_switch_without_volume" in codes

    def test_switching_real_zero_volume_active_eval_still_errors(self):
        """switching_is_real=true is still invalid without confirmed switches."""
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
        codes = [e.code for e in result.errors]
        assert "switching_requires_confirmed_switches" in codes

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
        assert card["evidence_window_label"] == "5-day evidence window"
        assert card["evidence_window_is_thin"] is True
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
        assert card["evidence_window_label"] == "33-day evidence window"
        assert card["evidence_window_is_thin"] is False
        assert "evidence_depth_warning" not in card


class TestReasoningSynthesisTask:
    @pytest.fixture(autouse=True)
    def _enable_reasoning_synthesis_for_task_tests(self, monkeypatch):
        from atlas_brain.config import settings

        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False,
        )

    @pytest.mark.asyncio
    async def test_persist_packet_artifacts_reuses_unchanged_witness_rows(self):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _persist_packet_artifacts,
        )

        class FakePool:
            def __init__(self, existing_rows):
                self._existing_rows = existing_rows
                self.executed = []

            async def fetch(self, query, *args):
                if "FROM b2b_vendor_witnesses" in query:
                    return self._existing_rows
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

        packet = compress_vendor_pools("PersistVendor", _make_layers())
        first = packet.witness_pack[0]
        fake_pool = FakePool([
            {
                "witness_id": first["witness_id"],
                "witness_hash": first["witness_hash"],
            },
            {
                "witness_id": "witness:stale:0",
                "witness_hash": "stalehash",
            },
        ])

        await _persist_packet_artifacts(
            fake_pool,
            vendor_name="PersistVendor",
            as_of_date=date(2026, 3, 30),
            analysis_window_days=90,
            evidence_hash="abc123",
            packet=packet,
        )

        delete_ops = [
            item for item in fake_pool.executed
            if "DELETE FROM b2b_vendor_witnesses" in item[0]
        ]
        upsert_ops = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_vendor_witnesses" in item[0]
        ]
        assert len(delete_ops) == 1
        assert delete_ops[0][1][4] == ["witness:stale:0"]
        assert len(upsert_ops) == len(packet.witness_pack) - 1

    def test_witness_row_payload_coerces_reviewed_at_string(self):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _witness_row_payload,
        )

        payload = _witness_row_payload({
            "witness_id": "witness:r1:0",
            "reviewed_at": "2026-03-20T00:00:00+00:00",
        })

        assert payload["reviewed_at"] is not None
        assert payload["reviewed_at"].year == 2026

    @pytest.mark.asyncio
    async def test_run_retries_validation_failure_and_persists_success(self, monkeypatch):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run
        from atlas_brain.config import settings

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, responses):
                self._responses = list(responses)
                self.calls = []

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                idx = len(self.calls)
                self.calls.append({
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                })
                return {
                    "response": self._responses[idx],
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("RetryVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        invalid_synthesis = deepcopy(valid_synthesis)
        invalid_synthesis["causal_narrative"]["primary_wedge"] = "totally_made_up"

        fake_pool = FakePool()
        fake_llm = FakeLLM([
            json.dumps(invalid_synthesis),
            json.dumps(valid_synthesis),
        ])

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"RetryVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.storage.repositories.competitive_set.CompetitiveSetRepository.list_due_scheduled",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._scheduled_scope_strategy",
            lambda _task: "full_universe",
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 2, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_retry_delay_seconds", 0.0, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_feedback_limit", 5, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_tokens", 2048, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_temperature", 0.0, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True, "test_vendors": ["BatchVendor"]}))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_failed"] == 0
        assert result["vendors_validation_failures"] == 0
        assert len(fake_llm.calls) == 2
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        assert len(synthesis_writes) == 1
        first_call = fake_llm.calls[0]
        retry_messages = fake_llm.calls[1]["messages"]
        first_payload = json.loads(first_call["messages"][1].content)
        assert first_call["max_tokens"] == 2048
        assert first_call["temperature"] == 0.0
        assert "metric_ledger" not in first_payload
        assert "precomputed_aggregates" not in first_payload
        assert first_payload["section_packets"]["displacement_packet"]["numeric_support"]["switch_volume"]["source_id"]
        assert any(
            msg.role == "user" and "previous response was rejected" in msg.content
            for msg in retry_messages
        )
        rejected_attempts = [
            item for item in fake_pool.executed
            if "INSERT INTO artifact_attempts" in item[0]
            and item[1][6] == "rejected"
        ]
        assert len(rejected_attempts) == 1
        assert rejected_attempts[0][1][4] == 1
        validation_rows = [
            item for item in fake_pool.executemany_calls
            if "INSERT INTO synthesis_validation_results" in item[0]
        ]
        assert len(validation_rows) == 2
        attempt_numbers = [rows[0][5] for _, rows in validation_rows if rows]
        assert attempt_numbers == [1, 2]
        retry_events = [
            item for item in fake_pool.executed
            if "INSERT INTO pipeline_visibility_events" in item[0]
            and item[1][2] == "synthesis"
            and item[1][3] == "validation_retry_rejected"
        ]
        assert len(retry_events) == 1

    @pytest.mark.asyncio
    async def test_run_uses_vendor_batch_when_available(self, monkeypatch):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run
        from atlas_brain.config import settings

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeSynthesisLLM:
            model = "fake-synthesis"

            def __init__(self):
                self.calls = []

            def chat(self, **kwargs):
                self.calls.append(kwargs)
                raise AssertionError(
                    "Direct vendor synthesis path should not be used when batch succeeds"
                )

        class FakeAnthropicLLM:
            model = "fake-anthropic-batch"

        layers = _make_layers()
        packet = compress_vendor_pools("BatchVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)

        fake_pool = FakePool()
        direct_llm = FakeSynthesisLLM()
        batch_llm = FakeAnthropicLLM()
        batch_calls = []
        fallback_marks = []

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"BatchVendor": layers}

        def _fake_get_pipeline_llm(*, workload, **kwargs):
            if workload == "anthropic":
                return batch_llm
            return direct_llm

        async def _fake_run_anthropic_message_batch(**kwargs):
            batch_calls.append(kwargs)
            return SimpleNamespace(
                local_batch_id="batch-vendor-1",
                provider_batch_id="provider-vendor-1",
                results_by_custom_id={
                    "vendor:batchvendor": SimpleNamespace(
                        response_text=json.dumps(valid_synthesis),
                        usage={"input_tokens": 31, "output_tokens": 17},
                        error_text=None,
                    ),
                },
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=0,
                completed_items=1,
                failed_items=0,
            )

        async def _fake_mark_batch_fallback_result(**kwargs):
            fallback_marks.append(kwargs)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.storage.repositories.competitive_set.CompetitiveSetRepository.list_due_scheduled",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._scheduled_scope_strategy",
            lambda _task: "full_universe",
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.services.llm.anthropic.AnthropicLLM",
            FakeAnthropicLLM,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            _fake_get_pipeline_llm,
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
            _fake_run_anthropic_message_batch,
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
            _fake_mark_batch_fallback_result,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "anthropic_batch_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_anthropic_batch_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_anthropic_batch_min_items", 1, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True, "test_vendors": ["BatchVendor"]}))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_failed"] == 0
        assert result["vendor_batch_jobs"] == 1
        assert result["vendor_batch_items_submitted"] == 1
        assert result["vendor_batch_completed_items"] == 1
        assert result["vendor_batch_failed_items"] == 0
        assert len(batch_calls) == 1
        assert direct_llm.calls == []
        assert fallback_marks == []
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        assert len(synthesis_writes) == 1

    @pytest.mark.asyncio
    async def test_run_vendor_batch_fallback_records_usage(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import compress_vendor_pools
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeAnthropicLLM:
            model = "claude-sonnet-4-5"
            name = "anthropic"

            def __init__(self):
                self.calls = []

            def chat(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "response": json.dumps(valid_synthesis),
                    "usage": {"input_tokens": 44, "output_tokens": 21},
                    "_trace_meta": {
                        "billable_input_tokens": 30,
                        "provider_request_id": "req_reason_vendor_123",
                    },
                }

        layers = _make_layers()
        packet = compress_vendor_pools("BatchVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)

        fake_pool = FakePool()
        batch_llm = FakeAnthropicLLM()
        fallback_marks = []

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"BatchVendor": layers}

        def _fake_get_pipeline_llm(*, workload, **kwargs):
            if workload == "anthropic":
                return batch_llm
            return batch_llm

        async def _fake_run_anthropic_message_batch(**kwargs):
            return SimpleNamespace(
                local_batch_id="batch-vendor-fallback",
                provider_batch_id="provider-vendor-fallback",
                results_by_custom_id={
                    "vendor:batchvendor": SimpleNamespace(
                        response_text='{"broken_json"',
                        usage={"input_tokens": 31, "output_tokens": 17},
                        error_text=None,
                    ),
                },
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=1,
                completed_items=0,
                failed_items=1,
            )

        async def _fake_mark_batch_fallback_result(**kwargs):
            fallback_marks.append(kwargs)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.storage.repositories.competitive_set.CompetitiveSetRepository.list_due_scheduled",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._scheduled_scope_strategy",
            lambda _task: "full_universe",
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.services.llm.anthropic.AnthropicLLM",
            FakeAnthropicLLM,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            _fake_get_pipeline_llm,
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
            _fake_run_anthropic_message_batch,
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
            _fake_mark_batch_fallback_result,
        )
        monkeypatch.setattr(settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False)
        monkeypatch.setattr(settings.b2b_churn, "anthropic_batch_enabled", True, raising=False)
        monkeypatch.setattr(settings.b2b_churn, "reasoning_synthesis_anthropic_batch_enabled", True, raising=False)
        monkeypatch.setattr(settings.b2b_churn, "reasoning_synthesis_anthropic_batch_min_items", 1, raising=False)
        monkeypatch.setattr(settings.b2b_churn, "reasoning_synthesis_attempts", 2, raising=False)

        result = await run(SimpleNamespace(metadata={"force": True, "test_vendors": ["BatchVendor"]}))

        assert result["vendors_reasoned"] == 1
        assert batch_llm.calls
        assert fallback_marks
        kwargs = fallback_marks[0]
        assert kwargs["succeeded"] is True
        assert kwargs["usage"]["input_tokens"] == 44
        assert kwargs["usage"]["output_tokens"] == 21
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert kwargs["provider_request_id"] == "req_reason_vendor_123"

    @pytest.mark.asyncio
    async def test_run_retries_with_targeted_guidance_for_contract_gaps(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, responses):
                self._responses = list(responses)
                self.calls = []

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                idx = len(self.calls)
                self.calls.append({"messages": messages})
                return {
                    "response": self._responses[idx],
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("RetryVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        invalid_synthesis = deepcopy(valid_synthesis)
        invalid_synthesis.pop("migration_proof", None)
        invalid_synthesis["category_reasoning"] = {
            "market_regime": "",
            "narrative": "",
            "citations": [],
        }
        invalid_synthesis["competitive_reframes"]["reframes"][0]["citations"] = [
            "witness:not_in_packet",
        ]

        fake_pool = FakePool()
        fake_llm = FakeLLM([
            json.dumps(invalid_synthesis),
            json.dumps(valid_synthesis),
        ])

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"RetryVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 2, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_retry_delay_seconds", 0.0, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 1
        assert len(fake_llm.calls) == 2
        retry_text = "\n".join(
            msg.content
            for msg in fake_llm.calls[1]["messages"]
            if msg.role == "user"
        )
        assert "migration_proof" in retry_text
        assert "category_reasoning.confidence" in retry_text
        assert "already present in the input packet" in retry_text

    @pytest.mark.asyncio
    async def test_record_validation_attempt_escalates_repeated_and_costly_retries(self):
        from atlas_brain.autonomous.tasks._b2b_synthesis_validation import (
            ValidationError,
            ValidationResult,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _record_validation_attempt,
        )

        class FakePool:
            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                self.executemany_calls.append((query, list(args_iterable)))

            async def fetchval(self, query, *args):
                if "detail->'rule_codes' ? $3" in query:
                    return 3
                return 0

            async def fetchrow(self, query, *args):
                return {"retry_count": 2, "retry_tokens": 90000}

        pool = FakePool()
        vresult = ValidationResult(
            vendor_name="RetryVendor",
            errors=[
                ValidationError(
                    path="$.scope",
                    code="scope_ambiguity",
                    message="Differentiate charted data from broader context.",
                    severity="error",
                )
            ],
            warnings=[],
        )

        await _record_validation_attempt(
            pool,
            vendor_name="RetryVendor",
            run_id="run-escalation",
            as_of_date=date(2026, 3, 30),
            analysis_window_days=30,
            attempt_no=1,
            vresult=vresult,
            feedback_limit=5,
            attempt_tokens=45000,
            escalation_window_hours=24,
            repeat_rule_threshold=3,
            cost_min_retries=2,
            cost_tokens_threshold=80000,
            emit_retry_event=True,
        )

        retry_events = [
            item for item in pool.executed
            if "INSERT INTO pipeline_visibility_events" in item[0]
            and item[1][3] == "validation_retry_rejected"
        ]
        assert len(retry_events) == 1
        assert json.loads(retry_events[0][1][13])["tokens_used"] == 45000

        escalations = [
            item for item in pool.executed
            if "INSERT INTO pipeline_visibility_events" in item[0]
            and item[1][3] == "validation_retry_escalated"
        ]
        assert len(escalations) == 2
        assert any(item[1][9] == "repeated_validation_retry" for item in escalations)
        assert any(item[1][9] == "costly_validation_retry" for item in escalations)

    @pytest.mark.asyncio
    async def test_run_persists_post_build_warnings_not_raw_warnings(self, monkeypatch):
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("RetryVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        valid_synthesis["migration_proof"]["citations"] = []
        fake_pool = FakePool()
        fake_llm = FakeLLM(json.dumps(valid_synthesis))

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"RetryVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_failed"] == 0
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        assert len(synthesis_writes) == 1
        persisted = json.loads(synthesis_writes[0][1][5])
        warnings = persisted.get("_validation_warnings") or []
        warning_codes = [w["code"] for w in warnings]
        assert "missing_citations" not in warning_codes
        mp = persisted["reasoning_contracts"]["displacement_reasoning"]["migration_proof"]
        assert mp["citations"]

    @pytest.mark.asyncio
    async def test_run_avoids_retry_for_deterministically_repairable_migration_output(
        self, monkeypatch,
    ):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response
                self.calls = []

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                self.calls.append(
                    {
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                )
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("RetryVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        repairable = deepcopy(valid_synthesis)
        repairable["migration_proof"]["switching_is_real"] = True
        repairable["migration_proof"]["evidence_type"] = "explicit_switch"
        repairable["migration_proof"]["switch_volume"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_explicit_switches",
        }
        repairable["migration_proof"]["active_evaluation_volume"] = {
            "value": 0,
            "source_id": "displacement:aggregate:total_active_evaluations",
        }
        fake_pool = FakePool()
        fake_llm = FakeLLM(json.dumps(repairable))

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"RetryVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 2, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_retry_delay_seconds", 0.0, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_failed"] == 0
        assert len(fake_llm.calls) == 1
        rejected_attempts = [
            item for item in fake_pool.executed
            if "INSERT INTO artifact_attempts" in item[0]
            and item[1][6] == "rejected"
        ]
        assert rejected_attempts == []
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        assert len(synthesis_writes) == 1

    @pytest.mark.asyncio
    async def test_run_returns_failed_vendor_details_for_validation_failure(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                rows = list(args_iterable)
                self.executemany_calls.append((query, rows))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("RetryVendor", layers)
        invalid_synthesis, _ = _make_valid_synthesis(packet)
        invalid_synthesis["causal_narrative"]["primary_wedge"] = "totally_made_up"

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"RetryVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: FakeLLM(json.dumps(invalid_synthesis)),
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_feedback_limit", 3, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 0
        assert result["vendors_failed"] == 1
        assert result["vendors_validation_failures"] == 1
        assert len(result["failed_vendors"]) == 1
        detail = result["failed_vendors"][0]
        assert detail["vendor_name"] == "RetryVendor"
        assert detail["stage"] == "validation"
        assert "vendor=RetryVendor" in detail["summary"]
        assert "errors=1" in detail["summary"]
        assert "FAIL" in detail["summary"]
        assert detail["tokens_used"] == 18
        assert detail["attempts_used"] == 1
        assert len(detail["reasons"]) == 1
        assert "causal_narrative.primary_wedge" in detail["reasons"][0]
        assert "totally_made_up" in detail["reasons"][0]

    @pytest.mark.asyncio
    async def test_run_times_out_reasoning_call_and_reports_failure(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

        class SlowLLM:
            model = "slow-reasoner"

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                time.sleep(1.1)
                return {
                    "response": "{}",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }

        layers = _make_layers()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"TimeoutVendor": layers}

        fake_pool = FakePool()
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: SlowLLM(),
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_retry_delay_seconds", 0.0, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_timeout_seconds", 1.0, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 0
        assert result["vendors_failed"] == 1
        assert result["failed_vendors"][0]["vendor_name"] == "TimeoutVendor"
        assert result["failed_vendors"][0]["stage"] == "llm_exception"
        assert "TimeoutError" in result["failed_vendors"][0]["reasons"][0]

    @pytest.mark.asyncio
    async def test_run_prefers_task_or_global_reasoning_model(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                return None

        class FakeLLM:
            model = "fake-reasoner"

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                return {
                    "response": json.dumps(_make_valid_synthesis()[0]),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }

        seen_models = []
        layers = _make_layers()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"ModelVendor": layers}

        def _fake_get_pipeline_llm(**kw):
            seen_models.append(kw.get("openrouter_model"))
            return FakeLLM()

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            _fake_get_pipeline_llm,
        )
        monkeypatch.setattr(
            settings.llm, "openrouter_reasoning_model", "openai/gpt-oss-120b", raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_model", "", raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )

        await run(SimpleNamespace(metadata={"force": True}))
        assert seen_models[0] == "openai/gpt-oss-120b"

        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_model", "anthropic/claude-sonnet-4-5", raising=False,
        )
        await run(SimpleNamespace(metadata={"force": True}))
        assert "anthropic/claude-sonnet-4-5" in seen_models

    @pytest.mark.asyncio
    async def test_run_force_cross_vendor_bypasses_vendor_skip_only(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash,
            run,
        )

        class FakePool:
            is_initialized = True

            async def fetch(self, query, *args):
                if "FROM b2b_reasoning_synthesis" in query:
                    return [{
                        "vendor_name": "ModelVendor",
                        "evidence_hash": _compute_pool_hash(_make_layers()),
                        "has_witness_pack": True,
                        "has_metric_refs": True,
                        "has_witness_refs": True,
                        "synthesis": json.dumps({
                            "reasoning_contracts": {
                                "vendor_core_reasoning": {
                                    "causal_narrative": {"confidence": "medium"},
                                    "segment_playbook": {"confidence": "medium"},
                                    "timing_intelligence": {"confidence": "medium"},
                                },
                                "displacement_reasoning": {
                                    "competitive_reframes": {"confidence": "medium"},
                                    "migration_proof": {"confidence": "medium"},
                                },
                                "account_reasoning": {"confidence": "medium"},
                                "category_reasoning": {"confidence": "medium"},
                            },
                            "packet_artifacts": {
                                "witness_pack": [{"witness_id": "w1"}],
                            },
                        }),
                    }]
                return []

            async def execute(self, query, *args):
                return None

        class FakeLLM:
            model = "fake-reasoner"

        seen = {}
        layers = _make_layers()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"ModelVendor": layers}

        async def _fake_run_cross_vendor_synthesis(**kwargs):
            seen["force"] = kwargs["force"]
            seen["vendor_names"] = sorted(kwargs["vendor_pools"].keys())
            return (4, 0, 3210, 4, 0)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: FakeLLM(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._run_cross_vendor_synthesis",
            _fake_run_cross_vendor_synthesis,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_model", "anthropic/claude-sonnet-4-5", raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force_cross_vendor": True}))

        assert result["vendors_reasoned"] == 0
        assert result["vendors_skipped"] == 1
        assert result["cross_vendor_succeeded"] == 4
        assert result["cross_vendor_failed"] == 0
        assert result["cross_vendor_tokens"] == 3210
        assert result["cross_vendor_mirrored"] == 4
        assert result["force"] is False
        assert result["force_cross_vendor"] is True
        assert seen["force"] is True
        assert seen["vendor_names"] == ["ModelVendor"]

    @pytest.mark.asyncio
    async def test_run_changed_vendors_only_prunes_scoped_cross_vendor_targets(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import compress_vendor_pools
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash,
            run,
        )

        vendor_a_layers = _make_layers()
        vendor_b_layers = _make_layers()
        vendor_c_layers = _make_layers()
        vendor_c_layers["category"] = {"category": "CRM"}
        vendor_pools = {
            "Salesforce": vendor_a_layers,
            "HubSpot": vendor_b_layers,
            "Dynamics": vendor_c_layers,
        }
        scoped_category_name = "crm"
        prior_synthesis = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"confidence": "medium"},
                    "segment_playbook": {"confidence": "medium"},
                    "timing_intelligence": {"confidence": "medium"},
                },
                "displacement_reasoning": {
                    "competitive_reframes": {"confidence": "medium"},
                    "migration_proof": {"confidence": "medium"},
                },
                "account_reasoning": {"confidence": "medium"},
                "category_reasoning": {"confidence": "medium"},
            },
            "packet_artifacts": {
                "witness_pack": [{"witness_id": "w1"}],
            },
            "_quality_status": "pass",
        }

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                if "FROM b2b_reasoning_synthesis" in query:
                    return [
                        {
                            "vendor_name": "Salesforce",
                            "as_of_date": date.today(),
                            "evidence_hash": _compute_pool_hash(vendor_a_layers),
                            "has_witness_pack": True,
                            "has_metric_refs": True,
                            "has_witness_refs": True,
                            "synthesis": json.dumps(prior_synthesis),
                        },
                        {
                            "vendor_name": "HubSpot",
                            "as_of_date": date.today(),
                            "evidence_hash": _compute_pool_hash(vendor_b_layers),
                            "has_witness_pack": True,
                            "has_metric_refs": True,
                            "has_witness_refs": True,
                            "synthesis": json.dumps(prior_synthesis),
                        },
                        {
                            "vendor_name": "Dynamics",
                            "as_of_date": date.today(),
                            "evidence_hash": "outdated-hash",
                            "has_witness_pack": True,
                            "has_metric_refs": True,
                            "has_witness_refs": True,
                            "synthesis": json.dumps(prior_synthesis),
                        },
                    ]
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                self.executemany_calls.append((query, list(args_iterable)))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        packet = compress_vendor_pools("Dynamics", vendor_c_layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        fake_pool = FakePool()
        fake_llm = FakeLLM(json.dumps(valid_synthesis))
        seen = {}

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return vendor_pools

        async def _fake_run_cross_vendor_synthesis(**kwargs):
            seen["scope_pairwise_pairs"] = kwargs["scope_pairwise_pairs"]
            seen["scope_category_names"] = kwargs["scope_category_names"]
            seen["scope_asymmetry_pairs"] = kwargs["scope_asymmetry_pairs"]
            return (1, 0, 100, 1, 0)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._run_cross_vendor_synthesis",
            _fake_run_cross_vendor_synthesis,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 1, raising=False,
        )

        result = await run(SimpleNamespace(metadata={
            "scope_type": "competitive_set",
            "scope_id": "scope-set-1",
            "scope_vendor_names": ["Salesforce", "HubSpot", "Dynamics"],
            "scope_pairwise_pairs": [["salesforce", "hubspot"], ["salesforce", "dynamics"]],
            "scope_category_names": [scoped_category_name],
            "scope_asymmetry_pairs": [["salesforce", "hubspot"], ["salesforce", "dynamics"]],
            "changed_vendors_only": True,
        }))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_skipped"] == 2
        assert result["cross_vendor_succeeded"] == 1
        assert result["changed_vendors_only"] is True
        assert result["cross_vendor_scoped_pairwise_jobs"] == 1
        assert result["cross_vendor_scoped_category_jobs"] == 1
        assert result["cross_vendor_scoped_asymmetry_jobs"] == 1
        assert seen["scope_pairwise_pairs"] == [("Salesforce", "Dynamics")]
        assert seen["scope_category_names"] == [scoped_category_name]
        assert seen["scope_asymmetry_pairs"] == [("Salesforce", "Dynamics")]

    @pytest.mark.asyncio
    async def test_run_scope_can_disable_vendor_phase_without_blocking_cross_vendor(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        vendor_pools = {
            "Salesforce": _make_layers(),
            "HubSpot": _make_layers(),
        }

        class FakePool:
            is_initialized = True

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                return None

        class FakeLLM:
            model = "fake-reasoner"

        seen = {}

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return vendor_pools

        async def _fake_run_cross_vendor_synthesis(**kwargs):
            seen["called"] = True
            seen["pairs"] = kwargs["scope_pairwise_pairs"]
            return (1, 0, 100, 1, 0)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: FakeLLM(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._run_cross_vendor_synthesis",
            _fake_run_cross_vendor_synthesis,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False,
        )

        result = await run(SimpleNamespace(metadata={
            "scope_type": "competitive_set",
            "scope_id": "scope-set-2",
            "scope_vendor_names": ["Salesforce", "HubSpot"],
            "scope_pairwise_pairs": [["Salesforce", "HubSpot"]],
            "vendor_synthesis_enabled": False,
            "changed_vendors_only": False,
        }))

        assert result["vendors_total"] == 0
        assert result["vendors_reasoned"] == 0
        assert result["vendors_skipped"] == 0
        assert result["cross_vendor_succeeded"] == 1
        assert seen["called"] is True
        assert seen["pairs"] == [("Salesforce", "HubSpot")]

    @pytest.mark.asyncio
    async def test_run_respects_reasoning_synthesis_enabled_flag(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", False, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["_skip_synthesis"] == "Vendor reasoning synthesis disabled"

    @pytest.mark.asyncio
    async def test_run_skips_cross_vendor_for_test_vendor_pilots(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import compress_vendor_pools
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                return None

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("PilotVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        fake_pool = FakePool()
        fake_llm = FakeLLM(json.dumps(valid_synthesis))
        seen = {"cross_vendor_called": False}

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"PilotVendor": layers}

        async def _fake_run_cross_vendor_synthesis(**kwargs):
            seen["cross_vendor_called"] = True
            return (1, 0, 100, 1, 0)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._run_cross_vendor_synthesis",
            _fake_run_cross_vendor_synthesis,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 1, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True, "test_vendors": ["PilotVendor"]}))

        assert result["vendors_reasoned"] == 1
        assert result["cross_vendor_succeeded"] == 0
        assert result["cross_vendor_failed"] == 0
        assert result["cross_vendor_tokens"] == 0
        assert seen["cross_vendor_called"] is False

    @pytest.mark.asyncio
    async def test_run_reuses_latest_unchanged_vendor_row_from_prior_day(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash,
            run,
        )

        _prior_synthesis_json = json.dumps({
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"confidence": "medium"},
                    "segment_playbook": {"confidence": "medium"},
                    "timing_intelligence": {"confidence": "medium"},
                },
                "displacement_reasoning": {
                    "competitive_reframes": {"confidence": "medium"},
                    "migration_proof": {"confidence": "medium"},
                },
                "account_reasoning": {"confidence": "medium"},
                "category_reasoning": {"confidence": "medium"},
            },
            "packet_artifacts": {
                "witness_pack": [{"witness_id": "w1"}],
            },
        })

        class FakePool:
            is_initialized = True

            async def fetch(self, query, *args):
                if "FROM b2b_reasoning_synthesis" in query:
                    assert "synthesis" in query
                    return [{
                        "vendor_name": "ModelVendor",
                        "as_of_date": date.today() - timedelta(days=1),
                        "evidence_hash": _compute_pool_hash(_make_layers()),
                        "has_witness_pack": True,
                        "has_metric_refs": True,
                        "has_witness_refs": True,
                        "synthesis": _prior_synthesis_json,
                    }]
                return []

            async def execute(self, query, *args):
                raise AssertionError("No vendor reasoning write expected when reusing unchanged row")

        layers = _make_layers()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"ModelVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_stale_days", 3, raising=False,
        )

        result = await run(SimpleNamespace(metadata={}))

        assert result["vendors_reasoned"] == 0
        assert result["vendors_failed"] == 0
        assert result["vendors_skipped"] == 1
        assert result["vendors_skipped_hash_reuse"] == 1
        assert result["vendors_skipped_stale_reuse"] == 0

    @pytest.mark.asyncio
    async def test_run_reuses_prior_day_legacy_hash_when_only_metadata_dates_changed(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash_legacy,
            run,
        )

        _prior_synthesis_json = json.dumps({
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"confidence": "medium"},
                    "segment_playbook": {"confidence": "medium"},
                    "timing_intelligence": {"confidence": "medium"},
                },
                "displacement_reasoning": {
                    "competitive_reframes": {"confidence": "medium"},
                    "migration_proof": {"confidence": "medium"},
                },
                "account_reasoning": {"confidence": "medium"},
                "category_reasoning": {"confidence": "medium"},
            },
            "packet_artifacts": {
                "witness_pack": [{"witness_id": "w1"}],
            },
        })

        class FakePool:
            is_initialized = True

            async def fetch(self, query, *args):
                if "FROM b2b_reasoning_synthesis" in query:
                    return [{
                        "vendor_name": "ModelVendor",
                        "as_of_date": date.today() - timedelta(days=1),
                        "evidence_hash": _compute_pool_hash_legacy(prior_layers),
                        "has_witness_pack": True,
                        "has_metric_refs": True,
                        "has_witness_refs": True,
                        "synthesis": _prior_synthesis_json,
                    }]
                return []

            async def execute(self, query, *args):
                raise AssertionError(
                    "No vendor reasoning write expected when only metadata dates changed",
                )

        current_layers = _make_layers()
        current_layers["evidence_vault"]["as_of_date"] = date.today().isoformat()
        current_layers["segment"]["as_of_date"] = date.today().isoformat()
        current_layers["temporal"]["as_of_date"] = date.today().isoformat()
        current_layers["category"]["as_of_date"] = date.today().isoformat()
        current_layers["accounts"]["as_of_date"] = date.today().isoformat()
        current_layers["displacement"][0]["as_of_date"] = date.today().isoformat()
        current_layers["evidence_vault"]["metric_snapshot"]["snapshot_date"] = date.today().isoformat()
        current_layers["evidence_vault"]["provenance"]["enrichment_window_end"] = date.today().isoformat()

        prior_layers = deepcopy(current_layers)
        prior_date = date.today() - timedelta(days=1)
        prior_layers["evidence_vault"]["as_of_date"] = prior_date.isoformat()
        prior_layers["segment"]["as_of_date"] = prior_date.isoformat()
        prior_layers["temporal"]["as_of_date"] = prior_date.isoformat()
        prior_layers["category"]["as_of_date"] = prior_date.isoformat()
        prior_layers["accounts"]["as_of_date"] = prior_date.isoformat()
        prior_layers["displacement"][0]["as_of_date"] = prior_date.isoformat()
        prior_layers["evidence_vault"]["metric_snapshot"]["snapshot_date"] = prior_date.isoformat()
        prior_layers["evidence_vault"]["provenance"]["enrichment_window_end"] = prior_date.isoformat()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            if as_of == prior_date:
                return {"ModelVendor": prior_layers}
            return {"ModelVendor": current_layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: FakePool(),
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_stale_days", 3, raising=False,
        )

        result = await run(SimpleNamespace(metadata={}))

        assert result["vendors_reasoned"] == 0
        assert result["vendors_failed"] == 0
        assert result["vendors_skipped"] == 1
        assert result["vendors_skipped_hash_reuse"] == 1

    @pytest.mark.asyncio
    async def test_run_reruns_when_latest_row_missing_packet_artifacts(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _compute_pool_hash,
            run,
        )

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                if "FROM b2b_reasoning_synthesis" in query:
                    return [{
                        "vendor_name": "ModelVendor",
                        "as_of_date": date.today(),
                        "evidence_hash": _compute_pool_hash(_make_layers()),
                        "has_witness_pack": False,
                        "has_metric_refs": True,
                        "has_witness_refs": True,
                    }]
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                self.executemany_calls.append((query, list(args_iterable)))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("ModelVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        fake_pool = FakePool()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"ModelVendor": layers}

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: FakeLLM(json.dumps(valid_synthesis)),
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )

        result = await run(SimpleNamespace(metadata={}))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_rerun_missing_packet_artifacts"] == 1
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        persisted = json.loads(synthesis_writes[0][1][5])
        assert persisted["meta"]["rerun_reason"] == "missing_packet_artifacts"

    @pytest.mark.asyncio
    async def test_run_falls_back_to_lean_vendor_payload_before_call(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks import _b2b_synthesis_validation as validation_mod
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                self.executemany_calls.append((query, list(args_iterable)))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.response = response
                self.calls = []

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                self.calls.append(messages)
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools("LeanVendor", layers)
        valid_synthesis, _ = _make_valid_synthesis(packet)
        fake_pool = FakePool()
        fake_llm = FakeLLM(json.dumps(valid_synthesis))
        normalize_packets = []
        validate_packets = []
        real_normalize = validation_mod.normalize_synthesis_source_ids
        real_validate = validation_mod.validate_synthesis

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"LeanVendor": layers}

        estimate_calls = {"count": 0}

        def _fake_estimate(prompt, payload):
            estimate_calls["count"] += 1
            return 600 if estimate_calls["count"] == 1 else 500

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        def _capture_normalize(synthesis, packet):
            normalize_packets.append(packet)
            return real_normalize(synthesis, packet)
        def _capture_validate(synthesis, packet=None, **kwargs):
            validate_packets.append(packet)
            return real_validate(synthesis, packet, **kwargs)
        monkeypatch.setattr(
            validation_mod, "normalize_synthesis_source_ids", _capture_normalize,
        )
        monkeypatch.setattr(
            validation_mod, "validate_synthesis", _capture_validate,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._approx_prompt_input_tokens",
            _fake_estimate,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_input_tokens", 550, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 1, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 1
        assert result["vendors_payload_mode_lean"] == 1
        assert result["vendors_payload_mode_full"] == 0
        assert '"section_packets"' not in fake_llm.calls[0][1].content
        assert normalize_packets[0].section_packets == {}
        assert normalize_packets[0].contradiction_rows == []
        assert normalize_packets[0].minority_signals == []
        assert validate_packets[0].section_packets == {}
        assert validate_packets[0].contradiction_rows == []
        assert validate_packets[0].minority_signals == []
        assert validate_packets[-1].section_packets == packet.section_packets
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        persisted = json.loads(synthesis_writes[0][1][5])
        assert persisted["meta"]["payload_mode"] == "lean"
        assert persisted["meta"]["section_packets_included"] is False
        assert persisted["meta"]["estimated_input_tokens"] == 500
        component_tokens = persisted["meta"]["payload_component_tokens"]
        assert component_tokens["payload_profile"] > 0
        assert component_tokens["witness_pack"] > 0
        assert component_tokens["coverage_gaps"] > 0
        assert "section_packets" not in component_tokens
        assert "contradiction_rows" not in component_tokens
        assert "minority_signals" not in component_tokens

    @pytest.mark.asyncio
    async def test_run_uses_configured_full_items_per_pool(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks import _b2b_pool_compression as pool_mod
        from atlas_brain.autonomous.tasks import _b2b_synthesis_validation as validation_mod
        from atlas_brain.autonomous.tasks._b2b_pool_compression import (
            compress_vendor_pools,
        )
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []
                self.executemany_calls = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                self.executemany_calls.append((query, list(args_iterable)))

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self, response):
                self.calls = []
                self.response = response

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                self.calls.append(messages)
                return {
                    "response": self.response,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        layers = _make_layers()
        packet = compress_vendor_pools(
            "ConfiguredVendor",
            layers,
            max_items_per_pool=4,
        )
        valid_synthesis, _ = _make_valid_synthesis(packet)
        fake_pool = FakePool()
        fake_llm = FakeLLM(json.dumps(valid_synthesis))
        real_compress_vendor_pools = pool_mod.compress_vendor_pools
        compress_calls = []
        validate_packets = []
        real_validate = validation_mod.validate_synthesis

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"ConfiguredVendor": layers}

        def _capture_compress_vendor_pools(vendor_name, layers, **kwargs):
            compress_calls.append(kwargs)
            return real_compress_vendor_pools(vendor_name, layers, **kwargs)

        def _capture_validate(synthesis, packet=None, **kwargs):
            validate_packets.append(packet)
            return real_validate(synthesis, packet, **kwargs)

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            pool_mod, "compress_vendor_pools", _capture_compress_vendor_pools,
        )
        monkeypatch.setattr(
            validation_mod, "validate_synthesis", _capture_validate,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_attempts", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_input_tokens", 20000, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_items_per_pool", 4, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 1
        assert compress_calls[0]["max_items_per_pool"] == 4
        payload = json.loads(fake_llm.calls[0][1].content)
        assert "precomputed_aggregates" not in payload
        assert "metric_ledger" not in payload
        assert payload["section_packets"]["account_packet"]["numeric_support"]["total_accounts"]["source_id"]
        assert payload["section_packets"]["displacement_packet"]["numeric_support"]["switch_volume"]["source_id"]
        synthesis_writes = [
            item for item in fake_pool.executed
            if "INSERT INTO b2b_reasoning_synthesis" in item[0]
        ]
        persisted = json.loads(synthesis_writes[0][1][5])
        assert persisted["meta"]["payload_mode"] == "full"
        assert persisted["meta"]["packet_items_per_pool"] == 4
        component_tokens = persisted["meta"]["payload_component_tokens"]
        assert component_tokens["payload_profile"] > 0
        assert component_tokens["witness_pack"] > 0
        assert component_tokens["section_packets"] > 0
        assert component_tokens["coverage_gaps"] > 0

    @pytest.mark.asyncio
    async def test_run_rejects_vendor_when_lean_payload_still_exceeds_budget(self, monkeypatch):
        from atlas_brain.config import settings
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import run

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))

            async def executemany(self, query, args_iterable):
                return None

        class FakeLLM:
            model = "fake-reasoner"

            def __init__(self):
                self.calls = []

            def chat(self, *, messages, max_tokens, temperature, **kwargs):
                self.calls.append(messages)
                return {"response": "{}", "usage": {"input_tokens": 0, "output_tokens": 0}}

        layers = _make_layers()
        fake_pool = FakePool()
        fake_llm = FakeLLM()

        async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
            return {"BudgetVendor": layers}

        estimate_calls = {"count": 0}

        def _fake_estimate(prompt, payload):
            estimate_calls["count"] += 1
            return 600 if estimate_calls["count"] == 1 else 580

        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.get_db_pool",
            lambda: fake_pool,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
            _fake_fetch_all_pool_layers,
        )
        monkeypatch.setattr(
            "atlas_brain.pipelines.llm.get_pipeline_llm",
            lambda **kw: fake_llm,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._approx_prompt_input_tokens",
            _fake_estimate,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "cross_vendor_synthesis_enabled", False, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_max_input_tokens", 550, raising=False,
        )

        result = await run(SimpleNamespace(metadata={"force": True}))

        assert result["vendors_reasoned"] == 0
        assert result["vendors_failed"] == 1
        assert result["vendors_rejected_input_budget"] == 1
        assert result["failed_vendors"][0]["stage"] == "input_budget"
        assert fake_llm.calls == []
        attempt_rows = [
            item for item in fake_pool.executed
            if "INSERT INTO artifact_attempts" in item[0]
        ]
        assert len(attempt_rows) == 1
        assert attempt_rows[0][1][5] == "generation"
        assert attempt_rows[0][1][6] == "rejected"

    @pytest.mark.asyncio
    async def test_run_cross_vendor_synthesis_records_success_attempt(self, monkeypatch):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _run_cross_vendor_synthesis,
        )

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))
                return None

        class FakeLLM:
            model = "fake-cross-vendor"

            def chat(self, *, messages, max_tokens, temperature, response_format):
                return {
                    "response": json.dumps({
                        "summary": "Alpha wins on admin simplicity",
                        "citations": ["xv:pairwise:alpha|beta"],
                    }),
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                }

        async def _fake_select_battles(
            pool,
            displacement_edges,
            evidence_lookup,
            *,
            product_profiles,
            max_battles,
            min_context_score,
        ):
            return [("Alpha", "Beta", {"score": 0.9})]

        async def _fake_select_asymmetry_pairs(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_battles",
            _fake_select_battles,
        )
        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_categories",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_asymmetry_pairs",
            _fake_select_asymmetry_pairs,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.build_pairwise_battle_packet",
            lambda vendor_a, vendor_b, edge, vendor_pools, product_profiles: {
                "analysis_type": "pairwise_battle",
                "vendors": [vendor_a, vendor_b],
                "edge": edge,
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.attach_cross_vendor_citation_registry",
            lambda packet, **kwargs: packet,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.compute_cross_vendor_evidence_hash",
            lambda packet: "xv-hash",
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.normalize_cross_vendor_contract",
            lambda parsed, analysis_type: parsed,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.materialize_cross_vendor_reference_ids",
            lambda synthesis, packet: {
                **synthesis,
                "reference_ids": {
                    "metric_ids": ["m1"],
                    "witness_ids": ["w1"],
                },
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.to_legacy_cross_vendor_conclusion",
            lambda synthesis, analysis_type, vendors, **kwargs: {
                "analysis_type": analysis_type,
                "vendors": vendors,
                "category": kwargs.get("category"),
                "conclusion": synthesis,
                "confidence": 0.9,
                "tokens_used": kwargs.get("tokens_used", 0),
                "cached": False,
            },
        )

        fake_pool = FakePool()
        vendor_pools = {
            "Alpha": _make_layers(),
            "Beta": _make_layers(),
        }
        cfg = SimpleNamespace(
            cross_vendor_max_battles=1,
            cross_vendor_battle_min_context_score=0.0,
            cross_vendor_category_min_vendors=2,
            cross_vendor_category_min_context_vendors=2,
            cross_vendor_category_min_displacement_intensity=0.0,
            cross_vendor_max_categories=0,
            cross_vendor_max_asymmetry=0,
            cross_vendor_asymmetry_pressure_delta_max=1.0,
            cross_vendor_asymmetry_review_ratio_min=1.0,
            cross_vendor_asymmetry_segment_divergence_bonus=0.0,
            cross_vendor_asymmetry_min_divergence_score=0.0,
            cross_vendor_asymmetry_min_context_score=0.0,
            cross_vendor_synthesis_concurrency=1,
            cross_vendor_synthesis_attempts=1,
            reasoning_synthesis_timeout_seconds=30.0,
            reasoning_synthesis_max_tokens=1024,
            reasoning_synthesis_temperature=0.0,
        )
        rcfg = SimpleNamespace(max_tokens=1024, temperature=0.0)

        result = await _run_cross_vendor_synthesis(
            pool=fake_pool,
            vendor_pools=vendor_pools,
            llm=FakeLLM(),
            cfg=cfg,
            today=date(2026, 3, 31),
            window_days=30,
            run_id="run-xv-1",
            force=True,
        )

        assert result == (1, 0, 150, 1, 0)
        attempt_rows = [
            item for item in fake_pool.executed
            if "INSERT INTO artifact_attempts" in item[0]
        ]
        assert len(attempt_rows) == 1
        assert attempt_rows[0][1][1] == "cross_vendor_reasoning"
        assert attempt_rows[0][1][2] == "pairwise_battle:alpha|beta"
        assert attempt_rows[0][1][3] == "run-xv-1"
        assert attempt_rows[0][1][4] == 1
        assert attempt_rows[0][1][5] == "complete"
        assert attempt_rows[0][1][6] == "succeeded"

    @pytest.mark.asyncio
    async def test_run_cross_vendor_synthesis_uses_batch_when_available(self, monkeypatch):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _run_cross_vendor_synthesis,
        )

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))
                return None

        class FakeDirectLLM:
            model = "fake-cross-vendor-direct"

            def __init__(self):
                self.calls = []

            def chat(self, **kwargs):
                self.calls.append(kwargs)
                raise AssertionError(
                    "Direct cross-vendor path should not be used when batch succeeds"
                )

        async def _fake_select_battles(
            pool,
            displacement_edges,
            evidence_lookup,
            *,
            product_profiles,
            max_battles,
            min_context_score,
        ):
            return [("Alpha", "Beta", {"score": 0.9})]

        async def _fake_select_asymmetry_pairs(*args, **kwargs):
            return []

        batch_calls = []
        fallback_marks = []

        async def _fake_run_anthropic_message_batch(**kwargs):
            batch_calls.append(kwargs)
            return SimpleNamespace(
                local_batch_id="batch-xv-1",
                provider_batch_id="provider-xv-1",
                results_by_custom_id={
                    "xv:pairwise_battle:alpha|beta": SimpleNamespace(
                        response_text=json.dumps({
                            "summary": "Alpha wins on admin simplicity",
                            "citations": ["xv:pairwise:alpha|beta"],
                        }),
                        usage={"input_tokens": 100, "output_tokens": 50},
                        error_text=None,
                    ),
                },
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=0,
                completed_items=1,
                failed_items=0,
            )

        async def _fake_mark_batch_fallback_result(**kwargs):
            fallback_marks.append(kwargs)

        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_battles",
            _fake_select_battles,
        )
        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_categories",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_asymmetry_pairs",
            _fake_select_asymmetry_pairs,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.build_pairwise_battle_packet",
            lambda vendor_a, vendor_b, edge, vendor_pools, product_profiles: {
                "analysis_type": "pairwise_battle",
                "vendors": [vendor_a, vendor_b],
                "edge": edge,
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.attach_cross_vendor_citation_registry",
            lambda packet, **kwargs: packet,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.compute_cross_vendor_evidence_hash",
            lambda packet: "xv-hash",
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.normalize_cross_vendor_contract",
            lambda parsed, analysis_type: parsed,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.materialize_cross_vendor_reference_ids",
            lambda synthesis, packet: {
                **synthesis,
                "reference_ids": {
                    "metric_ids": ["m1"],
                    "witness_ids": ["w1"],
                },
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.to_legacy_cross_vendor_conclusion",
            lambda synthesis, analysis_type, vendors, **kwargs: {
                "analysis_type": analysis_type,
                "vendors": vendors,
                "category": kwargs.get("category"),
                "conclusion": synthesis,
                "confidence": 0.9,
                "tokens_used": kwargs.get("tokens_used", 0),
                "cached": False,
            },
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
            _fake_run_anthropic_message_batch,
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
            _fake_mark_batch_fallback_result,
        )

        fake_pool = FakePool()
        vendor_pools = {
            "Alpha": _make_layers(),
            "Beta": _make_layers(),
        }
        cfg = SimpleNamespace(
            cross_vendor_max_battles=1,
            cross_vendor_battle_min_context_score=0.0,
            cross_vendor_category_min_vendors=2,
            cross_vendor_category_min_context_vendors=2,
            cross_vendor_category_min_displacement_intensity=0.0,
            cross_vendor_max_categories=0,
            cross_vendor_max_asymmetry=0,
            cross_vendor_asymmetry_pressure_delta_max=1.0,
            cross_vendor_asymmetry_review_ratio_min=1.0,
            cross_vendor_asymmetry_segment_divergence_bonus=0.0,
            cross_vendor_asymmetry_min_divergence_score=0.0,
            cross_vendor_asymmetry_min_context_score=0.0,
            cross_vendor_synthesis_concurrency=1,
            cross_vendor_synthesis_attempts=1,
            reasoning_synthesis_timeout_seconds=30.0,
            reasoning_synthesis_max_tokens=1024,
            reasoning_synthesis_temperature=0.0,
            cross_vendor_anthropic_batch_min_items=1,
        )
        batch_metrics = {
            "jobs": 0,
            "submitted_items": 0,
            "cache_prefiltered_items": 0,
            "fallback_single_call_items": 0,
            "completed_items": 0,
            "failed_items": 0,
        }
        direct_llm = FakeDirectLLM()

        result = await _run_cross_vendor_synthesis(
            pool=fake_pool,
            vendor_pools=vendor_pools,
            llm=direct_llm,
            cfg=cfg,
            today=date(2026, 3, 31),
            window_days=30,
            run_id="run-xv-batch-1",
            force=True,
            batch_llm=SimpleNamespace(model="fake-cross-vendor-batch"),
            batch_metrics=batch_metrics,
        )

        assert result == (1, 0, 150, 1, 0)
        assert len(batch_calls) == 1
        assert direct_llm.calls == []
        assert fallback_marks == []
        assert batch_metrics["jobs"] == 1
        assert batch_metrics["submitted_items"] == 1
        assert batch_metrics["completed_items"] == 1
        assert batch_metrics["failed_items"] == 0

    @pytest.mark.asyncio
    async def test_run_cross_vendor_synthesis_batch_fallback_records_usage(self, monkeypatch):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _run_cross_vendor_synthesis,
        )

        class FakePool:
            is_initialized = True

            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))
                return None

        class FakeDirectLLM:
            model = "claude-sonnet-4-5"
            name = "anthropic"

            def __init__(self):
                self.calls = []

            def chat(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "response": json.dumps({
                        "summary": "Alpha wins on admin simplicity",
                        "citations": ["xv:pairwise:alpha|beta"],
                    }),
                    "usage": {"input_tokens": 66, "output_tokens": 24},
                    "_trace_meta": {
                        "billable_input_tokens": 48,
                        "provider_request_id": "req_reason_xv_123",
                    },
                }

        async def _fake_select_battles(
            pool,
            displacement_edges,
            evidence_lookup,
            *,
            product_profiles,
            max_battles,
            min_context_score,
        ):
            return [("Alpha", "Beta", {"score": 0.9})]

        async def _fake_select_asymmetry_pairs(*args, **kwargs):
            return []

        fallback_marks = []

        async def _fake_run_anthropic_message_batch(**kwargs):
            return SimpleNamespace(
                local_batch_id="batch-xv-fallback",
                provider_batch_id="provider-xv-fallback",
                results_by_custom_id={
                    "xv:pairwise_battle:alpha|beta": SimpleNamespace(
                        response_text='{"broken_json"',
                        usage={"input_tokens": 100, "output_tokens": 50},
                        error_text=None,
                    ),
                },
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=1,
                completed_items=0,
                failed_items=1,
            )

        async def _fake_mark_batch_fallback_result(**kwargs):
            fallback_marks.append(kwargs)

        monkeypatch.setattr("atlas_brain.reasoning.cross_vendor_selection.select_battles", _fake_select_battles)
        monkeypatch.setattr("atlas_brain.reasoning.cross_vendor_selection.select_categories", lambda *args, **kwargs: [])
        monkeypatch.setattr("atlas_brain.reasoning.cross_vendor_selection.select_asymmetry_pairs", _fake_select_asymmetry_pairs)
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.build_pairwise_battle_packet",
            lambda vendor_a, vendor_b, edge, vendor_pools, product_profiles: {
                "analysis_type": "pairwise_battle",
                "vendors": [vendor_a, vendor_b],
                "edge": edge,
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.attach_cross_vendor_citation_registry",
            lambda packet, **kwargs: packet,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.compute_cross_vendor_evidence_hash",
            lambda packet: "xv-hash",
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.normalize_cross_vendor_contract",
            lambda parsed, analysis_type: parsed,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.materialize_cross_vendor_reference_ids",
            lambda synthesis, packet: {
                **synthesis,
                "reference_ids": {"metric_ids": ["m1"], "witness_ids": ["w1"]},
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.to_legacy_cross_vendor_conclusion",
            lambda synthesis, analysis_type, vendors, **kwargs: {
                "analysis_type": analysis_type,
                "vendors": vendors,
                "category": kwargs.get("category"),
                "conclusion": synthesis,
                "confidence": 0.9,
                "tokens_used": kwargs.get("tokens_used", 0),
                "cached": False,
            },
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
            _fake_run_anthropic_message_batch,
        )
        monkeypatch.setattr(
            "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
            _fake_mark_batch_fallback_result,
        )

        fake_pool = FakePool()
        vendor_pools = {"Alpha": _make_layers(), "Beta": _make_layers()}
        cfg = SimpleNamespace(
            cross_vendor_max_battles=1,
            cross_vendor_battle_min_context_score=0.0,
            cross_vendor_category_min_vendors=2,
            cross_vendor_category_min_context_vendors=2,
            cross_vendor_category_min_displacement_intensity=0.0,
            cross_vendor_max_categories=0,
            cross_vendor_max_asymmetry=0,
            cross_vendor_asymmetry_pressure_delta_max=1.0,
            cross_vendor_asymmetry_review_ratio_min=1.0,
            cross_vendor_asymmetry_segment_divergence_bonus=0.0,
            cross_vendor_asymmetry_min_divergence_score=0.0,
            cross_vendor_asymmetry_min_context_score=0.0,
            cross_vendor_synthesis_concurrency=1,
            cross_vendor_synthesis_attempts=2,
            reasoning_synthesis_timeout_seconds=30.0,
            reasoning_synthesis_max_tokens=1024,
            reasoning_synthesis_temperature=0.0,
            cross_vendor_anthropic_batch_min_items=1,
        )
        batch_metrics = {
            "jobs": 0,
            "submitted_items": 0,
            "cache_prefiltered_items": 0,
            "fallback_single_call_items": 0,
            "completed_items": 0,
            "failed_items": 0,
        }
        direct_llm = FakeDirectLLM()

        result = await _run_cross_vendor_synthesis(
            pool=fake_pool,
            vendor_pools=vendor_pools,
            llm=direct_llm,
            cfg=cfg,
            today=date(2026, 3, 31),
            window_days=30,
            run_id="run-xv-batch-fallback",
            force=True,
            batch_llm=SimpleNamespace(model="fake-cross-vendor-batch"),
            batch_metrics=batch_metrics,
        )

        assert result == (1, 0, 240, 1, 0)
        assert direct_llm.calls
        assert fallback_marks
        kwargs = fallback_marks[0]
        assert kwargs["succeeded"] is True
        assert kwargs["usage"]["input_tokens"] == 66
        assert kwargs["usage"]["output_tokens"] == 24
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert kwargs["provider_request_id"] == "req_reason_xv_123"

    @pytest.mark.asyncio
    async def test_run_cross_vendor_synthesis_rejects_oversized_prompt(self, monkeypatch):
        from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
            _run_cross_vendor_synthesis,
        )

        class FakePool:
            def __init__(self):
                self.executed = []

            async def fetch(self, query, *args):
                return []

            async def execute(self, query, *args):
                self.executed.append((query, args))
                return None

        class FakeLLM:
            model = "fake-cross-vendor"

            def chat(self, *, messages, max_tokens, temperature, response_format):
                raise AssertionError("LLM should not be called when input cap is exceeded")

        async def _fake_select_battles(
            pool,
            displacement_edges,
            evidence_lookup,
            *,
            product_profiles,
            max_battles,
            min_context_score,
        ):
            return [("Alpha", "Beta", {"score": 0.9})]

        async def _fake_select_asymmetry_pairs(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_battles",
            _fake_select_battles,
        )
        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_categories",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            "atlas_brain.reasoning.cross_vendor_selection.select_asymmetry_pairs",
            _fake_select_asymmetry_pairs,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.build_pairwise_battle_packet",
            lambda vendor_a, vendor_b, edge, vendor_pools, product_profiles: {
                "analysis_type": "pairwise_battle",
                "vendors": [vendor_a, vendor_b],
                "oversized_text": "x" * 5000,
            },
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.attach_cross_vendor_citation_registry",
            lambda packet, **kwargs: packet,
        )
        monkeypatch.setattr(
            "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.compute_cross_vendor_evidence_hash",
            lambda packet: "xv-hash",
        )

        fake_pool = FakePool()
        vendor_pools = {
            "Alpha": _make_layers(),
            "Beta": _make_layers(),
        }
        cfg = SimpleNamespace(
            cross_vendor_max_battles=1,
            cross_vendor_battle_min_context_score=0.0,
            cross_vendor_category_min_vendors=2,
            cross_vendor_category_min_context_vendors=2,
            cross_vendor_category_min_displacement_intensity=0.0,
            cross_vendor_max_categories=0,
            cross_vendor_max_asymmetry=0,
            cross_vendor_asymmetry_pressure_delta_max=1.0,
            cross_vendor_asymmetry_review_ratio_min=1.0,
            cross_vendor_asymmetry_segment_divergence_bonus=0.0,
            cross_vendor_asymmetry_min_divergence_score=0.0,
            cross_vendor_asymmetry_min_context_score=0.0,
            cross_vendor_synthesis_concurrency=1,
            cross_vendor_synthesis_attempts=1,
            cross_vendor_llm_max_input_tokens=10,
            reasoning_synthesis_timeout_seconds=30.0,
            reasoning_synthesis_max_tokens=1024,
            reasoning_synthesis_temperature=0.0,
        )
        rcfg = SimpleNamespace(max_tokens=1024, temperature=0.0)

        result = await _run_cross_vendor_synthesis(
            pool=fake_pool,
            vendor_pools=vendor_pools,
            llm=FakeLLM(),
            cfg=cfg,
            today=date(2026, 3, 31),
            window_days=30,
            run_id="run-xv-cap",
            force=True,
        )

        assert result == (0, 1, 0, 0, 1)
        attempt_rows = [
            item for item in fake_pool.executed
            if "INSERT INTO artifact_attempts" in item[0]
        ]
        assert len(attempt_rows) == 1
        assert attempt_rows[0][1][5] == "generation"
        assert attempt_rows[0][1][6] == "rejected"
