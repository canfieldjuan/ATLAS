"""Tests for Phase 2 pool compression packet widening.

Tests metric_ledger, contradiction_rows, minority_signals, coverage_gaps,
retention_proof, and their integration into CompressedPacket.to_llm_payload().
"""

import sys
from unittest.mock import MagicMock

import pytest

# Pre-mock heavy deps
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

from atlas_brain.autonomous.tasks._b2b_pool_compression import (
    CompressedPacket,
    ScoredItem,
    SourceRef,
    TrackedAggregate,
    _build_contradiction_rows,
    _build_coverage_gaps,
    _build_metric_ledger,
    _build_minority_signals,
    _build_retention_proof,
    compress_vendor_pools,
)
from atlas_brain.autonomous.tasks._b2b_witnesses import (
    build_vendor_witness_artifacts,
    compute_witness_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_weakness_item(key: str, mention_count: int = 10, urgency: float = 5.0) -> ScoredItem:
    return ScoredItem(
        data={
            "key": key,
            "mention_count_total": mention_count,
            "avg_urgency": urgency,
        },
        score=0.5,
        source_ref=SourceRef(pool="vault", kind="weakness", key=key),
    )


def _make_strength_item(key: str, mention_count: int = 10) -> ScoredItem:
    return ScoredItem(
        data={
            "key": key,
            "mention_count_total": mention_count,
            "summary": f"Strong {key} ecosystem",
        },
        score=0.5,
        source_ref=SourceRef(pool="vault", kind="strength", key=key),
    )


def _make_segment_item(kind: str, key: str, churn_rate: float, review_count: int = 20) -> ScoredItem:
    return ScoredItem(
        data={
            "churn_rate": churn_rate,
            "review_count": review_count,
        },
        score=0.5,
        source_ref=SourceRef(pool="segment", kind=kind, key=key),
    )


def _make_account_item(
    name: str, urgency: float = 6.0, is_dm: bool = False,
) -> ScoredItem:
    return ScoredItem(
        data={
            "company_name": name,
            "urgency_score": urgency,
            "decision_maker": is_dm,
        },
        score=0.5,
        source_ref=SourceRef(pool="accounts", kind="company", key=name.lower()),
    )


def _make_full_coverage_layers() -> dict:
    return {
        "evidence_vault": {
            "metric_snapshot": {
                "total_reviews": 200,
                "reviews_in_analysis_window": 180,
                "reviews_in_recent_window": 40,
                "churn_density": 0.42,
                "avg_urgency": 6.5,
                "recommend_yes": 65,
                "recommend_no": 35,
                "recommend_ratio": 0.65,
                "price_complaint_rate": 0.31,
                "dm_churn_rate": 0.22,
                "positive_review_pct": 0.44,
                "displacement_mention_count": 18,
                "keyword_spike_count": 3,
                "avg_rating": 3.6,
                "negative_review_pct": 0.41,
            },
            "provenance": {
                "enrichment_window_start": "2026-01-01",
                "enrichment_window_end": "2026-03-28",
            },
            "company_signals": [
                {
                    "company_name": "Acme Corp",
                    "urgency_score": 8.8,
                    "buying_stage": "evaluation",
                    "decision_maker": True,
                    "review_id": "r1",
                },
                {
                    "company_name": "Beta Inc",
                    "urgency_score": 7.4,
                    "buying_stage": "active_purchase",
                    "decision_maker": False,
                    "review_id": "r2",
                },
            ],
            "weakness_evidence": [
                {
                    "key": "pricing",
                    "mention_count_total": 30,
                    "mention_count_recent": 10,
                    "avg_urgency": 7.3,
                },
                {
                    "key": "security",
                    "mention_count_total": 6,
                    "mention_count_recent": 2,
                    "avg_urgency": 8.4,
                },
            ],
            "strength_evidence": [
                {
                    "key": "integrations",
                    "mention_count_total": 24,
                    "mention_count_recent": 7,
                    "summary": "Broad ecosystem",
                },
                {
                    "key": "support",
                    "mention_count_total": 12,
                    "mention_count_recent": 4,
                    "summary": "Responsive support",
                },
            ],
        },
        "segment": {
            "affected_departments": [
                {"department": "Finance", "review_count": 18, "churn_rate": 0.55},
            ],
            "affected_roles": [
                {"role_type": "Engineering", "review_count": 22, "churn_rate": 0.61},
            ],
            "contract_segments": [
                {"segment": "Annual", "count": 14, "churn_rate": 0.47},
            ],
            "usage_duration_segments": [
                {"duration": "0-6 months", "count": 16, "churn_rate": 0.58},
            ],
            "top_use_cases_under_pressure": [
                {"use_case": "Reporting", "mention_count": 15, "confidence_score": 0.7},
            ],
            "budget_pressure": {
                "dm_churn_rate": 0.28,
                "price_increase_rate": 0.19,
                "price_increase_count": 9,
                "annual_spend_signals": ["budget", "renewal"],
                "price_per_seat_signals": ["seat expansion"],
            },
            "affected_company_sizes": {
                "avg_seat_count": 180,
                "median_seat_count": 90,
                "max_seat_count": 600,
                "size_distribution": [
                    {"segment": "100-500", "review_count": 21, "churn_rate": 0.49},
                ],
            },
            "buying_stage_distribution": [
                {"stage": "evaluation", "count": 8},
                {"stage": "renewal", "count": 3},
            ],
        },
        "temporal": {
            "timeline_signal_summary": {
                "evaluation_deadline_signals": 5,
                "contract_end_signals": 4,
                "renewal_signals": 6,
                "budget_cycle_signals": 3,
            },
            "keyword_spikes": {
                "spike_count": 2,
                "keyword_details": [
                    {"keyword": "migration", "magnitude": 2.6, "change_pct": 0.8, "volume": 9},
                ],
            },
            "sentiment_trajectory": {
                "declining": 14,
                "stable": 6,
                "improving": 4,
                "total": 24,
                "declining_pct": 0.58,
                "improving_pct": 0.17,
            },
            "immediate_triggers": [
                {"trigger_type": "timeline_signal", "label": "Renewal pressure", "urgency": 8.1},
            ],
            "evaluation_deadlines": [
                {"trigger_type": "deadline", "deadline": "Q2 cutoff", "urgency": 8.6},
            ],
            "turning_points": [
                {"trigger": "pricing reset", "mentions": 7},
            ],
            "sentiment_tenure": [
                {"tenure": "6-12 months", "count": 9},
            ],
        },
        "displacement": [
            {
                "to_vendor": "HubSpot",
                "flow_summary": {
                    "mention_count": 11,
                    "explicit_switch_count": 4,
                    "active_evaluation_count": 3,
                },
            },
        ],
        "category": {
            "vendor_count": 11,
            "displacement_flow_count": 7,
            "market_regime": {
                "regime_type": "price_consolidation",
                "confidence": 0.72,
                "avg_churn_velocity": 0.41,
                "avg_price_pressure": 0.36,
            },
            "council_summary": {"confidence": 0.63},
        },
        "accounts": {
            "summary": {
                "decision_maker_count": 2,
            },
            "accounts": [
                {
                    "company_name": "Acme Corp",
                    "urgency_score": 8.9,
                    "decision_maker": True,
                    "buying_stage": "evaluation",
                    "review_ids": ["r1"],
                },
                {
                    "company_name": "Gamma LLC",
                    "urgency_score": 7.1,
                    "decision_maker": False,
                    "buying_stage": "active_purchase",
                    "review_ids": ["r2"],
                },
            ],
        },
        "reviews": [
            {
                "id": "r1",
                "source": "g2",
                "rating": 2.0,
                "rating_max": 5,
                "summary": "Renewal pricing shock",
                "review_text": "We were quoted $50k at renewal and started evaluating HubSpot.",
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
                            "competitor": "HubSpot",
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
                "reviewer_company": "Gamma LLC",
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
                            "company_name": "Gamma LLC",
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


# ---------------------------------------------------------------------------
# Tests: _build_metric_ledger
# ---------------------------------------------------------------------------

class TestBuildMetricLedger:
    def test_includes_known_metrics(self):
        aggs = [
            TrackedAggregate("total_reviews", 150, "vault:metric:total_reviews"),
            TrackedAggregate("avg_urgency", 6.2, "vault:metric:avg_urgency"),
            TrackedAggregate("churn_density", 0.35, "vault:metric:churn_density"),
        ]
        ledger = _build_metric_ledger(aggs)
        labels = {e["label"] for e in ledger}
        assert "total_reviews" in labels
        assert "avg_urgency" in labels
        assert "churn_density" in labels

    def test_scoped_with_window_and_surfaces(self):
        aggs = [TrackedAggregate("total_reviews", 150, "vault:metric:total_reviews")]
        ledger = _build_metric_ledger(aggs, analysis_window_days=60)
        entry = ledger[0]
        assert entry["time_window_days"] == 60
        assert entry["scope"] == "review_volume"
        assert "report" in entry["allowed_surfaces"]
        assert entry["_sid"] == "vault:metric:total_reviews"

    def test_dm_churn_rate_internal_only(self):
        aggs = [TrackedAggregate("dm_churn_rate", 0.12, "vault:metric:dm_churn_rate")]
        ledger = _build_metric_ledger(aggs)
        entry = ledger[0]
        assert "campaign" not in entry["allowed_surfaces"]
        assert "report" in entry["allowed_surfaces"]

    def test_vault_weakness_metrics_included(self):
        aggs = [
            TrackedAggregate(
                "vault_weakness_pricing_mention_count_total", 16,
                "vault:weakness:pricing:mention_count_total",
            ),
        ]
        ledger = _build_metric_ledger(aggs)
        assert len(ledger) == 1
        assert ledger[0]["label"] == "vault_weakness_pricing_mention_count_total"

    def test_deduplicates(self):
        aggs = [
            TrackedAggregate("total_reviews", 150, "vault:metric:total_reviews"),
            TrackedAggregate("total_reviews", 150, "vault:metric:total_reviews"),
        ]
        ledger = _build_metric_ledger(aggs)
        assert len(ledger) == 1


# ---------------------------------------------------------------------------
# Tests: _build_contradiction_rows
# ---------------------------------------------------------------------------

class TestBuildContradictionRows:
    def test_detects_churn_rate_contradiction(self):
        """High-churn and low-churn segments in the same kind -> contradiction."""
        pools = {
            "segment": [
                _make_segment_item("role", "ops_team", churn_rate=0.6),
                _make_segment_item("role", "engineering", churn_rate=0.1),
            ],
        }
        rows = _build_contradiction_rows(pools)
        assert len(rows) >= 1
        assert rows[0]["dimension"] == "role"
        assert rows[0]["statement_a"] == "negative"
        assert rows[0]["statement_b"] == "positive"

    def test_detects_vault_strength_weakness_overlap(self):
        """Same dimension in both weakness and strength evidence -> contradiction."""
        pools = {
            "evidence_vault": [
                _make_weakness_item("support"),
                _make_strength_item("support"),
            ],
            "segment": [],
        }
        rows = _build_contradiction_rows(pools)
        vault_rows = [r for r in rows if "vault:" in r["_sid"]]
        assert len(vault_rows) == 1
        assert vault_rows[0]["dimension"] == "support"

    def test_no_contradiction_when_all_same_direction(self):
        pools = {
            "segment": [
                _make_segment_item("role", "ops", churn_rate=0.5),
                _make_segment_item("role", "support", churn_rate=0.6),
            ],
        }
        rows = _build_contradiction_rows(pools)
        segment_rows = [r for r in rows if "segment:" in r["_sid"]]
        assert segment_rows == []

    def test_empty_pools(self):
        assert _build_contradiction_rows({}) == []


# ---------------------------------------------------------------------------
# Tests: _build_minority_signals
# ---------------------------------------------------------------------------

class TestBuildMinoritySignals:
    def test_detects_rare_but_severe(self):
        pools = {
            "evidence_vault": [
                _make_weakness_item("security", mention_count=3, urgency=8.7),
            ],
            "accounts": [],
        }
        signals = _build_minority_signals(pools, [])
        assert len(signals) == 1
        assert signals[0]["label"] == "security"
        assert signals[0]["reason"] == "rare_but_severe"
        assert signals[0]["urgency"] == 8.7
        assert signals[0]["count"] == 3

    def test_ignores_high_count_weakness(self):
        pools = {
            "evidence_vault": [
                _make_weakness_item("pricing", mention_count=50, urgency=9.0),
            ],
            "accounts": [],
        }
        signals = _build_minority_signals(pools, [])
        assert signals == []

    def test_detects_dm_extreme_urgency(self):
        pools = {
            "evidence_vault": [],
            "accounts": [
                _make_account_item("Acme Corp", urgency=9.5, is_dm=True),
            ],
        }
        signals = _build_minority_signals(pools, [])
        assert len(signals) == 1
        assert "dm_alert" in signals[0]["label"]

    def test_ignores_low_urgency_dm(self):
        pools = {
            "evidence_vault": [],
            "accounts": [
                _make_account_item("Acme Corp", urgency=6.0, is_dm=True),
            ],
        }
        signals = _build_minority_signals(pools, [])
        assert signals == []


# ---------------------------------------------------------------------------
# Tests: _build_coverage_gaps
# ---------------------------------------------------------------------------

class TestBuildCoverageGaps:
    def test_thin_segment(self):
        pools = {
            "segment": [
                _make_segment_item("role", "finance", churn_rate=0.5, review_count=7),
            ],
            "displacement": [ScoredItem(
                data={}, score=0.5,
                source_ref=SourceRef("displacement", "flow", "a_to_b"),
            )],
            "accounts": [_make_account_item(f"c{i}") for i in range(5)],
        }
        gaps = _build_coverage_gaps(pools, [])
        thin = [g for g in gaps if g["type"] == "thin_segment_sample"]
        assert len(thin) == 1
        assert thin[0]["sample_size"] == 7
        assert thin[0]["_sid"].startswith("gap:thin_segment:")

    def test_missing_displacement(self):
        pools = {
            "segment": [],
            "accounts": [_make_account_item(f"c{i}") for i in range(5)],
        }
        gaps = _build_coverage_gaps(pools, [])
        missing = [g for g in gaps if g["type"] == "missing_pool"]
        assert len(missing) == 1
        assert missing[0]["area"] == "displacement"
        assert missing[0]["_sid"] == "gap:missing_pool:displacement"

    def test_thin_accounts(self):
        pools = {
            "segment": [],
            "displacement": [ScoredItem(
                data={}, score=0.5,
                source_ref=SourceRef("displacement", "flow", "x"),
            )],
            "accounts": [_make_account_item("solo")],
        }
        gaps = _build_coverage_gaps(pools, [])
        thin_acct = [g for g in gaps if g["type"] == "thin_account_signals"]
        assert len(thin_acct) == 1

    def test_shallow_evidence_window(self):
        aggs = [
            TrackedAggregate("enrichment_window_start", "2026-03-20", "vault:provenance:enrichment_window_start"),
            TrackedAggregate("enrichment_window_end", "2026-03-25", "vault:provenance:enrichment_window_end"),
        ]
        pools = {
            "segment": [],
            "displacement": [ScoredItem(
                data={}, score=0.5,
                source_ref=SourceRef("displacement", "flow", "x"),
            )],
            "accounts": [_make_account_item(f"c{i}") for i in range(5)],
        }
        gaps = _build_coverage_gaps(pools, aggs)
        shallow = [g for g in gaps if g["type"] == "shallow_evidence_window"]
        assert len(shallow) == 1
        assert shallow[0]["sample_size"] == 5  # 5 days


# ---------------------------------------------------------------------------
# Tests: _build_retention_proof
# ---------------------------------------------------------------------------

class TestBuildRetentionProof:
    def test_extracts_strong_strengths(self):
        pools = {
            "evidence_vault": [
                _make_strength_item("integrations", mention_count=15),
                _make_strength_item("ecosystem", mention_count=10),
            ],
        }
        proofs = _build_retention_proof(pools)
        assert len(proofs) == 2
        areas = {p["area"] for p in proofs}
        assert "integrations" in areas
        assert "ecosystem" in areas
        assert proofs[0]["_sid"].startswith("vault:strength:")

    def test_excludes_weak_strengths(self):
        pools = {
            "evidence_vault": [
                _make_strength_item("niche_feature", mention_count=1),
            ],
        }
        proofs = _build_retention_proof(pools)
        assert proofs == []

    def test_ignores_weakness_items(self):
        pools = {
            "evidence_vault": [
                _make_weakness_item("pricing", mention_count=20),
            ],
        }
        proofs = _build_retention_proof(pools)
        assert proofs == []


# ---------------------------------------------------------------------------
# Tests: CompressedPacket.to_llm_payload integration
# ---------------------------------------------------------------------------

class TestPacketPayloadIntegration:
    def test_governance_sections_in_payload(self):
        packet = CompressedPacket(
            vendor_name="TestVendor",
            pools={},
            aggregates=[],
            metric_ledger=[{"label": "total_reviews", "value": 100, "_sid": "vault:metric:total_reviews"}],
            contradiction_rows=[{"dimension": "support", "_sid": "segment:contradiction:support"}],
            minority_signals=[{"label": "security", "urgency": 8.5, "_sid": "vault:minority:security"}],
            coverage_gaps=[{"type": "missing_pool", "area": "displacement"}],
            retention_proof=[{"area": "integrations", "_sid": "vault:strength:integrations"}],
            witness_pack=[{"witness_id": "witness:r1:0", "_sid": "witness:r1:0", "excerpt_text": "quoted $50k"}],
            section_packets={"causal_packet": {"witness_ids": ["witness:r1:0"]}},
        )
        payload = packet.to_llm_payload()

        assert "metric_ledger" in payload
        assert len(payload["metric_ledger"]) == 1
        assert "contradiction_rows" in payload
        assert "minority_signals" in payload
        assert "coverage_gaps" in payload
        assert "retention_proof" in payload
        assert "witness_pack" in payload
        assert "section_packets" in payload

    def test_compact_metric_ledger_omits_redundant_fields(self):
        packet = CompressedPacket(
            vendor_name="TestVendor",
            pools={},
            aggregates=[],
            metric_ledger=[{
                "label": "total_reviews",
                "value": 100,
                "scope": "review_volume",
                "time_window_days": 90,
                "allowed_surfaces": ["report"],
                "_sid": "vault:metric:total_reviews",
            }],
        )

        payload = packet.to_llm_payload(compact_metric_ledger=True)

        assert payload["metric_ledger"] == [{
            "label": "total_reviews",
            "_sid": "vault:metric:total_reviews",
        }]

    def test_compact_aggregates_only_keeps_reasoning_relevant_entries(self):
        packet = CompressedPacket(
            vendor_name="TestVendor",
            pools={
                "evidence_vault": [
                    ScoredItem(
                        data={"key": "pricing"},
                        score=1.0,
                        source_ref=SourceRef(pool="vault", kind="weakness", key="pricing"),
                    ),
                ],
            },
            aggregates=[
                TrackedAggregate("total_reviews", 100, "vault:metric:total_reviews"),
                TrackedAggregate(
                    "vault_weakness_pricing_mention_count_total",
                    12,
                    "vault:weakness:pricing:mention_count_total",
                ),
                TrackedAggregate(
                    "vault_weakness_support_mention_count_total",
                    4,
                    "vault:weakness:support:mention_count_total",
                ),
            ],
            metric_ledger=[
                {"label": "total_reviews", "value": 100, "scope": "review_volume", "_sid": "vault:metric:total_reviews"},
                {
                    "label": "vault_weakness_pricing_mention_count_total",
                    "value": 12,
                    "scope": "weakness_mentions",
                    "_sid": "vault:weakness:pricing:mention_count_total",
                },
                {
                    "label": "vault_weakness_support_mention_count_total",
                    "value": 4,
                    "scope": "weakness_mentions",
                    "_sid": "vault:weakness:support:mention_count_total",
                },
            ],
        )

        payload = packet.to_llm_payload(
            compact_metric_ledger=True,
            compact_aggregates=True,
        )

        assert set(payload["precomputed_aggregates"]) == {
            "total_reviews",
            "vault_weakness_pricing_mention_count_total",
        }
        assert {entry["_sid"] for entry in payload["metric_ledger"]} == {
            "vault:metric:total_reviews",
            "vault:weakness:pricing:mention_count_total",
        }

    def test_empty_governance_sections_omitted(self):
        packet = CompressedPacket(
            vendor_name="TestVendor",
            pools={},
            aggregates=[],
        )
        payload = packet.to_llm_payload()
        assert "metric_ledger" not in payload
        assert "contradiction_rows" not in payload
        assert "minority_signals" not in payload
        assert "coverage_gaps" not in payload
        assert "retention_proof" not in payload
        assert "witness_pack" not in payload
        assert "section_packets" not in payload

    def test_governance_source_ids_in_packet(self):
        packet = CompressedPacket(
            vendor_name="TestVendor",
            pools={},
            aggregates=[],
            metric_ledger=[{"label": "x", "_sid": "vault:metric:x"}],
            contradiction_rows=[{"dimension": "y", "_sid": "segment:contradiction:y"}],
            coverage_gaps=[{"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"}],
            retention_proof=[{"area": "z", "_sid": "vault:strength:z"}],
        )
        sids = packet.source_ids()
        assert "vault:metric:x" in sids
        assert "segment:contradiction:y" in sids
        assert "gap:missing_pool:displacement" in sids
        assert "vault:strength:z" in sids

    def test_witness_source_ids_in_packet(self):
        packet = CompressedPacket(
            vendor_name="TestVendor",
            pools={},
            aggregates=[],
            witness_pack=[{"witness_id": "witness:r1:0", "_sid": "witness:r1:0", "excerpt_text": "quoted $50k"}],
        )

        assert "witness:r1:0" in packet.source_ids()


# ---------------------------------------------------------------------------
# Integration: compress_vendor_pools produces governance sections
# ---------------------------------------------------------------------------

class TestCompressVendorPoolsIntegration:
    def test_full_compression_includes_governance(self):
        layers = {
            "evidence_vault": {
                "metric_snapshot": {
                    "total_reviews": 200,
                    "avg_urgency": 6.5,
                    "churn_density": 0.4,
                    "recommend_ratio": 0.3,
                },
                "provenance": {
                    "enrichment_window_start": "2026-01-01",
                    "enrichment_window_end": "2026-03-28",
                },
                "weakness_evidence": [
                    {
                        "key": "pricing",
                        "mention_count_total": 30,
                        "avg_urgency": 7.0,
                    },
                    {
                        "key": "security",
                        "mention_count_total": 3,
                        "avg_urgency": 9.0,
                    },
                ],
                "strength_evidence": [
                    {
                        "key": "integrations",
                        "mention_count_total": 25,
                        "summary": "Broad ecosystem",
                    },
                    {
                        "key": "pricing",
                        "mention_count_total": 8,
                        "summary": "Competitive for SMB",
                    },
                ],
                "company_signals": [],
            },
            "segment": {
                "affected_roles": [
                    {"role_type": "Engineering", "review_count": 50, "churn_rate": 0.6},
                    {"role_type": "Finance", "review_count": 30, "churn_rate": 0.1},
                ],
                "affected_departments": [],
            },
            "displacement": [],
            "accounts": {},
        }
        packet = compress_vendor_pools("TestVendor", layers)

        assert len(packet.metric_ledger) > 0
        # pricing appears in both weakness and strength -> vault contradiction
        pricing_contradictions = [
            r for r in packet.contradiction_rows
            if r.get("dimension") == "pricing"
        ]
        assert len(pricing_contradictions) == 1

        # security weakness is rare (3 mentions) but severe (9.0 urgency)
        security_minority = [
            s for s in packet.minority_signals
            if s.get("label") == "security"
        ]
        assert len(security_minority) == 1

        # integrations is a strong strength
        integration_proof = [
            p for p in packet.retention_proof
            if p.get("area") == "integrations"
        ]
        assert len(integration_proof) == 1

        # displacement is empty -> coverage gap
        disp_gap = [
            g for g in packet.coverage_gaps
            if g.get("type") == "missing_pool" and g.get("area") == "displacement"
        ]
        assert len(disp_gap) == 1

        # Verify payload serialization
        payload = packet.to_llm_payload()
        assert "metric_ledger" in payload
        assert "contradiction_rows" in payload
        assert "minority_signals" in payload
        assert "coverage_gaps" in payload
        assert "retention_proof" in payload
        assert "witness_pack" not in payload

    def test_metric_ledger_covers_all_claimable_packet_aggregates(self):
        packet = compress_vendor_pools("CoverageVendor", _make_full_coverage_layers())

        aggregate_source_ids = {agg.source_id for agg in packet.aggregates}
        ledger_source_ids = {entry["_sid"] for entry in packet.metric_ledger}
        intentional_exclusions = {
            "vault:provenance:enrichment_window_start",
            "vault:provenance:enrichment_window_end",
        }
        claimable_source_ids = aggregate_source_ids - intentional_exclusions
        unexpected_uncovered = sorted(claimable_source_ids - ledger_source_ids)

        coverage_counts = (
            len(aggregate_source_ids),
            len(claimable_source_ids & ledger_source_ids),
            len(intentional_exclusions),
            len(unexpected_uncovered),
        )

        assert unexpected_uncovered == [], (
            f"metric_ledger missing claimable aggregate source_ids: {unexpected_uncovered}; "
            f"coverage_counts={coverage_counts}"
        )
        assert aggregate_source_ids & intentional_exclusions == intentional_exclusions
        assert coverage_counts == (68, 66, 2, 0)

    def test_full_compression_includes_witness_pack(self):
        packet = compress_vendor_pools("CoverageVendor", _make_full_coverage_layers())

        assert packet.witness_pack
        assert packet.section_packets["causal_packet"]["witness_ids"]
        witness_ids = {w["witness_id"] for w in packet.witness_pack}
        assert any(witness_id.startswith("witness:r1") for witness_id in witness_ids)

    def test_reasoning_payload_is_witness_first(self):
        packet = compress_vendor_pools("CoverageVendor", _make_full_coverage_layers())

        payload = packet.to_reasoning_payload()

        assert payload["payload_profile"] == "witness_first_v2"
        assert "witness_pack" in payload
        assert "section_packets" in payload
        assert "precomputed_aggregates" not in payload
        assert "metric_ledger" not in payload
        assert "compact_context" not in payload
        assert "evidence_vault" not in payload
        assert "segment" not in payload
        assert "accounts" not in payload
        assert "displacement" not in payload
        assert "temporal" not in payload
        assert "category" not in payload
        witness = payload["witness_pack"][0]
        assert witness["_sid"]
        assert "witness_id" not in witness
        assert "vendor_name" not in witness
        assert "review_id" not in witness
        assert "source_span_id" not in witness
        assert "selection_reason" not in witness
        assert "salience_score" not in witness
        assert "candidate_types" not in witness
        assert "specificity_score" not in witness
        assert "witness_hash" not in witness
        assert "generic_reason" not in witness
        assert "_witness_governance" not in payload["section_packets"]
        assert payload["section_packets"]["account_packet"]["top_accounts"][0]["name"]
        assert payload["section_packets"]["displacement_packet"]["numeric_support"]["switch_volume"]["source_id"]

    def test_reasoning_payload_component_tokens_are_emitted(self):
        packet = compress_vendor_pools("CoverageVendor", _make_full_coverage_layers())

        estimates = packet.reasoning_payload_component_tokens()

        assert estimates["payload_profile"] > 0
        assert estimates["witness_pack"] > 0
        assert estimates["section_packets"] > 0
        assert "precomputed_aggregates" not in estimates
        assert "metric_ledger" not in estimates
        assert "compact_context" not in estimates
        assert "evidence_vault" not in estimates
        assert "accounts" not in estimates

    def test_reasoning_payload_section_packets_include_numeric_support(self):
        packet = compress_vendor_pools("CoverageVendor", _make_full_coverage_layers())

        payload = packet.to_reasoning_payload()
        packets = payload["section_packets"]

        assert packets["timing_packet"]["numeric_support"]["active_eval_signals"]["source_id"]
        assert packets["account_packet"]["numeric_support"]["total_accounts"]["source_id"]
        assert packets["account_packet"]["top_accounts"]
        assert packets["category_packet"]["regime_candidates"]

    def test_reasoning_payload_section_packet_limits_are_configurable(self, monkeypatch):
        from atlas_brain.config import settings

        packet = compress_vendor_pools("CoverageVendor", _make_full_coverage_layers())
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_segment_candidate_limit", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_temporal_candidate_limit", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_displacement_candidate_limit", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_account_candidate_limit", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_category_candidate_limit", 1, raising=False,
        )
        monkeypatch.setattr(
            settings.b2b_churn, "reasoning_synthesis_retention_candidate_limit", 1, raising=False,
        )

        packets = packet.to_reasoning_payload()["section_packets"]

        assert len(packets["segment_packet"]["priority_segment_candidates"]) <= 1
        assert len(packets["timing_packet"]["trigger_candidates"]) <= 1
        assert len(packets["displacement_packet"]["destination_candidates"]) <= 1
        assert len(packets["account_packet"]["top_accounts"]) <= 1
        assert len(packets["category_packet"]["regime_candidates"]) <= 1
        assert len(packets["retention_packet"]["strength_candidates"]) <= 1


class TestWitnessGovernance:
    def test_witness_hash_ignores_selection_metadata(self):
        witness = {
            "witness_id": "witness:r1:0",
            "_sid": "witness:r1:0",
            "review_id": "r1",
            "excerpt_text": "At renewal we were quoted $50,000 for 120 seats.",
            "source": "g2",
            "source_span_id": "span-1",
            "signal_type": "negative_signal",
            "pain_category": "pricing",
            "competitor": "CompetitorA",
            "time_anchor": "Q2 renewal",
            "replacement_mode": "suite_consolidation",
            "operating_model_shift": "async_to_sync",
            "productivity_delta_claim": "less_productive",
            "signal_tags": ["explicit_dollar"],
            "witness_type": "timing",
            "selection_reason": "selected_for_timing",
            "specificity_score": 7.5,
            "generic_reason": None,
        }

        alternate = dict(witness)
        alternate["witness_type"] = "flex"
        alternate["selection_reason"] = "selected_for_flex"
        alternate["specificity_score"] = 6.0
        alternate["generic_reason"] = "low_specificity_score"

        assert compute_witness_hash(witness) == compute_witness_hash(alternate)

    def test_witness_hash_changes_when_excerpt_changes(self):
        witness = {
            "witness_id": "witness:r1:0",
            "_sid": "witness:r1:0",
            "review_id": "r1",
            "excerpt_text": "Quoted $50,000 at renewal.",
            "source": "g2",
            "signal_type": "negative_signal",
            "pain_category": "pricing",
            "signal_tags": [],
        }
        changed = dict(witness)
        changed["excerpt_text"] = "Quoted $75,000 at renewal."

        assert compute_witness_hash(witness) != compute_witness_hash(changed)

    def test_generic_excerpt_is_filtered_from_normal_witness_pack(self):
        reviews = [{
            "id": "r1",
            "source": "g2",
            "enrichment": {
                "pain_categories": [{"category": "pricing", "severity": "primary"}],
                "evidence_spans": [{
                    "text": "Great tool, easy to use, very helpful.",
                    "pain_category": "pricing",
                    "flags": [],
                    "signal_type": "negative_signal",
                }],
            },
        }]

        selected, section_packets = build_vendor_witness_artifacts(
            "GenericVendor",
            reviews,
            min_specificity_score=2.0,
            fallback_min_witnesses=0,
        )

        assert selected == []
        assert section_packets["_witness_governance"]["filtered_generic_candidates"] == 1
        assert section_packets["_witness_governance"]["thin_specific_witness_pool"] is False

    def test_specific_excerpt_survives_with_hash_and_specificity_score(self):
        reviews = [{
            "id": "r1",
            "source": "g2",
            "reviewer_company": "Acme Corp",
            "enrichment": {
                "pain_categories": [{"category": "pricing", "severity": "primary"}],
                "reviewer_context": {"decision_maker": True},
                "churn_signals": {"contract_renewal_mentioned": True},
                "evidence_spans": [{
                    "text": (
                        "At renewal we were quoted $50,000 for 120 seats and started "
                        "evaluating CompetitorA because pricing no longer worked."
                    ),
                    "pain_category": "pricing",
                    "competitor": "CompetitorA",
                    "flags": ["explicit_dollar"],
                    "signal_type": "negative_signal",
                    "time_anchor": "Q2 renewal",
                }],
            },
        }]

        selected, _section_packets = build_vendor_witness_artifacts(
            "SpecificVendor",
            reviews,
            min_specificity_score=2.0,
            fallback_min_witnesses=0,
        )

        assert selected
        assert selected[0]["specificity_score"] >= 2.0
        assert not selected[0].get("generic_reason")
        assert selected[0]["witness_hash"]

    def test_generic_fallback_marks_thin_pool(self):
        reviews = [{
            "id": "r1",
            "source": "g2",
            "enrichment": {
                "pain_categories": [{"category": "pricing", "severity": "primary"}],
                "evidence_spans": [{
                    "text": "Good platform, works well, very helpful.",
                    "pain_category": "pricing",
                    "flags": [],
                    "signal_type": "negative_signal",
                }],
            },
        }]

        selected, section_packets = build_vendor_witness_artifacts(
            "FallbackVendor",
            reviews,
            min_specificity_score=2.0,
            fallback_min_witnesses=1,
        )

        assert len(selected) == 1
        assert selected[0]["selection_reason"].endswith("generic_fallback")
        assert selected[0]["generic_reason"]
        assert section_packets["_witness_governance"]["thin_specific_witness_pool"] is True

    def test_specificity_policy_can_be_overridden(self):
        reviews = [{
            "id": "r1",
            "source": "g2",
            "enrichment": {
                "pain_categories": [{"category": "pricing", "severity": "primary"}],
                "evidence_spans": [{
                    "text": "pricing issue",
                    "pain_category": "pricing",
                    "flags": [],
                    "signal_type": "negative_signal",
                }],
            },
        }]

        selected, _section_packets = build_vendor_witness_artifacts(
            "PolicyVendor",
            reviews,
            min_specificity_score=0.5,
            fallback_min_witnesses=0,
            concrete_patterns=["pricing issue"],
            short_excerpt_chars=5,
            specificity_weights={"concrete_pattern": 1.0},
        )

        assert selected
