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
        )
        payload = packet.to_llm_payload()

        assert "metric_ledger" in payload
        assert len(payload["metric_ledger"]) == 1
        assert "contradiction_rows" in payload
        assert "minority_signals" in payload
        assert "coverage_gaps" in payload
        assert "retention_proof" in payload

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
