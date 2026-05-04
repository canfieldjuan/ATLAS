from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from extracted_competitive_intelligence.autonomous.tasks._b2b_cross_vendor_synthesis import (
    attach_cross_vendor_citation_registry,
    build_cross_vendor_conclusions_for_vendor,
    build_pairwise_battle_packet,
    compute_cross_vendor_evidence_hash,
    load_cross_vendor_synthesis_lookup,
    materialize_cross_vendor_reference_ids,
    normalize_cross_vendor_contract,
    to_legacy_cross_vendor_conclusion,
)


POOL_LAYERS = {
    "Zendesk": {
        "core": {
            "total_reviews": 120,
            "avg_urgency_score": 7.2,
            "churn_signal_density": 22.1,
            "price_complaint_rate": 0.28,
            "top_competitors": [{"name": "Freshdesk"}],
        },
        "pain_distribution": [{"category": "pricing", "count": 30}],
        "budget_pressure": {"price_increase_rate": 0.03, "avg_seat_count": 50},
    },
    "Freshdesk": {
        "core": {
            "total_reviews": 80,
            "avg_urgency_score": 5.5,
            "churn_signal_density": 15.2,
            "price_complaint_rate": 0.24,
            "top_competitors": [{"name": "Zendesk"}],
        },
        "pain_distribution": [{"category": "support", "count": 20}],
        "budget_pressure": {"price_increase_rate": 0.03, "avg_seat_count": 35},
    },
}

PROFILES = {
    "Zendesk": {
        "product_category": "Helpdesk",
        "strengths": ["Enterprise features"],
        "weaknesses": ["Complex pricing"],
        "primary_use_cases": ["Ticketing"],
        "typical_company_size": "51-200",
    },
    "Freshdesk": {
        "product_category": "Helpdesk",
        "strengths": ["Easy setup"],
        "weaknesses": ["Limited reporting"],
        "primary_use_cases": ["Support"],
        "typical_company_size": "11-50",
    },
}

VENDOR_REFERENCE_LOOKUP = {
    "Zendesk": {
        "metric_ids": ["metric:zendesk:1"],
        "witness_ids": ["witness:zendesk:1"],
    },
    "Freshdesk": {
        "metric_ids": ["metric:freshdesk:1"],
        "witness_ids": ["witness:freshdesk:1"],
    },
}

EDGE = {
    "from_vendor": "Zendesk",
    "to_vendor": "Freshdesk",
    "mention_count": 12,
    "signal_strength": "strong",
    "primary_driver": "pricing",
    "evidence_breakdown": {"explicit_switch": 2},
    "velocity_7d": 3,
}


class FakePool:
    def __init__(self, synthesis_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]] | None = None) -> None:
        self.synthesis_rows = synthesis_rows
        self.edge_rows = edge_rows or []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "FROM b2b_displacement_edges" in query:
            return self.edge_rows
        if "FROM b2b_cross_vendor_reasoning_synthesis" in query:
            return self.synthesis_rows
        raise AssertionError(f"Unexpected query: {query}")


def test_pairwise_packet_preserves_locked_direction_and_vendor_pools() -> None:
    packet = build_pairwise_battle_packet(
        "Zendesk",
        "Freshdesk",
        EDGE,
        POOL_LAYERS,
        PROFILES,
    )

    assert packet["locked_direction"] == {"winner": "Freshdesk", "loser": "Zendesk"}
    assert packet["vendor_a_pool"]["total_reviews"] == 120
    assert packet["vendor_b_profile"]["strengths"] == ["Easy setup"]


def test_cross_vendor_hash_is_deterministic_and_input_sensitive() -> None:
    packet = build_pairwise_battle_packet(
        "Zendesk",
        "Freshdesk",
        EDGE,
        POOL_LAYERS,
        PROFILES,
    )
    changed_packet = build_pairwise_battle_packet(
        "Zendesk",
        "Freshdesk",
        {**EDGE, "mention_count": 99},
        POOL_LAYERS,
        PROFILES,
    )

    assert compute_cross_vendor_evidence_hash(packet) == compute_cross_vendor_evidence_hash(packet)
    assert compute_cross_vendor_evidence_hash(packet) != compute_cross_vendor_evidence_hash(changed_packet)


def test_materialize_reference_ids_maps_cited_registry_entries() -> None:
    packet = build_pairwise_battle_packet(
        "Zendesk",
        "Freshdesk",
        EDGE,
        POOL_LAYERS,
        PROFILES,
    )
    packet = attach_cross_vendor_citation_registry(
        packet,
        analysis_type="pairwise_battle",
        vendors=["Zendesk", "Freshdesk"],
        category=None,
        vendor_reference_lookup=VENDOR_REFERENCE_LOOKUP,
    )
    synthesis = {
        "citations": [
            "xv:vendor:zendesk:pool",
            "Freshdesk pool summary: total_reviews=80, avg_urgency=5.5",
            "not:a:real:packet:id",
        ],
    }

    result = materialize_cross_vendor_reference_ids(synthesis, packet)

    assert result["citations"] == [
        "xv:vendor:zendesk:pool",
        "xv:vendor:freshdesk:pool",
    ]
    assert result["reference_ids"]["metric_ids"] == [
        "metric:freshdesk:1",
        "metric:zendesk:1",
    ]


@pytest.mark.asyncio
async def test_load_cross_vendor_synthesis_lookup_preserves_reference_ids() -> None:
    pool = FakePool([
        {
            "analysis_type": "pairwise_battle",
            "vendors": ["Zendesk", "Freshdesk"],
            "category": None,
            "synthesis": {
                "winner": "Freshdesk",
                "loser": "Zendesk",
                "conclusion": "Freshdesk wins on pricing pressure.",
                "confidence": 0.82,
                "reference_ids": {
                    "metric_ids": ["metric:pair:1"],
                    "witness_ids": ["witness:pair:1"],
                },
            },
            "as_of_date": date(2026, 3, 30),
            "created_at": "2026-03-31T00:00:00Z",
        }
    ])

    lookup = await load_cross_vendor_synthesis_lookup(
        pool,
        as_of=date(2026, 3, 30),
        analysis_window_days=30,
    )

    entry = lookup["battles"][("Freshdesk", "Zendesk")]
    assert entry["conclusion"]["winner"] == "Freshdesk"
    assert entry["reference_ids"]["witness_ids"] == ["witness:pair:1"]


@pytest.mark.asyncio
async def test_load_cross_vendor_synthesis_lookup_backfills_pairwise_refs() -> None:
    pool = FakePool(
        [
            {
                "analysis_type": "pairwise_battle",
                "vendors": ["HubSpot", "Salesforce"],
                "category": None,
                "synthesis": json.dumps({
                    "winner": "HubSpot",
                    "loser": "Salesforce",
                    "conclusion": "HubSpot is gaining share.",
                    "confidence": 0.81,
                }),
                "as_of_date": date(2026, 3, 30),
                "created_at": "2026-03-31T04:26:40Z",
            },
        ],
        edge_rows=[
            {
                "from_vendor": "Salesforce",
                "to_vendor": "HubSpot",
                "sample_review_ids": ["review-1", "review-2", "review-2"],
                "computed_date": date(2026, 3, 30),
                "created_at": "2026-03-31T03:00:31Z",
            },
        ],
    )

    lookup = await load_cross_vendor_synthesis_lookup(
        pool,
        as_of=date(2026, 3, 31),
        analysis_window_days=30,
    )

    entry = lookup["battles"][("HubSpot", "Salesforce")]
    assert entry["reference_ids"]["witness_ids"] == ["review-1", "review-2"]


def test_contract_normalization_and_legacy_mirror_shape() -> None:
    normalized = normalize_cross_vendor_contract(
        {"winner": "Freshdesk", "loser": "Zendesk", "confidence": 5.0},
        "pairwise_battle",
    )
    legacy = to_legacy_cross_vendor_conclusion(
        normalized,
        "pairwise_battle",
        ["Freshdesk", "Zendesk"],
        evidence_hash="abc123",
        tokens_used=500,
    )

    assert normalized["confidence"] == 1.0
    assert normalized["durability_assessment"] == "uncertain"
    assert legacy["conclusion"]["winner"] == "Freshdesk"
    assert legacy["evidence_hash"] == "abc123"


def test_vendor_facing_conclusions_include_category_refs() -> None:
    results = build_cross_vendor_conclusions_for_vendor(
        "Zendesk",
        category="CRM",
        xv_lookup={
            "battles": {
                ("HubSpot", "Zendesk"): {
                    "confidence": 0.74,
                    "source": "synthesis",
                    "computed_date": date(2026, 3, 31),
                    "conclusion": {"conclusion": "HubSpot is winning SMB deals."},
                },
            },
            "councils": {
                "CRM": {
                    "confidence": 0.62,
                    "source": "synthesis",
                    "computed_date": date(2026, 3, 31),
                    "reference_ids": {
                        "metric_ids": ["metric:crm:1"],
                        "witness_ids": ["witness:crm:1"],
                    },
                    "conclusion": {
                        "conclusion": "Pricing pressure is fragmenting CRM.",
                        "market_regime": "price_competition",
                    },
                    "vendors": ["HubSpot", "Zendesk"],
                },
            },
            "asymmetries": {},
        },
        limit=5,
    )

    assert results[0]["analysis_type"] == "pairwise_battle"
    assert results[1]["analysis_type"] == "category_council"
    assert results[1]["reference_ids"]["witness_ids"] == ["witness:crm:1"]
