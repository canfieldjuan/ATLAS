"""Tests for cross-vendor synthesis reader and battle card consumer cutover."""

import json
from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis import (
    build_cross_vendor_conclusions_for_vendor,
    load_cross_vendor_synthesis_lookup,
    load_best_cross_vendor_lookup,
    merge_cross_vendor_lookups,
)


def _make_row(
    analysis_type: str,
    vendors: list[str],
    category: str | None,
    synthesis: dict[str, Any],
    as_of: str = "2026-03-29",
) -> dict[str, Any]:
    return {
        "analysis_type": analysis_type,
        "vendors": vendors,
        "category": category,
        "synthesis": json.dumps(synthesis),
        "as_of_date": date.fromisoformat(as_of),
        "created_at": f"{as_of}T00:00:00+00:00",
    }


class FakePool:
    """Minimal pool mock that returns preset rows."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    async def fetch(self, query: str, *args) -> list[dict]:
        return self._rows


# ---------------------------------------------------------------------------
# Synthesis reader unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_reader_returns_battle():
    pool = FakePool([
        _make_row(
            "pairwise_battle",
            ["Salesforce", "HubSpot"],
            None,
            {
                "conclusion": {
                    "winner": "HubSpot",
                    "loser": "Salesforce",
                    "conclusion": "HubSpot wins on SMB usability.",
                    "confidence": 0.75,
                    "durability_assessment": "stable",
                    "key_insights": [{"insight": "Price gap", "evidence": "review data"}],
                },
            },
        ),
    ])
    lookup = await load_cross_vendor_synthesis_lookup(pool, as_of=date(2026, 3, 29))
    assert len(lookup["battles"]) == 1
    key = ("HubSpot", "Salesforce")
    entry = lookup["battles"][key]
    assert entry["confidence"] == 0.75
    assert entry["source"] == "synthesis"
    assert entry["conclusion"]["winner"] == "HubSpot"


@pytest.mark.asyncio
async def test_synthesis_reader_returns_council():
    pool = FakePool([
        _make_row(
            "category_council",
            ["Salesforce", "HubSpot", "Zoho"],
            "CRM",
            {
                "conclusion": {
                    "market_regime": "consolidating",
                    "conclusion": "Salesforce dominates but faces SMB churn.",
                    "winner": "Salesforce",
                    "confidence": 0.8,
                    "key_insights": [],
                },
            },
        ),
    ])
    lookup = await load_cross_vendor_synthesis_lookup(pool, as_of=date(2026, 3, 29))
    assert len(lookup["councils"]) == 1
    entry = lookup["councils"]["CRM"]
    assert entry["confidence"] == 0.8
    assert entry["source"] == "synthesis"
    assert entry["conclusion"]["market_regime"] == "consolidating"


@pytest.mark.asyncio
async def test_synthesis_reader_returns_asymmetry():
    pool = FakePool([
        _make_row(
            "resource_asymmetry",
            ["Jira", "Trello"],
            None,
            {
                "conclusion": {
                    "favored_vendor": "Jira",
                    "conclusion": "Jira has 3x the review volume.",
                    "confidence": 0.6,
                },
            },
        ),
    ])
    lookup = await load_cross_vendor_synthesis_lookup(pool, as_of=date(2026, 3, 29))
    assert len(lookup["asymmetries"]) == 1
    key = ("Jira", "Trello")
    entry = lookup["asymmetries"][key]
    assert entry["confidence"] == 0.6
    assert entry["source"] == "synthesis"


@pytest.mark.asyncio
async def test_synthesis_reader_empty_table():
    pool = FakePool([])
    lookup = await load_cross_vendor_synthesis_lookup(pool, as_of=date(2026, 3, 29))
    assert lookup == {"battles": {}, "councils": {}, "asymmetries": {}}


@pytest.mark.asyncio
async def test_synthesis_reader_higher_confidence_wins():
    pool = FakePool([
        _make_row(
            "pairwise_battle",
            ["A", "B"],
            None,
            {"conclusion": {"conclusion": "Old", "confidence": 0.5}},
            as_of="2026-03-28",
        ),
        _make_row(
            "pairwise_battle",
            ["A", "B"],
            None,
            {"conclusion": {"conclusion": "New", "confidence": 0.9}},
            as_of="2026-03-29",
        ),
    ])
    lookup = await load_cross_vendor_synthesis_lookup(pool, as_of=date(2026, 3, 29))
    # DISTINCT ON picks the latest as_of_date first, so only one row per key
    assert len(lookup["battles"]) == 1
    entry = lookup["battles"][("A", "B")]
    assert entry["confidence"] == 0.9


def test_merge_cross_vendor_lookups_prefers_primary_and_fills_gaps():
    merged, overrides = merge_cross_vendor_lookups(
        primary={
            "battles": {
                ("A", "B"): {"source": "synthesis", "conclusion": {"conclusion": "Primary"}},
            },
            "councils": {},
            "asymmetries": {},
        },
        fallback={
            "battles": {
                ("A", "B"): {"source": "legacy", "conclusion": {"conclusion": "Fallback"}},
                ("A", "C"): {"source": "legacy", "conclusion": {"conclusion": "Gap fill"}},
            },
            "councils": {},
            "asymmetries": {},
        },
    )

    assert overrides == 1
    assert merged["battles"][("A", "B")]["source"] == "synthesis"
    assert merged["battles"][("A", "C")]["source"] == "legacy"


@pytest.mark.asyncio
async def test_load_best_cross_vendor_lookup_is_synthesis_only_by_default(monkeypatch):
    async def _fake_reconstruct(pool, as_of=None):
        return {
            "battles": {
                ("A", "B"): {"source": "legacy", "conclusion": {"conclusion": "Legacy battle"}},
            },
            "councils": {
                "CRM": {"source": "legacy", "conclusion": {"conclusion": "Legacy council"}},
            },
            "asymmetries": {},
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_churn_intelligence.reconstruct_cross_vendor_lookup",
        _fake_reconstruct,
    )
    pool = FakePool([
        _make_row(
            "pairwise_battle",
            ["A", "B"],
            None,
            {
                "conclusion": {
                    "conclusion": "Synthesis battle",
                    "confidence": 0.8,
                },
            },
        ),
    ])

    lookup = await load_best_cross_vendor_lookup(pool, as_of=date(2026, 3, 29))

    assert lookup["battles"][("A", "B")]["conclusion"]["conclusion"] == "Synthesis battle"
    assert lookup["councils"] == {}


@pytest.mark.asyncio
async def test_load_best_cross_vendor_lookup_merges_legacy_when_opted_in(monkeypatch):
    async def _fake_reconstruct(pool, as_of=None):
        return {
            "battles": {
                ("A", "B"): {"source": "legacy", "conclusion": {"conclusion": "Legacy battle"}},
            },
            "councils": {
                "CRM": {"source": "legacy", "conclusion": {"conclusion": "Legacy council"}},
            },
            "asymmetries": {},
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_churn_intelligence.reconstruct_cross_vendor_lookup",
        _fake_reconstruct,
    )
    pool = FakePool([
        _make_row(
            "pairwise_battle",
            ["A", "B"],
            None,
            {
                "conclusion": {
                    "conclusion": "Synthesis battle",
                    "confidence": 0.8,
                },
            },
        ),
    ])

    lookup = await load_best_cross_vendor_lookup(
        pool,
        as_of=date(2026, 3, 29),
        allow_legacy_fallback=True,
    )

    assert lookup["battles"][("A", "B")]["conclusion"]["conclusion"] == "Synthesis battle"
    assert lookup["councils"]["CRM"]["conclusion"]["conclusion"] == "Legacy council"


def test_build_cross_vendor_conclusions_for_vendor_includes_council_refs():
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
                    "reference_ids": {"metric_ids": ["metric:crm:1"], "witness_ids": ["witness:crm:1"]},
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
