"""Tests for cross-vendor synthesis reader and battle card consumer cutover."""

import json
from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis import (
    load_cross_vendor_synthesis_lookup,
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
