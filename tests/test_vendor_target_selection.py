"""Tests for vendor_target selection under mixed owned/global rows."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


def test_dedupe_vendor_target_rows_prefers_owned_row_over_global_contacted_row():
    from atlas_brain.services.vendor_target_selection import dedupe_vendor_target_rows

    rows = [
        {
            "id": "global-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_email": "legacy@example.com",
            "account_id": None,
            "created_at": "2026-03-18T10:00:00+00:00",
            "updated_at": "2026-03-18T10:00:00+00:00",
        },
        {
            "id": "owned-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_email": None,
            "account_id": "acct-1",
            "created_at": "2026-03-19T10:00:00+00:00",
            "updated_at": "2026-03-19T10:00:00+00:00",
        },
    ]

    deduped = dedupe_vendor_target_rows(rows)

    assert len(deduped) == 1
    assert deduped[0]["id"] == "owned-1"


@pytest.mark.asyncio
async def test_fetch_vendor_targets_dedupes_owned_and_global_rows():
    from atlas_brain.autonomous.tasks.b2b_campaign_generation import _fetch_vendor_targets

    pool = AsyncMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "global-1",
                "company_name": "HubSpot",
                "target_mode": "vendor_retention",
                "contact_name": None,
                "contact_email": "legacy@example.com",
                "contact_role": None,
                "products_tracked": None,
                "competitors_tracked": None,
                "tier": "report",
                "status": "active",
                "notes": None,
                "account_id": None,
                "created_at": "2026-03-18T10:00:00+00:00",
                "updated_at": "2026-03-18T10:00:00+00:00",
            },
            {
                "id": "owned-1",
                "company_name": "HubSpot",
                "target_mode": "vendor_retention",
                "contact_name": None,
                "contact_email": None,
                "contact_role": None,
                "products_tracked": None,
                "competitors_tracked": None,
                "tier": "report",
                "status": "active",
                "notes": None,
                "account_id": "acct-1",
                "created_at": "2026-03-19T10:00:00+00:00",
                "updated_at": "2026-03-19T10:00:00+00:00",
            },
        ]
    )

    rows = await _fetch_vendor_targets(pool)

    assert len(rows) == 1
    assert rows[0]["id"] == "owned-1"


@pytest.mark.asyncio
async def test_discover_targets_dedupes_before_batch_limit():
    from atlas_brain.autonomous.tasks.vendor_target_enrichment import _discover_targets

    pool = AsyncMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "global-1",
                "company_name": "HubSpot",
                "target_mode": "vendor_retention",
                "prospect_id": None,
                "contact_email": None,
                "contact_enriched_at": None,
                "account_id": None,
                "created_at": "2026-03-18T10:00:00+00:00",
                "updated_at": "2026-03-18T10:00:00+00:00",
            },
            {
                "id": "owned-1",
                "company_name": "HubSpot",
                "target_mode": "vendor_retention",
                "prospect_id": None,
                "contact_email": None,
                "contact_enriched_at": None,
                "account_id": "acct-1",
                "created_at": "2026-03-19T10:00:00+00:00",
                "updated_at": "2026-03-19T10:00:00+00:00",
            },
            {
                "id": "other-1",
                "company_name": "Linear",
                "target_mode": "vendor_retention",
                "prospect_id": None,
                "contact_email": None,
                "contact_enriched_at": None,
                "account_id": None,
                "created_at": "2026-03-19T11:00:00+00:00",
                "updated_at": "2026-03-19T11:00:00+00:00",
            },
        ]
    )
    cfg = MagicMock(org_cache_days=30, max_vendor_credits_per_run=2)

    rows = await _discover_targets(pool, cfg)

    assert len(rows) == 2
    assert {row["id"] for row in rows} == {"owned-1", "other-1"}


@pytest.mark.asyncio
async def test_resolve_briefing_recipient_orders_owned_targets_first():
    from atlas_brain.autonomous.tasks.b2b_vendor_briefing import resolve_briefing_recipient

    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "email": "owner@example.com",
            "name": "Alice",
            "role": "VP Sales",
            "target_mode": "challenger_intel",
        }
    )

    result = await resolve_briefing_recipient(pool, "HubSpot")

    assert result["email"] == "owner@example.com"
    query = pool.fetchrow.await_args.args[0]
    assert "CASE WHEN account_id IS NULL THEN 1 ELSE 0 END" in query
