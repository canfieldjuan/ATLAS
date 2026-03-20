from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from atlas_brain.services.vendor_target_backfill import (
    CLAIM_REASON_COMPETITOR_OVERLAP,
    CLAIM_REASON_DIRECT_SOURCE,
    CLAIM_REASON_EXACT_COMPETITOR,
    CLAIM_REASON_EXACT_OWN,
    apply_legacy_vendor_target_account_backfill,
    plan_legacy_vendor_target_account_backfill,
    _select_candidates,
)


class _Tx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def transaction(self):
        return _Tx(self._conn)


def test_select_candidates_prefers_direct_source_matches():
    legacy_rows = [
        {
            "target_id": "target-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_email": "ops@hubspot.com",
        }
    ]
    direct_rows = [
        {
            "target_id": "target-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_email": "ops@hubspot.com",
            "account_id": "acct-1",
            "tracked_vendor_name": "HubSpot",
            "track_mode": "competitor",
        }
    ]
    competitor_rows = [
        {
            "target_id": "target-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_email": "ops@hubspot.com",
            "account_id": "acct-2",
            "tracked_vendor_name": "HubSpot",
            "track_mode": "competitor",
        }
    ]

    candidates = _select_candidates(
        legacy_rows,
        direct_rows,
        own_rows=[],
        competitor_rows=competitor_rows,
        overlap_rows=[],
    )

    assert len(candidates) == 1
    assert candidates[0].account_id == "acct-1"
    assert candidates[0].claim_reason == CLAIM_REASON_DIRECT_SOURCE
    assert candidates[0].track_mode == "competitor"


@pytest.mark.asyncio
async def test_plan_reports_exact_mode_matches():
    pool = AsyncMock()
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "target_id": "target-1",
                    "company_name": "HubSpot",
                    "target_mode": "challenger_intel",
                    "contact_email": "ops@hubspot.com",
                },
                {
                    "target_id": "target-2",
                    "company_name": "Salesforce",
                    "target_mode": "vendor_retention",
                    "contact_email": "ops@salesforce.com",
                },
            ],
            [],
            [
                {
                    "target_id": "target-2",
                    "company_name": "Salesforce",
                    "target_mode": "vendor_retention",
                    "contact_email": "ops@salesforce.com",
                    "account_id": "acct-own",
                    "tracked_vendor_name": "Salesforce",
                    "track_mode": "own",
                }
            ],
            [
                {
                    "target_id": "target-1",
                    "company_name": "HubSpot",
                    "target_mode": "challenger_intel",
                    "contact_email": "ops@hubspot.com",
                    "account_id": "acct-comp",
                    "tracked_vendor_name": "HubSpot",
                    "track_mode": "competitor",
                }
            ],
            [],
        ]
    )

    result = await plan_legacy_vendor_target_account_backfill(pool)

    assert result["legacy_targets"] == 2
    assert result["exact_own_matches"] == 1
    assert result["exact_competitor_matches"] == 1
    assert result["claimable_targets_total"] == 2
    assert {
        candidate["claim_reason"] for candidate in result["candidates"]
    } == {CLAIM_REASON_EXACT_OWN, CLAIM_REASON_EXACT_COMPETITOR}


@pytest.mark.asyncio
async def test_plan_reports_overlap_matches_when_exact_match_missing():
    pool = AsyncMock()
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "target_id": "target-3",
                    "company_name": "Discord",
                    "target_mode": "challenger_intel",
                    "contact_email": "ops@discord.com",
                }
            ],
            [],
            [],
            [],
            [
                {
                    "target_id": "target-3",
                    "company_name": "Discord",
                    "target_mode": "challenger_intel",
                    "contact_email": "ops@discord.com",
                    "account_id": "acct-overlap",
                    "tracked_vendor_name": "Slack",
                    "tracked_vendor_names": ["Slack", "Zoom"],
                    "track_mode": "competitor",
                }
            ],
        ]
    )

    result = await plan_legacy_vendor_target_account_backfill(pool)

    assert result["challenger_overlap_matches"] == 1
    assert result["claimable_targets_total"] == 1
    assert result["candidates"][0]["claim_reason"] == CLAIM_REASON_COMPETITOR_OVERLAP
    assert result["candidates"][0]["tracked_vendor_names"] == ["Slack", "Zoom"]


@pytest.mark.asyncio
async def test_apply_claims_targets_and_upserts_vendor_target_source():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=[{"id": "target-1"}])
    pool = _Pool(conn)

    plan_result = {
        "legacy_targets": 1,
        "claimable_targets_total": 1,
        "claimable_targets_selected": 1,
        "candidates": [
            {
                "target_id": "target-1",
                "company_name": "HubSpot",
                "target_mode": "challenger_intel",
                "contact_email": "ops@hubspot.com",
                "account_id": "acct-1",
                "tracked_vendor_name": "HubSpot",
                "tracked_vendor_names": ["HubSpot"],
                "track_mode": "competitor",
                "claim_reason": CLAIM_REASON_EXACT_COMPETITOR,
            }
        ],
    }

    with patch(
        "atlas_brain.services.vendor_target_backfill.plan_legacy_vendor_target_account_backfill",
        new=AsyncMock(return_value=plan_result),
    ):
        with patch(
            "atlas_brain.services.vendor_target_backfill.upsert_tracked_vendor_source",
            new=AsyncMock(return_value=True),
        ) as upsert_source:
            result = await apply_legacy_vendor_target_account_backfill(pool)

    assert result["applied"] == 1
    conn.fetchrow.assert_awaited_once()
    upsert_source.assert_awaited_once_with(
        conn,
        "acct-1",
        "HubSpot",
        source_type="vendor_target",
        source_key="target-1",
        track_mode="competitor",
    )


@pytest.mark.asyncio
async def test_apply_overlap_claim_upserts_each_overlapping_vendor():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=[{"id": "target-4"}])
    pool = _Pool(conn)

    plan_result = {
        "legacy_targets": 1,
        "claimable_targets_total": 1,
        "claimable_targets_selected": 1,
        "candidates": [
            {
                "target_id": "target-4",
                "company_name": "Discord",
                "target_mode": "challenger_intel",
                "contact_email": "ops@discord.com",
                "account_id": "acct-2",
                "tracked_vendor_name": "Slack",
                "tracked_vendor_names": ["Slack", "Zoom"],
                "track_mode": "competitor",
                "claim_reason": CLAIM_REASON_COMPETITOR_OVERLAP,
            }
        ],
    }

    with patch(
        "atlas_brain.services.vendor_target_backfill.plan_legacy_vendor_target_account_backfill",
        new=AsyncMock(return_value=plan_result),
    ):
        with patch(
            "atlas_brain.services.vendor_target_backfill.upsert_tracked_vendor_source",
            new=AsyncMock(return_value=True),
        ) as upsert_source:
            result = await apply_legacy_vendor_target_account_backfill(pool)

    assert result["applied"] == 1
    assert upsert_source.await_count == 2
    upsert_source.assert_any_await(
        conn,
        "acct-2",
        "Slack",
        source_type="vendor_target",
        source_key="target-4",
        track_mode="competitor",
    )
    upsert_source.assert_any_await(
        conn,
        "acct-2",
        "Zoom",
        source_type="vendor_target",
        source_key="target-4",
        track_mode="competitor",
    )


@pytest.mark.asyncio
async def test_apply_dry_run_leaves_database_untouched():
    pool = SimpleNamespace()
    plan_result = {
        "legacy_targets": 1,
        "claimable_targets_total": 1,
        "claimable_targets_selected": 0,
        "candidates": [],
    }

    with patch(
        "atlas_brain.services.vendor_target_backfill.plan_legacy_vendor_target_account_backfill",
        new=AsyncMock(return_value=plan_result),
    ):
        result = await apply_legacy_vendor_target_account_backfill(pool, limit=0)

    assert result["applied"] == 0
    assert result["already_claimed"] == 0
