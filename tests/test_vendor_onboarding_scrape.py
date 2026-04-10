"""Tests for safe scrape provisioning during vendor onboarding."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


class _Tx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, conn):
        self.is_initialized = True
        self._conn = conn

    def transaction(self):
        return _Tx(self._conn)


async def _provision_result():
    return {
        "status": "applied",
        "requested": 2,
        "applied": 2,
        "matched_vendors": ["HubSpot"],
        "unmatched_vendors": [],
        "bootstrap_used": False,
        "actions": [{"source": "reddit"}, {"source": "g2"}],
    }


@pytest.mark.asyncio
async def test_add_tracked_vendor_auto_provisions_missing_core_targets():
    from atlas_brain.api.b2b_tenant_dashboard import AddVendorRequest, add_tracked_vendor

    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[None, 0, 5])
    conn.fetchrow = AsyncMock(
        side_effect=[
            None,
            {
                "id": "tracked-1",
                "vendor_name": "HubSpot",
                "track_mode": "own",
                "label": "",
                "added_at": "2026-03-19T23:00:00+00:00",
            },
        ]
    )
    pool = _Pool(conn)
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001", product="b2b_retention")

    with patch("atlas_brain.api.b2b_tenant_dashboard.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_tenant_dashboard.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.b2b_tenant_dashboard.upsert_tracked_vendor_source",
                new=AsyncMock(return_value=True),
            ) as upsert_source:
                with patch(
                    "atlas_brain.api.b2b_tenant_dashboard.provision_vendor_onboarding_targets",
                    new=AsyncMock(return_value=await _provision_result()),
                ) as provision:
                    result = await add_tracked_vendor(
                        AddVendorRequest(vendor_name="hubspot", track_mode="own"),
                        user=user,
                    )

    assert result["vendor_name"] == "HubSpot"
    assert result["scrape_provisioning"]["applied"] == 2
    upsert_source.assert_awaited_once_with(
        conn,
        user.account_id,
        "HubSpot",
        source_type="manual",
        source_key="direct",
        track_mode="own",
    )
    provision.assert_awaited_once_with(
        pool,
        "HubSpot",
        product_category=None,
        source_slug_overrides={},
        dry_run=False,
    )


@pytest.mark.asyncio
async def test_add_tracked_vendor_allows_manual_source_for_existing_target_tracked_vendor():
    from atlas_brain.api.b2b_tenant_dashboard import AddVendorRequest, add_tracked_vendor

    existing_row = {
        "id": "tracked-2",
        "vendor_name": "HubSpot",
        "track_mode": "own",
        "label": None,
        "added_at": "2026-03-19T23:00:00+00:00",
    }
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=[existing_row, existing_row])
    conn.fetchval = AsyncMock(return_value=None)
    pool = _Pool(conn)
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001", product="b2b_retention")

    with patch("atlas_brain.api.b2b_tenant_dashboard.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_tenant_dashboard.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.b2b_tenant_dashboard.upsert_tracked_vendor_source",
                new=AsyncMock(return_value=True),
            ) as upsert_source:
                with patch(
                    "atlas_brain.api.b2b_tenant_dashboard.provision_vendor_onboarding_targets",
                    new=AsyncMock(return_value=await _provision_result()),
                ):
                    result = await add_tracked_vendor(
                        AddVendorRequest(vendor_name="hubspot", track_mode="own", label="Core"),
                        user=user,
                    )

    assert result["vendor_name"] == "HubSpot"
    assert conn.execute.await_count == 1
    upsert_source.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_tracked_vendor_purges_all_sources_for_account_vendor():
    from atlas_brain.api.b2b_tenant_dashboard import remove_tracked_vendor

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchval = AsyncMock(return_value=1)
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001", product="b2b_retention")

    with patch("atlas_brain.api.b2b_tenant_dashboard.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.b2b_tenant_dashboard.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.b2b_tenant_dashboard.purge_tracked_vendor_sources",
                new=AsyncMock(
                    return_value={
                        "removed_sources": [
                            {"source_type": "manual", "source_key": "direct"},
                            {"source_type": "vendor_target", "source_key": "target-1"},
                        ],
                        "still_tracked": False,
                    }
                ),
            ) as purge_sources:
                result = await remove_tracked_vendor("hubspot", user=user)

    assert result["vendor_name"] == "HubSpot"
    assert result["still_tracked"] is False
    assert len(result["removed_sources"]) == 2
    purge_sources.assert_awaited_once_with(pool, user.account_id, "HubSpot")


@pytest.mark.asyncio
async def test_create_vendor_target_provisions_company_only_not_free_text_arrays():
    from atlas_brain.api.vendor_targets import VendorTargetCreate, create_vendor_target

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "target-1",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_name": None,
            "contact_email": None,
            "contact_role": None,
            "products_tracked": ["CRM management"],
            "competitors_tracked": ["customer support", "Zendesk"],
            "tier": "report",
            "status": "active",
            "notes": None,
            "account_id": None,
            "created_at": "2026-03-19T23:00:00+00:00",
            "updated_at": "2026-03-19T23:00:00+00:00",
        }
    )

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.vendor_targets.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.vendor_targets.provision_vendor_onboarding_targets",
                new=AsyncMock(return_value=await _provision_result()),
            ) as provision:
                result = await create_vendor_target(
                    VendorTargetCreate(
                        company_name="HubSpot",
                        target_mode="challenger_intel",
                        products_tracked=["CRM management"],
                        competitors_tracked=["customer support", "Zendesk"],
                    ),
                    user=None,
                )

    assert result["company_name"] == "HubSpot"
    assert result["scrape_provisioning"]["matched_vendors"] == ["HubSpot"]
    provision.assert_awaited_once_with(
        pool,
        "HubSpot",
        product_category=None,
        source_slug_overrides={},
        dry_run=False,
    )


@pytest.mark.asyncio
async def test_create_vendor_target_syncs_only_known_competitors_to_tracked_vendors():
    from atlas_brain.api.vendor_targets import VendorTargetCreate, create_vendor_target

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "target-3",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_name": None,
            "contact_email": None,
            "contact_role": None,
            "products_tracked": ["CRM management"],
            "competitors_tracked": ["customer support", "Zendesk"],
            "tier": "report",
            "status": "active",
            "notes": None,
            "account_id": "00000000-0000-0000-0000-000000000001",
            "created_at": "2026-03-19T23:00:00+00:00",
            "updated_at": "2026-03-19T23:00:00+00:00",
        }
    )
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.vendor_targets.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.vendor_targets.resolve_known_vendor_name",
                new=AsyncMock(side_effect=[None, "Zendesk"]),
            ):
                with patch(
                    "atlas_brain.api.vendor_targets.replace_vendor_target_sources",
                    new=AsyncMock(return_value={"synced_vendors": []}),
                ) as replace_sources:
                    with patch(
                        "atlas_brain.api.vendor_targets.provision_vendor_onboarding_targets",
                        new=AsyncMock(return_value=await _provision_result()),
                    ):
                        result = await create_vendor_target(
                            VendorTargetCreate(
                                company_name="HubSpot",
                                target_mode="challenger_intel",
                                products_tracked=["CRM management"],
                                competitors_tracked=["customer support", "Zendesk"],
                            ),
                            user=user,
                        )

    replace_sources.assert_awaited_once_with(
        pool,
        user.account_id,
        "target-3",
        [
            {"vendor_name": "HubSpot", "track_mode": "own"},
            {"vendor_name": "Zendesk", "track_mode": "competitor"},
        ],
    )
    assert result["tracked_vendor_sync"]["skipped_competitors"] == ["customer support"]
    assert result["account_id"] == user.account_id
    assert result["ownership_scope"] == "owned"


@pytest.mark.asyncio
async def test_update_vendor_target_syncs_only_known_competitors_to_tracked_vendors():
    from atlas_brain.api.vendor_targets import VendorTargetUpdate, update_vendor_target

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "target-4",
                "account_id": None,
            },
            {
                "id": "target-4",
                "company_name": "HubSpot",
                "target_mode": "challenger_intel",
                "contact_name": None,
                "contact_email": None,
                "contact_role": None,
                "products_tracked": ["CRM management"],
                "competitors_tracked": ["customer support", "Zendesk"],
                "tier": "report",
                "status": "active",
                "notes": None,
                "account_id": "00000000-0000-0000-0000-000000000001",
                "created_at": "2026-03-19T23:00:00+00:00",
                "updated_at": "2026-03-19T23:05:00+00:00",
            },
        ]
    )
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.vendor_targets.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.vendor_targets.resolve_known_vendor_name",
                new=AsyncMock(side_effect=[None, "Zendesk"]),
            ):
                with patch(
                    "atlas_brain.api.vendor_targets.replace_vendor_target_sources",
                    new=AsyncMock(return_value={"synced_vendors": []}),
                ) as replace_sources:
                    result = await update_vendor_target(
                        "00000000-0000-0000-0000-000000000004",
                        VendorTargetUpdate(competitors_tracked=["customer support", "Zendesk"]),
                        user=user,
                    )

    assert result["tracked_vendor_sync"]["skipped_competitors"] == ["customer support"]
    replace_sources.assert_awaited_once_with(
        pool,
        user.account_id,
        "target-4",
        [
            {"vendor_name": "HubSpot", "track_mode": "own"},
            {"vendor_name": "Zendesk", "track_mode": "competitor"},
        ],
    )
    assert result["account_id"] == user.account_id
    assert result["ownership_scope"] == "owned"


@pytest.mark.asyncio
async def test_delete_vendor_target_cleans_sources_for_all_accounts():
    from atlas_brain.api.vendor_targets import delete_vendor_target

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "target-9",
                "account_id": "00000000-0000-0000-0000-000000000001",
            },
            {"id": "target-9"},
        ]
    )
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch(
            "atlas_brain.api.vendor_targets.delete_vendor_target_sources_for_all_accounts",
            new=AsyncMock(
                return_value={
                    "removed_sources": [
                        {
                            "account_id": "00000000-0000-0000-0000-000000000001",
                            "vendor_name": "HubSpot",
                        }
                    ]
                }
            ),
        ) as delete_sources:
            result = await delete_vendor_target(
                "00000000-0000-0000-0000-000000000009",
                user=user,
            )

    assert result["deleted"] is True
    assert result["tracked_vendor_cleanup"]["removed_sources"][0]["vendor_name"] == "HubSpot"
    delete_sources.assert_awaited_once_with(pool, "00000000-0000-0000-0000-000000000009")


@pytest.mark.asyncio
async def test_claim_vendor_target_claims_legacy_global_row():
    from atlas_brain.api.vendor_targets import claim_vendor_target

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "target-11",
                "company_name": "HubSpot",
                "target_mode": "challenger_intel",
                "contact_name": None,
                "contact_email": "ops@hubspot.com",
                "contact_role": None,
                "products_tracked": None,
                "competitors_tracked": ["Zendesk"],
                "tier": "report",
                "status": "active",
                "notes": None,
                "account_id": None,
                "created_at": "2026-03-19T23:00:00+00:00",
                "updated_at": "2026-03-19T23:00:00+00:00",
            },
            None,
            {
                "id": "target-11",
                "company_name": "HubSpot",
                "target_mode": "challenger_intel",
                "contact_name": None,
                "contact_email": "ops@hubspot.com",
                "contact_role": None,
                "products_tracked": None,
                "competitors_tracked": ["Zendesk"],
                "tier": "report",
                "status": "active",
                "notes": None,
                "account_id": "00000000-0000-0000-0000-000000000001",
                "created_at": "2026-03-19T23:00:00+00:00",
                "updated_at": "2026-03-19T23:05:00+00:00",
            },
        ]
    )
    pool = _Pool(conn)
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.vendor_targets.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.vendor_targets.resolve_known_vendor_name",
                new=AsyncMock(return_value="Zendesk"),
            ):
                with patch(
                    "atlas_brain.api.vendor_targets.replace_vendor_target_sources",
                    new=AsyncMock(return_value={"synced_vendors": []}),
                ) as replace_sources:
                    result = await claim_vendor_target(
                        "00000000-0000-0000-0000-000000000011",
                        user=user,
                    )

    assert result["account_id"] == user.account_id
    assert result["ownership_scope"] == "owned"
    assert result["already_claimed"] is False
    replace_sources.assert_awaited_once_with(
        pool,
        user.account_id,
        "target-11",
        [
            {"vendor_name": "HubSpot", "track_mode": "own"},
            {"vendor_name": "Zendesk", "track_mode": "competitor"},
        ],
    )


@pytest.mark.asyncio
async def test_claim_vendor_target_is_idempotent_for_owned_row():
    from atlas_brain.api.vendor_targets import claim_vendor_target

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "id": "target-12",
            "company_name": "HubSpot",
            "target_mode": "challenger_intel",
            "contact_name": None,
            "contact_email": "ops@hubspot.com",
            "contact_role": None,
            "products_tracked": None,
            "competitors_tracked": ["Zendesk"],
            "tier": "report",
            "status": "active",
            "notes": None,
            "account_id": "00000000-0000-0000-0000-000000000001",
            "created_at": "2026-03-19T23:00:00+00:00",
            "updated_at": "2026-03-19T23:05:00+00:00",
        }
    )
    pool = _Pool(conn)
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.vendor_targets.resolve_vendor_name", new=AsyncMock(return_value="HubSpot")):
            with patch(
                "atlas_brain.api.vendor_targets.resolve_known_vendor_name",
                new=AsyncMock(return_value="Zendesk"),
            ):
                with patch(
                    "atlas_brain.api.vendor_targets.replace_vendor_target_sources",
                    new=AsyncMock(return_value={"synced_vendors": []}),
                ) as replace_sources:
                    result = await claim_vendor_target(
                        "00000000-0000-0000-0000-000000000012",
                        user=user,
                    )

    assert result["already_claimed"] is True
    replace_sources.assert_awaited_once()
    assert conn.fetchrow.await_count == 1


@pytest.mark.asyncio
async def test_claim_vendor_target_rejects_duplicate_owned_row():
    from atlas_brain.api.vendor_targets import claim_vendor_target

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "target-13",
                "company_name": "HubSpot",
                "target_mode": "challenger_intel",
                "contact_name": None,
                "contact_email": "ops@hubspot.com",
                "contact_role": None,
                "products_tracked": None,
                "competitors_tracked": ["Zendesk"],
                "tier": "report",
                "status": "active",
                "notes": None,
                "account_id": None,
                "created_at": "2026-03-19T23:00:00+00:00",
                "updated_at": "2026-03-19T23:00:00+00:00",
            },
            {"id": "target-owned"},
        ]
    )
    pool = _Pool(conn)
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with pytest.raises(HTTPException) as exc:
            await claim_vendor_target(
                "00000000-0000-0000-0000-000000000013",
                user=user,
            )

    assert exc.value.status_code == 409
    assert "duplicate" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_delete_vendor_target_rejects_legacy_global_row_for_authenticated_user():
    from atlas_brain.api.vendor_targets import delete_vendor_target

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(return_value={"id": "target-10", "account_id": None})
    user = SimpleNamespace(account_id="00000000-0000-0000-0000-000000000001")

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with pytest.raises(HTTPException) as exc:
            await delete_vendor_target(
                "00000000-0000-0000-0000-000000000010",
                user=user,
            )

    assert exc.value.status_code == 409
    assert "claimed" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_create_vendor_target_manual_bootstrap_uses_category_and_slug_overrides():
    from atlas_brain.api.vendor_targets import VendorTargetCreate, create_vendor_target

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "target-2",
            "company_name": "Linear",
            "target_mode": "vendor_retention",
            "contact_name": None,
            "contact_email": None,
            "contact_role": None,
            "products_tracked": None,
            "competitors_tracked": None,
            "tier": "report",
            "status": "active",
            "notes": None,
            "account_id": None,
            "created_at": "2026-03-19T23:00:00+00:00",
            "updated_at": "2026-03-19T23:00:00+00:00",
        }
    )
    provision_result = {
        "status": "applied",
        "requested": 3,
        "applied": 3,
        "matched_vendors": ["Linear"],
        "unmatched_vendors": [],
        "bootstrap_used": True,
        "actions": [{"source": "reddit"}, {"source": "g2"}, {"source": "capterra"}],
    }

    with patch("atlas_brain.api.vendor_targets.get_db_pool", return_value=pool):
        with patch("atlas_brain.api.vendor_targets.resolve_vendor_name", new=AsyncMock(return_value="Linear")):
            with patch(
                "atlas_brain.api.vendor_targets.provision_vendor_onboarding_targets",
                new=AsyncMock(return_value=provision_result),
            ) as provision:
                result = await create_vendor_target(
                    VendorTargetCreate(
                        company_name="Linear",
                        target_mode="vendor_retention",
                        product_category="Project Management",
                        scrape_target_slugs={
                            "g2": "linear",
                            "capterra": "12345/Linear-PM",
                        },
                    ),
                    user=None,
                )

    assert result["scrape_provisioning"]["bootstrap_used"] is True
    provision.assert_awaited_once_with(
        pool,
        "Linear",
        product_category="Project Management",
        source_slug_overrides={
            "g2": "linear",
            "capterra": "12345/Linear-PM",
        },
        dry_run=False,
    )


@pytest.mark.asyncio
async def test_provision_vendor_onboarding_targets_bootstraps_search_and_slug_sources():
    from atlas_brain.services.scraping.target_provisioning import (
        provision_vendor_onboarding_targets,
    )

    pool = AsyncMock()
    pool.fetch = AsyncMock(side_effect=[[], [], []])
    pool.fetchrow = AsyncMock(side_effect=[{"id": "t-1"}, {"id": "t-2"}, {"id": "t-3"}])
    cfg = MagicMock()
    cfg.source_allowlist = "reddit,g2,capterra"

    with patch(
        "atlas_brain.services.scraping.target_provisioning.settings",
        MagicMock(b2b_scrape=cfg),
    ):
        with patch(
            "atlas_brain.services.scraping.target_provisioning.resolve_vendor_name",
            new=AsyncMock(return_value="Linear"),
        ):
            result = await provision_vendor_onboarding_targets(
                pool,
                "Linear",
                product_category="Project Management",
                source_slug_overrides={
                    "g2": "linear",
                    "capterra": "12345/Linear-PM",
                },
                dry_run=False,
            )

    assert result["status"] == "applied"
    assert result["bootstrap_used"] is True
    assert result["matched_vendors"] == ["Linear"]
    assert result["applied"] == 3
    assert result["applied_core_targets"] == 2
    assert result["applied_signal_targets"] == 1
    assert [item["source"] for item in result["actions"]] == ["capterra", "g2", "reddit"]
    assert pool.fetchrow.await_count == 3


@pytest.mark.asyncio
async def test_provision_vendor_onboarding_targets_seeds_conditional_signal_lane_when_core_is_thin():
    from atlas_brain.services.scraping.target_provisioning import (
        provision_vendor_onboarding_targets,
    )

    pool = AsyncMock()
    pool.fetch = AsyncMock(side_effect=[[], [], []])
    pool.fetchrow = AsyncMock(return_value={"id": "t-signal"})
    cfg = MagicMock()
    cfg.source_allowlist = "github"
    cfg.deprecated_sources = ""

    with patch(
        "atlas_brain.services.scraping.target_provisioning.settings",
        MagicMock(b2b_scrape=cfg),
    ):
        with patch(
            "atlas_brain.services.scraping.target_provisioning.resolve_vendor_name",
            new=AsyncMock(return_value="HubSpot"),
        ):
            result = await provision_vendor_onboarding_targets(
                pool,
                "HubSpot",
                product_category="CRM",
                dry_run=False,
            )

    assert result["status"] == "applied"
    assert result["bootstrap_used"] is True
    assert result["applied_core_targets"] == 0
    assert result["applied_signal_targets"] == 1
    assert result["actions"][0]["source"] == "github"
    assert result["actions"][0]["metadata"]["source_fit_probation"] is True
