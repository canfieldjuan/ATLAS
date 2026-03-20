"""Tests for challenger_target_discovery global-row ownership boundaries."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


@pytest.mark.asyncio
async def test_challenger_target_discovery_only_matches_global_targets():
    from atlas_brain.autonomous.tasks.challenger_target_discovery import run

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "to_vendor": "HubSpot",
                    "losing_vendors": ["Salesforce"],
                    "total_mentions": 5,
                }
            ],
            [],
            [],
        ]
    )
    pool.fetchrow = AsyncMock(return_value=None)
    task = MagicMock()

    with patch(
        "atlas_brain.autonomous.tasks.challenger_target_discovery.get_db_pool",
        return_value=pool,
    ):
        result = await run(task)

    assert result["created"] == 1
    existing_query = pool.fetchrow.await_args_list[0].args[0]
    assert "account_id IS NULL" in existing_query


@pytest.mark.asyncio
async def test_challenger_target_discovery_updates_existing_global_target_only():
    from atlas_brain.autonomous.tasks.challenger_target_discovery import run

    pool = AsyncMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "to_vendor": "HubSpot",
                    "losing_vendors": ["Salesforce", "Zendesk"],
                    "total_mentions": 7,
                }
            ],
            [],
            [],
        ]
    )
    pool.fetchrow = AsyncMock(
        side_effect=[
            {"id": "global-target-1", "competitors_tracked": ["Salesforce"]},
            None,
        ]
    )
    task = MagicMock()

    with patch(
        "atlas_brain.autonomous.tasks.challenger_target_discovery.get_db_pool",
        return_value=pool,
    ):
        result = await run(task)

    assert result["updated"] == 1
    update_call = pool.execute.await_args
    assert "UPDATE vendor_targets" in update_call.args[0]
    assert update_call.args[3] == "global-target-1"
