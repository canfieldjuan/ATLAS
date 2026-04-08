from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import atlas_brain.autonomous.tasks.b2b_watchlist_alert_delivery as mod
from atlas_brain.services.b2b import watchlist_alerts as watchlist_alert_service


@pytest.mark.asyncio
async def test_run_skips_when_watchlist_delivery_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_watchlist_delivery, "enabled", False, raising=False)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {"_skip_synthesis": "Watchlist alert delivery disabled"}


@pytest.mark.asyncio
async def test_run_processes_due_view_and_advances_schedule(monkeypatch):
    now = datetime.now(timezone.utc)
    scheduled_for = now - timedelta(minutes=5)
    account_id = uuid4()
    view_id = uuid4()
    claim_id = uuid4()

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[
            {
                "id": view_id,
                "account_id": account_id,
                "name": "Daily CRM watch",
                "vendor_name": "Intercom",
                "category": "Helpdesk",
                "source": "reddit",
                "min_urgency": 8.0,
                "include_stale": False,
                "named_accounts_only": True,
                "changed_wedges_only": False,
                "vendor_alert_threshold": 7.5,
                "account_alert_threshold": 8.5,
                "stale_days_threshold": 2,
                "alert_email_enabled": True,
                "alert_delivery_frequency": "daily",
                "next_alert_delivery_at": scheduled_for,
                "last_alert_delivery_at": None,
                "last_alert_delivery_status": None,
                "last_alert_delivery_summary": None,
                "created_at": now,
                "updated_at": now,
                "account_name": "Acme",
                "product": "b2b_retention",
                "plan_status": "active",
            }
        ]),
        fetchrow=AsyncMock(return_value={"id": claim_id}),
        execute=AsyncMock(return_value="UPDATE 1"),
    )

    monkeypatch.setattr(mod.settings.b2b_watchlist_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_watchlist_delivery, "max_views_per_run", 10, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: object())
    monkeypatch.setattr(watchlist_alert_service, "watchlist_alert_sender_configured", lambda: True)
    monkeypatch.setattr(
        watchlist_alert_service,
        "evaluate_watchlist_alert_events_for_view",
        AsyncMock(return_value={
            "watchlist_view_id": str(view_id),
            "watchlist_view_name": "Daily CRM watch",
            "evaluated_at": now.isoformat(),
            "events": [
                {"id": str(uuid4()), "summary": "Intercom crossed the vendor alert threshold at 8.1"},
            ],
            "count": 1,
            "new_open_event_count": 1,
            "resolved_event_count": 0,
        }),
    )
    monkeypatch.setattr(
        watchlist_alert_service,
        "resolve_watchlist_alert_recipients",
        AsyncMock(return_value=[{"email": "owner@example.com", "full_name": "Owner"}]),
    )
    monkeypatch.setattr(watchlist_alert_service, "watchlist_alert_from_email", lambda: "Atlas <atlas@example.com>")
    monkeypatch.setattr(
        watchlist_alert_service,
        "send_watchlist_alert_email",
        AsyncMock(return_value=(True, "msg-1", None, False)),
    )
    monkeypatch.setattr(watchlist_alert_service, "update_watchlist_alert_email_log", AsyncMock())
    monkeypatch.setattr(watchlist_alert_service, "render_watchlist_alert_email_html", lambda **_kwargs: "<p>body</p>")

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result["_skip_synthesis"] == "Watchlist alert delivery complete"
    assert result["attempts_claimed"] == 1
    assert result["sent"] == 1
    assert watchlist_alert_service.update_watchlist_alert_email_log.await_count == 1
    update_sql = pool.execute.await_args.args[0]
    assert "UPDATE b2b_watchlist_views" in update_sql


@pytest.mark.asyncio
async def test_claim_delivery_attempt_matches_partial_unique_index(monkeypatch):
    now = datetime.now(timezone.utc)
    row = {
        "account_id": uuid4(),
        "id": uuid4(),
        "next_alert_delivery_at": now,
        "alert_delivery_frequency": "daily",
    }
    pool = SimpleNamespace(fetchrow=AsyncMock(return_value={"id": uuid4()}))

    await mod._claim_delivery_attempt(pool, row)

    sql = pool.fetchrow.await_args.args[0]
    assert "ON CONFLICT (watchlist_view_id, scheduled_for, delivery_mode)" in sql
    assert "WHERE scheduled_for IS NOT NULL" in sql
